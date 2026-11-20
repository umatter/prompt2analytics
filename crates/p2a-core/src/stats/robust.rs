//! Robust and extended descriptive statistics.
//!
//! Implements fivenum, IQR, quantile (R's types 1-9), quantiles, percentile_interval,
//! boxplot_stats, mad, ecdf, and density from R stats.
//!
//! ```
//! use p2a_core::stats::robust::{quantiles, QuantileType};
//! let q = quantiles(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[0.25], QuantileType::T6).unwrap();
//! assert_eq!(q, vec![2.75]);
//! ```

use crate::errors::{EconError, EconResult};
use crate::resource;
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

    sorted.sort_by(|a, b| a.total_cmp(b));

    let [minimum, lower_hinge, median, upper_hinge, maximum] = fivenum_sorted(&sorted);

    Ok(FivenumResult {
        minimum,
        lower_hinge,
        median,
        upper_hinge,
        maximum,
        n,
    })
}

/// Compute median of sorted data.
fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

// ============================================================================
// IQR - Interquartile Range
// ============================================================================

/// Compute the interquartile range, R's `IQR(data, type = qtype, na.rm = TRUE)`.
///
/// NaN values are dropped; `qtype` is R's quantile type 1-9 (default 7); any other type is an
/// error. The data are sorted once and both quartiles are read from the sorted copy.
pub fn iqr(data: &[f64], qtype: Option<usize>) -> EconResult<f64> {
    let t = QuantileType::from_r(qtype.unwrap_or(7))?;
    let s = sorted_non_nan_copy(data, "IQR sort buffer")?;
    Ok(quantile_sorted(&s, 0.75, t) - quantile_sorted(&s, 0.25, t))
}

/// Compute a single quantile, R's `quantile(data, p, type = qtype, na.rm = TRUE)`.
///
/// R's quantile types 1-9 (an error for other values); NaN values are dropped.
pub fn quantile(data: &[f64], p: f64, qtype: usize) -> EconResult<f64> {
    let t = QuantileType::from_r(qtype)?;
    if !(0.0..=1.0).contains(&p) {
        return Err(EconError::InvalidSpecification {
            message: format!("p must be in [0, 1], got {p}"),
        });
    }
    Ok(quantile_sorted(
        &sorted_non_nan_copy(data, "quantile sort buffer")?,
        p,
        t,
    ))
}

/// R's sample quantile types (Hyndman & Fan 1996), as `stats::quantile(type = 1..9)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantileType {
    T1,
    T2,
    T3,
    T4,
    T5,
    T6,
    #[default]
    T7,
    T8,
    T9,
}

impl QuantileType {
    /// All nine types in R's order.
    pub const ALL: [QuantileType; 9] = [
        Self::T1,
        Self::T2,
        Self::T3,
        Self::T4,
        Self::T5,
        Self::T6,
        Self::T7,
        Self::T8,
        Self::T9,
    ];
    /// The type for R's `type` argument (1..=9).
    pub fn from_r(t: usize) -> EconResult<Self> {
        match t {
            1..=9 => Ok(Self::ALL[t - 1]),
            _ => Err(EconError::InvalidSpecification {
                message: format!(
                    "quantile type must be an integer from 1 to 9 (R's quantile types), got {t}"
                ),
            }),
        }
    }
    /// R's `type` number.
    pub fn r_type(self) -> usize {
        self as usize + 1
    }
    fn ab(self) -> (f64, f64) {
        match self {
            Self::T4 => (0.0, 1.0),
            Self::T5 => (0.5, 0.5),
            Self::T6 => (0.0, 0.0),
            Self::T7 => (1.0, 1.0),
            Self::T8 => (1.0 / 3.0, 1.0 / 3.0),
            Self::T9 => (3.0 / 8.0, 3.0 / 8.0),
            Self::T1 | Self::T2 | Self::T3 => (0.0, 0.0),
        }
    }
}

/// One sample quantile of sorted, NaN-free data, exactly as R 4.3's `quantile.default`
/// (type 7's interpolation; types 1-3 without fuzz; types 4-9 with R's `fuzz <- 4 * .Machine$double.eps`).
/// `p` is clamped to [0, 1]; empty data or NaN `p` give NaN.
pub fn quantile_sorted(sorted: &[f64], p: f64, qtype: QuantileType) -> f64 {
    let n = sorted.len();
    if n == 0 || p.is_nan() {
        return f64::NAN;
    }
    let p = p.clamp(0.0, 1.0);
    let nf = n as f64;
    if qtype == QuantileType::T7 {
        // index <- 1 + max(n - 1, 0) * probs; lo <- floor(index); hi <- ceiling(index)
        let index = 1.0 + (nf - 1.0) * p;
        let lo = index.floor();
        let hi = index.ceil();
        let qs = sorted[lo as usize - 1];
        let xhi = sorted[hi as usize - 1];
        if index > lo && xhi != qs {
            let h = index - lo;
            return (1.0 - h) * qs + h * xhi;
        }
        return qs;
    }
    let (j, h) = match qtype {
        QuantileType::T1 | QuantileType::T2 | QuantileType::T3 => {
            let nppm = if qtype == QuantileType::T3 {
                nf * p - 0.5
            } else {
                nf * p
            };
            let j = nppm.floor();
            let gt = if nppm > j { 1.0 } else { 0.0 };
            let h = match qtype {
                QuantileType::T1 => gt,
                QuantileType::T2 => (gt + 1.0) / 2.0,
                _ => {
                    if nppm != j || (j as i64).rem_euclid(2) == 1 {
                        1.0
                    } else {
                        0.0
                    }
                }
            };
            (j, h)
        }
        _ => {
            let (a, b) = qtype.ab();
            let fuzz = 4.0 * f64::EPSILON;
            let nppm = a + p * (nf + 1.0 - a - b);
            let j = (nppm + fuzz).floor();
            let mut h = nppm - j;
            if h.abs() < fuzz {
                h = 0.0;
            }
            (j, h)
        }
    };
    // R: x <- c(x[1], x[1], x, x[n], x[n]); qs <- x[j + 2] (1-based), right neighbour x[j + 3]
    let ext = |i1: i64| -> f64 {
        if i1 <= 2 {
            sorted[0]
        } else if i1 >= n as i64 + 3 {
            sorted[n - 1]
        } else {
            sorted[(i1 - 3) as usize]
        }
    };
    let j = j as i64;
    let (left, right) = (ext(j + 2), ext(j + 3));
    if h == 1.0 {
        right
    } else if h > 0.0 && h < 1.0 && left != right {
        (1.0 - h) * left + h * right
    } else {
        left
    }
}

/// The non-NaN values of `data`, sorted with `f64::total_cmp`, in one guarded buffer (R's
/// `na.rm = TRUE`); +/-Inf are kept, as R's quantile does. `EmptyDataset` when nothing is left.
fn sorted_non_nan_copy(data: &[f64], what: &str) -> EconResult<Vec<f64>> {
    let mut s = crate::resource::try_with_capacity::<f64>(data.len(), what)?;
    s.extend(data.iter().copied().filter(|x| !x.is_nan()));
    if s.is_empty() {
        return Err(EconError::EmptyDataset);
    }
    s.sort_by(f64::total_cmp);
    Ok(s)
}

fn sorted_copy(data: &[f64], what: &str) -> EconResult<Vec<f64>> {
    if data.is_empty() {
        return Err(EconError::EmptyDataset);
    }
    if data.iter().any(|x| x.is_nan()) {
        return Err(EconError::InvalidSpecification {
            message: format!("{what}: missing values and NaN are not allowed (R: na.rm = FALSE)"),
        });
    }
    let mut s = crate::resource::try_with_capacity::<f64>(data.len(), "quantile sort buffer")?;
    s.extend_from_slice(data);
    s.sort_by(f64::total_cmp);
    Ok(s)
}

/// `stats::quantile(data, probs, type)`; NaN in `data` is an error (R without `na.rm`), `probs` in [0, 1].
pub fn quantiles(data: &[f64], probs: &[f64], qtype: QuantileType) -> EconResult<Vec<f64>> {
    if let Some(p) = probs.iter().find(|p| !(**p >= 0.0 && **p <= 1.0)) {
        return Err(EconError::InvalidSpecification {
            message: format!("probabilities must be in [0, 1], got {p}"),
        });
    }
    let s = sorted_copy(data, "quantile")?;
    let mut out = crate::resource::try_with_capacity::<f64>(probs.len(), "quantile output")?;
    out.extend(probs.iter().map(|&p| quantile_sorted(&s, p, qtype)));
    Ok(out)
}

/// Bootstrap percentile interval: type 7 quantiles at (1 - level)/2 and (1 + level)/2. Drop failed
/// (NaN) replicates first and report their number; NaN draws are an error here.
pub fn percentile_interval(draws: &[f64], level: f64) -> EconResult<(f64, f64)> {
    crate::traits::estimator::validate_level(level, "level")?;
    let s = sorted_copy(draws, "percentile interval draws")?;
    Ok((
        quantile_sorted(&s, (1.0 - level) / 2.0, QuantileType::T7),
        quantile_sorted(&s, (1.0 + level) / 2.0, QuantileType::T7),
    ))
}

/// `grDevices::boxplot.stats(x, coef)` without notches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoxplotStats {
    /// Lower whisker, lower hinge, median, upper hinge, upper whisker.
    pub stats: [f64; 5],
    /// Non-missing observations.
    pub n: usize,
    /// Values beyond coef * hinge spread, in data order.
    pub out: Vec<f64>,
}

/// R `fivenum` on sorted, non-empty, NaN-free data: 0.5 * (x[floor(d)] + x[ceiling(d)]) at
/// d = 1, n4, (n + 1)/2, n + 1 - n4, n with n4 = floor((n + 3)/2)/2.
fn fivenum_sorted(s: &[f64]) -> [f64; 5] {
    let n = s.len() as f64;
    let n4 = ((n + 3.0) / 2.0).floor() / 2.0;
    [1.0, n4, (n + 1.0) / 2.0, n + 1.0 - n4, n]
        .map(|d| 0.5 * (s[d.floor() as usize - 1] + s[d.ceil() as usize - 1]))
}

/// `grDevices::boxplot.stats`: Tukey hinges (fivenum), whiskers at the most extreme values within
/// coef * (upper hinge - lower hinge), outliers in data order. NaN dropped; coef finite and >= 0.
pub fn boxplot_stats(data: &[f64], coef: f64) -> EconResult<BoxplotStats> {
    if !(coef.is_finite() && coef >= 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("coef must be finite and non-negative, got {coef}"),
        });
    }
    let s = sorted_non_nan_copy(data, "boxplot sort buffer")?;
    let mut kept = crate::resource::try_with_capacity::<f64>(s.len(), "boxplot data")?;
    kept.extend(data.iter().copied().filter(|x| !x.is_nan()));
    let mut stats = fivenum_sorted(&s);
    let mut out = Vec::new();
    if coef > 0.0 {
        let iqr = stats[3] - stats[1];
        let is_out = |x: f64| {
            if iqr.is_nan() {
                !x.is_finite()
            } else {
                x < stats[1] - coef * iqr || x > stats[3] + coef * iqr
            }
        };
        out = kept.iter().copied().filter(|&x| is_out(x)).collect();
        if !out.is_empty() {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for x in kept.iter().copied().filter(|&x| !is_out(x)) {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            stats[0] = lo;
            stats[4] = hi;
        }
    }
    Ok(BoxplotStats {
        stats,
        n: kept.len(),
        out,
    })
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
        s.sort_by(|a, b| a.total_cmp(b));
        median_sorted(&s)
    });

    // Compute absolute deviations from center
    let mut deviations: Vec<f64> = sorted.iter().map(|x| (x - center).abs()).collect();

    deviations.sort_by(|a, b| a.total_cmp(b));

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
    /// Evaluate the ECDF at a point: the proportion of data values `<= point`.
    ///
    /// The step is found by binary search over the sorted knots. A NaN point gives NaN
    /// (R: `ecdf(x)(NA)` is `NA`).
    pub fn evaluate(&self, point: f64) -> f64 {
        if point.is_nan() {
            return f64::NAN;
        }
        if self.x.is_empty() {
            return 0.0;
        }

        // Number of knots <= point; the step value is the last of them
        let i = self.x.partition_point(|&xi| xi <= point);
        if i == 0 { 0.0 } else { self.y[i - 1] }
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

    sorted.sort_by(|a, b| a.total_cmp(b));

    // Get unique values and their cumulative proportions
    let mut x_unique = Vec::new();
    let mut y_cumulative = Vec::new();

    let mut i = 0;
    while i < n {
        let val = sorted[i];
        // Count how many equal values
        let mut count = 1;
        while i + count < n && (sorted[i + count] == val || (sorted[i + count] - val).abs() < 1e-15)
        {
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
    /// R's raised cosine kernel, (1 + cos(pi t / a)) / (2a)
    Cosine,
    /// R's "optcosine" kernel, pi/4 cos(pi t / (2a)) / a
    OptCosine,
}

crate::option_enum!(
    DensityKernel,
    crate::options::Choices {
        option: "kernel",
        values: &[
            (&["gaussian", "normal"], DensityKernel::Gaussian),
            (&["epanechnikov"], DensityKernel::Epanechnikov),
            (&["rectangular", "uniform"], DensityKernel::Rectangular),
            (&["triangular"], DensityKernel::Triangular),
            (&["biweight", "quartic"], DensityKernel::Biweight),
            (&["cosine"], DensityKernel::Cosine),
            (&["optcosine"], DensityKernel::OptCosine),
        ],
    }
);

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

/// Largest evaluation grid [`density`] accepts (`n` in `2..=2^20`, in every mode).
const MAX_DENSITY_POINTS: usize = 1 << 20;

/// Kernel density estimate, as R's `density()`.
///
/// With the `spectral-analysis` feature (all shipped builds) this is R's algorithm for every
/// kernel and sample size: the finite values are linearly binned onto 2m points over
/// [from - 4 bw, to + 4 bw], m = max(n, 512) (rounded up to a power of two above 512), convolved
/// with the kernel by FFT, and linearly interpolated onto `n` points from `from` to `to`. Without
/// the feature the estimate is the exact kernel sum at those points, which differs from R's by
/// R's binning error. Every kernel is scaled so that `bw` is its standard deviation, as in R.
///
/// # Arguments
///
/// * `data` - Input data. NaN values are dropped. Infinite values are dropped the way R
///   does it: the density is computed from the finite values and scaled by their share of
///   the non-NaN values, and `n` reports the non-NaN count.
/// * `bw` - Bandwidth (None for R's `bw.nrd0`, which needs two finite values); a supplied
///   bandwidth must be finite and greater than 0
/// * `kernel` - Kernel function to use
/// * `n_points` - Number of evaluation points, 2 to 1,048,576 (default 512)
/// * `from` - Lower bound of evaluation range (default: min - 3*bw); must be finite
/// * `to` - Upper bound of evaluation range (default: max + 3*bw); must be finite. A default
///   bound that overflows, or a range whose width `to - from` overflows, is refused.
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
    // Grid size, checked before any arithmetic on it
    let n_points_requested = n_points.unwrap_or(512);
    if !(2..=MAX_DENSITY_POINTS).contains(&n_points_requested) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "n must be between 2 and {MAX_DENSITY_POINTS} grid points, got {n_points_requested}"
            ),
        });
    }

    if from.is_some_and(|v| !v.is_finite()) || to.is_some_and(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "from and to must be finite numbers".to_string(),
        });
    }

    // N counts the non-NaN values; the estimate uses the finite ones (R's density)
    let n_total = data.iter().filter(|x| !x.is_nan()).count();
    let n_finite = data.iter().filter(|x| x.is_finite()).count();
    let mut clean: Vec<f64> = resource::try_with_capacity(n_finite, "density data")?;
    clean.extend(data.iter().copied().filter(|x| x.is_finite()));

    let n = clean.len();
    if n == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Bandwidth: a supplied one must be usable; otherwise R's bw.nrd0
    let bw = match bw {
        Some(bw) => {
            if !(bw.is_finite() && bw > 0.0) {
                return Err(EconError::InvalidSpecification {
                    message: format!("bw must be a finite number greater than 0, got {bw}"),
                });
            }
            bw
        }
        None => {
            if n < 2 {
                return Err(EconError::InvalidSpecification {
                    message: "need at least 2 finite values to choose a bandwidth \
                              automatically; supply bw"
                        .to_string(),
                });
            }
            bw_nrd0(&clean)?
        }
    };

    if !(bw.is_finite() && bw > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: "Bandwidth must be positive".to_string(),
        });
    }

    // Determine range
    let data_min = clean.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = clean.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let from = from.unwrap_or(data_min - 3.0 * bw);
    let to = to.unwrap_or(data_max + 3.0 * bw);
    // The default range can overflow for finite data near f64::MAX with a large bw, which would
    // make the grid and the density NaN
    if !(from.is_finite() && to.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "from and to must be finite numbers (the default range min - 3*bw .. \
                      max + 3*bw overflows)"
                .to_string(),
        });
    }
    // Two finite bounds can still be too far apart for the grid step
    if !(to - from).is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!("the range from {from} to {to} is too wide (to - from overflows)"),
        });
    }

    #[cfg(feature = "spectral-analysis")]
    let mut result = density_binned(&clean, bw, kernel, n_points_requested, from, to)?;
    #[cfg(not(feature = "spectral-analysis"))]
    let mut result = density_direct(&clean, bw, kernel, n_points_requested, from, to)?;

    // Infinite values carry mass that no finite grid point can show (R's totMass)
    if n < n_total {
        let mass = n as f64 / n_total as f64;
        for v in &mut result.y {
            *v *= mass;
        }
    }
    result.n = n_total;

    Ok(result)
}

/// R's `density()` on its binned grid: the finite values are linearly binned (weight 1/N each)
/// onto 2m points over [from - 4 bw, to + 4 bw], m = max(n_user, 512) rounded up to a power of
/// two above 512, convolved with the kernel by FFT, and the first m values linearly interpolated
/// onto `n_user` points from `from` to `to`. `n_user` is in `2..=2^20`.
#[cfg(feature = "spectral-analysis")]
fn density_binned(
    clean: &[f64],
    bw: f64,
    kernel: DensityKernel,
    n_user: usize,
    from: f64,
    to: f64,
) -> EconResult<DensityResult> {
    resource::checkpoint()?;
    let m = n_user.max(R_DENSITY_MIN_GRID);
    let m = if m > R_DENSITY_MIN_GRID {
        m.next_power_of_two()
    } else {
        m
    };
    let size = 2 * m;
    let lo = from - 4.0 * bw;
    let up = to + 4.0 * bw;
    if !(lo.is_finite() && up.is_finite() && (up - lo).is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the grid from - 4*bw .. to + 4*bw ({from} - 4*{bw} .. {to} + 4*{bw}) overflows"
            ),
        });
    }

    // BinDist: linear binning, weight 1/N per finite value
    let mut y: Vec<Complex<f64>> = resource::try_vec(size, Complex::new(0.0, 0.0), "density grid")?;
    let w = 1.0 / clean.len() as f64;
    let xdelta = (up - lo) / (m - 1) as f64;
    let ixmax = (m - 2) as f64;
    for &xi in clean {
        let pos = (xi - lo) / xdelta;
        let fl = pos.floor();
        let fx = pos - fl;
        if (0.0..=ixmax).contains(&fl) {
            let ix = fl as usize;
            y[ix].re += w * (1.0 - fx);
            y[ix + 1].re += w * fx;
        } else if fl == -1.0 {
            y[0].re += w * fx;
        } else if fl == ixmax + 1.0 {
            y[m - 1].re += w * (1.0 - fx);
        }
    }

    let mut k = density_kernel_ordinates(up - lo, m, bw, kernel)?;

    resource::checkpoint()?;
    // FftPlanner caches its plans: grids up to DENSITY_CACHED_FFT_MAX reuse a per-thread planner
    // (the default 512-point grid among them), larger ones plan afresh so no big plan is kept.
    let plan = |planner: &mut FftPlanner<f64>| {
        (
            planner.plan_fft_forward(size),
            planner.plan_fft_inverse(size),
        )
    };
    let (fft, ifft) = if size <= DENSITY_CACHED_FFT_MAX {
        DENSITY_FFT_PLANNER.with(|planner| plan(&mut planner.borrow_mut()))
    } else {
        plan(&mut FftPlanner::new())
    };
    fft.process(&mut y);
    fft.process(&mut k);
    for (a, b) in y.iter_mut().zip(&k) {
        *a *= b.conj();
    }
    drop(k);
    ifft.process(&mut y);
    let scale = 1.0 / size as f64;
    // kords: the convolution on xords = seq(lo, up, length m), floored at 0
    let kords = |i: usize| (y[i].re * scale).max(0.0);

    // approx(xords, kords, x) at x = seq(from, to, length n_user), xords = seq(lo, up, length m)
    let xords_at = RSeq::new(lo, up, m);
    let xords = |j: usize| xords_at.at(j);
    let x_at = RSeq::new(from, to, n_user);
    let mut xs: Vec<f64> = resource::try_with_capacity(n_user, "density grid")?;
    let mut ys: Vec<f64> = resource::try_with_capacity(n_user, "density values")?;
    for i in 0..n_user {
        let v = x_at.at(i);
        // the interval [xords(j), xords(j + 1)) holding v, as approx's bisection finds it
        let mut j = ((v - lo) / xdelta).floor().clamp(0.0, (m - 2) as f64) as usize;
        while j > 0 && v < xords(j) {
            j -= 1;
        }
        while j < m - 2 && v >= xords(j + 1) {
            j += 1;
        }
        let (x0, x1) = (xords(j), xords(j + 1));
        let (y0, y1) = (kords(j), kords(j + 1));
        let val = if v == x1 {
            y1
        } else if v == x0 {
            y0
        } else {
            y0 + (y1 - y0) * ((v - x0) / (x1 - x0))
        };
        xs.push(v);
        ys.push(val);
    }

    Ok(DensityResult {
        x: xs,
        y: ys,
        bw,
        n: clean.len(),
        kernel,
    })
}

/// The largest FFT length whose plans the per-thread density planner keeps.
#[cfg(feature = "spectral-analysis")]
const DENSITY_CACHED_FFT_MAX: usize = 1 << 13;

#[cfg(feature = "spectral-analysis")]
thread_local! {
    static DENSITY_FFT_PLANNER: std::cell::RefCell<FftPlanner<f64>> =
        std::cell::RefCell::new(FftPlanner::new());
}

/// R's smallest density grid (`density(n = 512)`); larger grids are rounded up to a power of two.
#[cfg(feature = "spectral-analysis")]
const R_DENSITY_MIN_GRID: usize = 512;

/// R's `seq.int(from, to, length.out = len)`, `len >= 2`: the ends exactly, the first half
/// counted up from `from` and the second half down from `to`.
#[derive(Clone, Copy)]
struct RSeq {
    from: f64,
    to: f64,
    last: usize,
    by: f64,
}

impl RSeq {
    fn new(from: f64, to: f64, len: usize) -> Self {
        let last = len - 1;
        RSeq {
            from,
            to,
            last,
            by: (to - from) / last as f64,
        }
    }

    /// Element `i` (0-based).
    #[inline]
    fn at(&self, i: usize) -> f64 {
        if i == 0 {
            self.from
        } else if i == self.last {
            self.to
        } else if i < self.last.div_ceil(2) {
            // i < len / 2
            self.from + i as f64 * self.by
        } else {
            self.to - (self.last - i) as f64 * self.by
        }
    }
}

/// The exact kernel sum mean(K(x0 - x_i)) at `n_points` points from `from` to `to` (builds
/// without `spectral-analysis`; R's kernels). `n_points` is in `2..=2^20`; each grid point
/// honours cancellation.
#[cfg_attr(feature = "spectral-analysis", allow(dead_code))]
pub(crate) fn density_direct(
    data: &[f64],
    bw: f64,
    kernel: DensityKernel,
    n_points: usize,
    from: f64,
    to: f64,
) -> EconResult<DensityResult> {
    let n = data.len();
    let mut x: Vec<f64> = resource::try_with_capacity(n_points, "density grid")?;
    let x_at = RSeq::new(from, to, n_points);
    x.extend((0..n_points).map(|i| x_at.at(i)));

    let mut y: Vec<f64> = resource::try_with_capacity(n_points, "density values")?;
    for &xi in &x {
        resource::checkpoint()?;
        let sum: f64 = data
            .iter()
            .map(|&xj| kernel_function(xi - xj, bw, kernel))
            .sum();
        y.push(sum / n as f64);
    }

    Ok(DensityResult {
        x,
        y,
        bw,
        n,
        kernel,
    })
}

/// R's `bw.nrd0`: 0.9 min(sd, IQR/1.34) n^(-1/5), with the type-7 IQR. When that minimum is 0,
/// sd is used, then |x[0]| (the first value), then 1, so the bandwidth is never 0. Needs at least
/// two values, all finite.
pub(crate) fn bw_nrd0(x: &[f64]) -> EconResult<f64> {
    let n = x.len();
    if n < 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("bw.nrd0 needs at least 2 data points, got {n}"),
        });
    }
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let hi = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0)).sqrt();
    let sorted = sorted_copy(x, "bandwidth data")?;
    let iqr = quantile_sorted(&sorted, 0.75, QuantileType::T7)
        - quantile_sorted(&sorted, 0.25, QuantileType::T7);
    let mut lo = hi.min(iqr / 1.34);
    if lo == 0.0 {
        lo = hi;
        if lo == 0.0 {
            lo = x[0].abs();
            if lo == 0.0 {
                lo = 1.0;
            }
        }
    }
    Ok(0.9 * lo * nf.powf(-0.2))
}

/// The kernel ordinates of [`density_binned`] at kords = seq(0, 2 * span, length 2m), mirrored
/// above m (R's `kords[(m + 2):(2m)] <- -kords[m:2]`). Every kernel is even, so the ordinates
/// above m are those below it in reverse order and are copied rather than evaluated again.
#[cfg(feature = "spectral-analysis")]
fn density_kernel_ordinates(
    span: f64,
    m: usize,
    bw: f64,
    kernel: DensityKernel,
) -> EconResult<Vec<Complex<f64>>> {
    let size = 2 * m;
    let kords_at = RSeq::new(0.0, 2.0 * span, size);
    let mut k: Vec<Complex<f64>> = resource::try_with_capacity(size, "density kernel grid")?;
    k.extend((0..=m).map(|i| Complex::new(kernel_function(kords_at.at(i), bw, kernel), 0.0)));
    for i in m + 1..size {
        let mirrored = k[size - i];
        k.push(mirrored);
    }
    Ok(k)
}

/// R's density kernels at distance `t` with bandwidth `bw` (the kernel's standard deviation),
/// including the 1/bw scaling. The compact kernels are zero at |t| >= a, R's half-width.
fn kernel_function(t: f64, bw: f64, kernel: DensityKernel) -> f64 {
    use std::f64::consts::PI;
    let ax = t.abs();
    let inside = |a: f64, f: &dyn Fn(f64) -> f64| if ax < a { f(a) } else { 0.0 };
    match kernel {
        DensityKernel::Gaussian => {
            let u = t / bw;
            (-0.5 * u * u).exp() / ((2.0 * PI).sqrt() * bw)
        }
        DensityKernel::Epanechnikov => inside(bw * 5f64.sqrt(), &|a| {
            let u = ax / a;
            0.75 * (1.0 - u * u) / a
        }),
        DensityKernel::Rectangular => inside(bw * 3f64.sqrt(), &|a| 0.5 / a),
        DensityKernel::Triangular => inside(bw * 6f64.sqrt(), &|a| (1.0 - ax / a) / a),
        DensityKernel::Biweight => inside(bw * 7f64.sqrt(), &|a| {
            let u = ax / a;
            let w = 1.0 - u * u;
            15.0 / 16.0 * w * w / a
        }),
        DensityKernel::Cosine => inside(bw / (1.0 / 3.0 - 2.0 / (PI * PI)).sqrt(), &|a| {
            (1.0 + (PI * t / a).cos()) / (2.0 * a)
        }),
        DensityKernel::OptCosine => inside(bw / (1.0 - 8.0 / (PI * PI)).sqrt(), &|a| {
            PI / 4.0 * (PI * t / (2.0 * a)).cos() / a
        }),
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
    let kernel: DensityKernel = kernel.parse()?;

    density(data, bw, kernel, n_points, None, None)
}

#[cfg(test)]
mod tests {

    /// The mirrored kernel ordinates equal the ones evaluated at -kords, as before, for every
    /// kernel, on the default grid and a larger one, at bandwidths that put the support inside
    /// and outside the grid (noisy spans and bandwidths, not round numbers).
    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_density_kernel_mirroring_matches_direct_evaluation() {
        let kernels = [
            DensityKernel::Gaussian,
            DensityKernel::Epanechnikov,
            DensityKernel::Rectangular,
            DensityKernel::Triangular,
            DensityKernel::Biweight,
            DensityKernel::Cosine,
            DensityKernel::OptCosine,
        ];
        for &kernel in &kernels {
            for &(span, m, bw) in &[
                (13.71, 512usize, 0.413),
                (3.2e-3, 512, 7.9e-5),
                (941.3, 2048, 31.7),
                (5.17, 1024, 12.3),
            ] {
                let size = 2 * m;
                let kords_at = RSeq::new(0.0, 2.0 * span, size);
                let direct: Vec<f64> = (0..size)
                    .map(|i| {
                        let t = if i <= m {
                            kords_at.at(i)
                        } else {
                            -kords_at.at(size - i)
                        };
                        kernel_function(t, bw, kernel)
                    })
                    .collect();
                let mirrored = density_kernel_ordinates(span, m, bw, kernel).unwrap();
                assert_eq!(mirrored.len(), size);
                for (i, (a, b)) in mirrored.iter().zip(&direct).enumerate() {
                    assert_eq!(a.im, 0.0);
                    assert!(
                        a.re == *b || (a.re - b).abs() <= 4.0 * f64::EPSILON * b.abs(),
                        "{kernel:?} span {span} m {m} bw {bw} i {i}: {} vs {b}",
                        a.re
                    );
                }
            }
        }
    }
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    // The DX set of case stat_density_t4
    const DX: [f64; 30] = [
        1.21, 0.37, -0.45, 2.08, 1.66, 0.92, -1.13, 0.05, 1.48, 2.61, 0.74, -0.28, 1.02, 3.05,
        0.51, -0.66, 1.87, 1.33, 0.19, 2.34, -0.91, 0.88, 1.55, 0.62, 2.79, -0.12, 1.11, 0.43,
        1.94, 0.27,
    ];
    /// The column `x` of set `set` of case `stat_density_t4`.
    fn density_set(ds: &crate::data::Dataset, set: &str) -> Vec<f64> {
        crate::stats::ranks::test_data::column_where(ds, "x", &[("set", set)])
    }

    const ALL_KERNELS: [(DensityKernel, &str); 7] = [
        (DensityKernel::Gaussian, "gaussian"),
        (DensityKernel::Epanechnikov, "epanechnikov"),
        (DensityKernel::Rectangular, "rectangular"),
        (DensityKernel::Triangular, "triangular"),
        (DensityKernel::Biweight, "biweight"),
        (DensityKernel::Cosine, "cosine"),
        (DensityKernel::OptCosine, "optcosine"),
    ];

    /// The exact kernel sum (the path of builds without spectral-analysis) against R's kernels
    /// evaluated directly: mean(K((x0 - x)/bw))/bw at R's density grid.
    #[test]
    fn test_validate_density_direct_path_against_r() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_density_t4");
        let mut check = Check::new(&case);
        let dx = density_set(&ds, "dx");
        assert_eq!(dx, DX);
        let d = density_direct(&dx, 0.4, DensityKernel::Gaussian, 50, -2.33, 4.25).unwrap();
        check
            .vector("dx_n50_x", &d.x, Tol::coef(30))
            .vector("kde_dx_n50_y", &d.y, Tol::coef(30));
        let x9 = density_set(&ds, "x9");
        let x9_x = case.vector("x9_x");
        for (k, name) in ALL_KERNELS {
            let d = density_direct(&x9, 0.5, k, 7, x9_x[0], x9_x[6]).unwrap();
            check.vector(&format!("kde_x9_{name}_y"), &d.y, Tol::coef(9));
        }
        let big = density_set(&ds, "big");
        let d = density_direct(&big, 0.4, DensityKernel::Gaussian, 100, -3.0, 3.0).unwrap();
        check
            .vector("big_ft_x", &d.x, Tol::coef(200))
            .vector("kde_big_ft_y", &d.y, Tol::coef(200));
        check.finish();
    }

    /// R's binned grid for grid sizes that are not powers of two, with from/to, the default
    /// bandwidth, IQR = 0 and infinite values. Fails before: a 100-point grid was the first 100
    /// of 128 FFT values labelled over [from, to]; IQR = 0 was an error.
    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_validate_density_grid_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_density_t4");
        let mut check = Check::new(&case);
        let gauss = DensityKernel::Gaussian;
        let big = density_set(&ds, "big");
        for n in [100usize, 512, 1000] {
            let d = density(&big, Some(0.4), gauss, Some(n), None, None).unwrap();
            check
                .vector(&format!("big_n{n}_x"), &d.x, Tol::coef(200))
                .vector(&format!("big_n{n}_y"), &d.y, Tol::coef(200));
        }
        let d = density(&big, Some(0.4), gauss, Some(100), Some(-3.0), Some(3.0)).unwrap();
        check
            .vector("big_ft_x", &d.x, Tol::coef(200))
            .vector("big_ft_y", &d.y, Tol::coef(200));
        let d = density(&big, None, gauss, None, None, None).unwrap();
        check.scalar("big_default_bw", d.bw, Tol::stat(200)).vector(
            "big_default_y",
            &d.y,
            Tol::coef(200),
        );
        let iqr0 = density_set(&ds, "iqr0");
        let d = density(&iqr0, None, gauss, None, None, None).unwrap();
        check.vector("iqr0_default_x", &d.x, Tol::coef(22)).vector(
            "iqr0_default_y",
            &d.y,
            Tol::coef(22),
        );
        let d = density(&DX, Some(0.4), gauss, Some(50), None, None).unwrap();
        check
            .vector("dx_n50_x", &d.x, Tol::coef(30))
            .vector("dx_n50_y", &d.y, Tol::coef(30));
        let mut inf = DX.to_vec();
        inf.extend([f64::INFINITY, f64::NEG_INFINITY]);
        let d = density(&inf, Some(0.4), gauss, Some(50), None, None).unwrap();
        check.vector("dx_inf_n50_y", &d.y, Tol::coef(30)).scalar(
            "dx_inf_n",
            d.n as f64,
            Tol::exact(),
        );
        let d = density(
            &DX,
            Some(0.4),
            DensityKernel::Epanechnikov,
            Some(50),
            None,
            None,
        )
        .unwrap();
        check.vector("dx_epan_n50_y", &d.y, Tol::coef(30));
        check.finish();
    }

    /// Every kernel with bw as its standard deviation; cosine is R's raised cosine.
    /// Fails before: compact kernels had half-width bw; cosine was R's optcosine.
    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_validate_density_kernels_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_density_t4");
        let mut check = Check::new(&case);
        let big = density_set(&ds, "big");
        let x9 = density_set(&ds, "x9");
        for (k, name) in ALL_KERNELS {
            if k != DensityKernel::Gaussian {
                let d = density(&big, Some(0.4), k, Some(100), None, None).unwrap();
                check.vector(&format!("big_{name}_y"), &d.y, Tol::coef(200));
            }
            let d = density(&x9, Some(0.5), k, Some(7), None, None).unwrap();
            check.vector(&format!("x9_{name}_y"), &d.y, Tol::coef(9));
            if k == DensityKernel::Gaussian {
                check.vector("x9_x", &d.x, Tol::coef(9));
            }
        }
        check.finish();
    }

    /// bw.nrd0 and its fallbacks (IQR = 0: sd; sd = 0: |x[1]|; x[1] = 0: 1), directly and as
    /// density's default. Fails before: quartiles by index, an error on IQR = 0.
    #[test]
    fn test_validate_bw_nrd0_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_density_t4");
        let mut check = Check::new(&case);
        for set in ["big", "x9", "ties", "iqr0", "const", "zero"] {
            let x = density_set(&ds, set);
            let n = x.len();
            let bw = bw_nrd0(&x).unwrap();
            check.scalar(&format!("nrd0_{set}"), bw, Tol::stat(n));
            let d = density(&x, None, DensityKernel::Gaussian, Some(64), None, None).unwrap();
            assert_eq!(d.bw, bw, "{set}");
        }
        check.finish();
        assert!(bw_nrd0(&[1.0]).is_err());
    }

    /// stats_density's defaults (gaussian, n = 512, bw.nrd0) on the descriptive_na columns with
    /// the missing values dropped.
    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_validate_density_dataset_na_t4() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_density_mcp_na_t4");
        let mut check = Check::new(&case);
        for col in ["v", "vi"] {
            let (x, _) = numeric_column(ds.df(), col, NaPolicy::Omit).unwrap();
            let n = x.len();
            let d = run_density(&x, None, "gaussian", Some(512)).unwrap();
            check
                .scalar(&format!("dens_{col}_bw"), d.bw, Tol::stat(n))
                .vector(&format!("dens_{col}_x"), &d.x, Tol::coef(n))
                .vector(&format!("dens_{col}_y"), &d.y, Tol::coef(n));
        }
        check.finish();
    }

    #[test]
    fn test_density_rejects_grid_size_outside_bounds() {
        // pre-fix: n = 0 underflows `n_points - 1` (debug panic); n = 1 returns [NaN]; 2^20 + 1 is accepted
        for n in [0usize, 1, (1 << 20) + 1, usize::MAX] {
            let e =
                density(&DX, Some(0.4), DensityKernel::Gaussian, Some(n), None, None).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("1048576")),
                "n={n}: {e}"
            );
        }
        assert!(density(&DX, Some(0.4), DensityKernel::Gaussian, Some(2), None, None).is_ok());
    }

    #[test]
    fn test_density_rejects_bad_bandwidth_and_range() {
        for bw in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let e =
                density(&DX, Some(bw), DensityKernel::Gaussian, Some(50), None, None).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("bw")),
                "bw={bw}: {e}"
            );
        }
        for (from, to) in [(Some(f64::NAN), None), (None, Some(f64::INFINITY))] {
            assert!(matches!(
                density(&DX, Some(0.4), DensityKernel::Gaussian, Some(50), from, to),
                Err(EconError::InvalidSpecification { .. })
            ));
        }
        // automatic bandwidth needs two finite values (pre-fix: Silverman gives NaN and the call succeeds)
        let e = density(
            &[1.0, f64::INFINITY],
            None,
            DensityKernel::Gaussian,
            Some(50),
            None,
            None,
        )
        .unwrap_err();
        assert!(e.to_string().contains("bw"), "{e}");
    }

    #[test]
    fn test_density_rejects_overflowing_default_range() {
        // min - 3*bw and max + 3*bw overflow to -inf/+inf for finite data and a finite bw.
        // Pre-fix the grid and the density were NaN and the call returned Ok.
        for (data, from, to) in [
            (vec![1e308, 1.5e308], None, None),
            (vec![1e308, 1.5e308], Some(0.0), None),
            (vec![-1e308, -1.5e308], None, Some(0.0)),
            // two finite bounds whose difference overflows
            (vec![0.0, 1.0], Some(-1e308), Some(1e308)),
        ] {
            let e = density(
                &data,
                Some(1e308),
                DensityKernel::Gaussian,
                Some(16),
                from,
                to,
            )
            .unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("overflows")),
                "{data:?}: {e}"
            );
        }
        // The same data with a range that fits still works
        let r = density(
            &[1e308, 1.5e308],
            Some(1e306),
            DensityKernel::Gaussian,
            Some(16),
            None,
            None,
        )
        .unwrap();
        assert!(r.x.iter().chain(&r.y).all(|v| v.is_finite()));
    }

    #[test]
    fn test_density_drops_infinite_values_like_r() {
        // R: density(c(dx, Inf, -Inf), bw = 0.4, n = 50) has d$x unchanged, y scaled by 30/32 and n = 32
        let mut data = DX.to_vec();
        data.extend([f64::INFINITY, f64::NEG_INFINITY, f64::NAN]); // NaN is dropped and not counted, as in R with na.rm
        let gauss = DensityKernel::Gaussian;
        let d = density(&data, Some(0.4), gauss, Some(50), None, None).unwrap();
        let finite = density(&DX, Some(0.4), gauss, Some(50), None, None).unwrap();
        assert_eq!(d.n, 32);
        assert_eq!(d.x, finite.x);
        for i in 0..50 {
            assert!((d.y[i] - finite.y[i] * 30.0 / 32.0).abs() < 1e-15, "y[{i}]");
        }
    }

    #[test]
    fn test_density_honours_cancellation() {
        let t = CancelToken::new();
        t.cancel();
        let r = resource::with_budget(Budget::new(None, t), || {
            density(
                &DX,
                Some(0.4),
                DensityKernel::Epanechnikov,
                Some(50),
                None,
                None,
            )
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    // R 4.3.3: ex <- c(2.3,-0.7,1.1,2.3,0.4,-0.7,3.9,1.1,1.1,0.0,-2.5,0.4); Fn <- ecdf(ex); Fn(at)
    #[test]
    fn test_validate_ecdf_against_r() {
        let ex = [
            2.3, -0.7, 1.1, 2.3, 0.4, -0.7, 3.9, 1.1, 1.1, 0.0, -2.5, 0.4,
        ];
        let at = [
            -3.0, -2.5, -1.0, -0.7, 0.0, 0.2, 0.4, 1.1, 1.5, 2.3, 3.9, 4.5,
        ];
        let r_at = [
            0.0,
            1.0 / 12.0,
            1.0 / 12.0,
            3.0 / 12.0,
            4.0 / 12.0,
            4.0 / 12.0,
            6.0 / 12.0,
            9.0 / 12.0,
            9.0 / 12.0,
            11.0 / 12.0,
            1.0,
            1.0,
        ];
        let f = ecdf(&ex).unwrap();
        assert_eq!(f.x, vec![-2.5, -0.7, 0.0, 0.4, 1.1, 2.3, 3.9]); // R: knots(Fn)
        for (got, want) in f.evaluate_many(&at).iter().zip(r_at) {
            assert!((got - want).abs() < 1e-15);
        }
    }

    #[test]
    fn test_ecdf_evaluate_nan_is_nan() {
        // R: ecdf(x)(NA) is NA; pre-fix returned 1.0
        assert!(ecdf(&[1.0, 2.0, 3.0]).unwrap().evaluate(f64::NAN).is_nan());
    }

    #[test]
    fn test_ecdf_evaluate_matches_linear_scan() {
        let data: Vec<f64> = (0..500)
            .map(|i| ((i * 7919) % 103) as f64 * 0.25 - 10.0)
            .collect(); // many ties
        let f = ecdf(&data).unwrap();
        let naive = |p: f64| match f.x.iter().position(|&xi| xi > p) {
            Some(0) => 0.0,
            Some(i) => f.y[i - 1],
            None => 1.0,
        };
        let mut pts: Vec<f64> = f.x.clone();
        pts.extend(f.x.iter().map(|v| v + 0.125));
        pts.extend([f64::NEG_INFINITY, f64::INFINITY, -1e300, 1e300]);
        for p in pts {
            assert_eq!(f.evaluate(p), naive(p), "p={p}");
        }
    }

    #[test]
    fn test_ecdf_groups_equal_infinities() {
        // R: knots(ecdf(c(1, Inf, Inf))) = c(1, Inf)
        let f = ecdf(&[1.0, f64::INFINITY, f64::INFINITY]).unwrap();
        assert_eq!(f.x, vec![1.0, f64::INFINITY]);
        assert_eq!(f.y, vec![1.0 / 3.0, 1.0]);
    }

    #[cfg(feature = "spectral-analysis")]
    #[test]
    fn test_density_fft_grid_respects_allocation_ceiling() {
        // the binned grid is 2 x 512 complex values = 16384 bytes > 4096
        let data: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.37).sin() * 3.0 + i as f64 * 0.01)
            .collect();
        let r = resource::with_alloc_ceiling(Some(4096), || {
            density(
                &data,
                Some(0.4),
                DensityKernel::Gaussian,
                Some(512),
                None,
                None,
            )
        });
        assert!(matches!(r, Err(EconError::ResourceLimit(_))));
    }

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
    fn test_fivenum_tolerates_nan() {
        // Regression guard: the sort used to panic on NaN input.
        let data = vec![1.0, f64::NAN, 3.0, 5.0, 7.0, 9.0];
        let result = fivenum(&data).unwrap();
        assert_eq!(result.minimum, 1.0);
        assert_eq!(result.maximum, 9.0);
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

    #[test]
    fn test_quantile_names_bad_p_and_drops_nan() {
        let m = spec_message(quantile(&[1.0, 2.0, 3.0], 1.5, 7));
        assert!(m.contains("got 1.5"), "{m}");
        let d = [4.2, f64::NAN, 1.1, 3.3, f64::NAN, 2.7];
        assert!((quantile(&d, 0.5, 7).unwrap() - 3.0).abs() < 1e-12);
        assert!((iqr(&d, None).unwrap() - (3.525 - 2.3)).abs() < 1e-12);
        assert!(matches!(
            quantile(&[f64::NAN, f64::NAN], 0.5, 7),
            Err(EconError::EmptyDataset)
        ));
        let b = boxplot_stats(&[9.0, f64::NAN, 1.0, 1.2, 0.9, 1.1], 1.5).unwrap();
        assert_eq!((b.n, b.out.as_slice()), (5, &[9.0][..]));
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    // -------------------------------------------------------------------------
    // Quantile types 1-9, fivenum, boxplot.stats and percentile intervals
    // (case stat_quantile_types: x has n = 28 with ties, y has n = 7)
    // -------------------------------------------------------------------------

    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Case, Check, Tol};

    /// Called as `sample("stat_quantile_types")`, so each registered test names its case
    /// id in its own body (validation_status.py rule registered-test-loads-no-case).
    fn sample(id: &str) -> (Case, Vec<f64>, Vec<f64>) {
        let (case, ds) = crate::test_support::load_case(id);
        let (x, _) = numeric_column(ds.df(), "x", NaPolicy::Fail).unwrap();
        let (y, miss) = numeric_column(ds.df(), "y", NaPolicy::Omit).unwrap();
        assert_eq!((x.len(), y.len(), miss.n_dropped), (28, 7, 21));
        (case, x, y)
    }

    fn one_to(n: usize) -> Vec<f64> {
        (1..=n).map(|i| i as f64).collect()
    }

    fn spec_message<T: std::fmt::Debug>(r: EconResult<T>) -> String {
        match r {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    /// The legacy `quantile(data, p, type)` API for every R type (STAT-36: types 2-6, 8 and 9
    /// used to fall back to type 7).
    #[test]
    fn test_validate_quantile_legacy_api_types() {
        let (case, x, y) = sample("stat_quantile_types");
        let probs = case.vector("probs");
        let s10 = one_to(10);
        let mut check = Check::new(&case);
        let mut seq10 = Vec::new();
        for t in 1..=9 {
            let gx: Vec<f64> = probs.iter().map(|&p| quantile(&x, p, t).unwrap()).collect();
            let gy: Vec<f64> = probs.iter().map(|&p| quantile(&y, p, t).unwrap()).collect();
            check.vector(&format!("x_type{t}"), &gx, Tol::coef(28));
            check.vector(&format!("y_type{t}"), &gy, Tol::coef(7));
            seq10.push(quantile(&s10, 0.25, t).unwrap());
        }
        check.vector("seq10_q25", &seq10, Tol::coef(10));
        check.finish();
    }

    #[test]
    fn test_validate_iqr() {
        let (case, x, _) = sample("stat_quantile_types");
        let mut check = Check::new(&case);
        for t in 1..=9 {
            check.scalar(
                &format!("x_iqr_type{t}"),
                iqr(&x, Some(t)).unwrap(),
                Tol::coef(28),
            );
        }
        check.finish();
        assert_eq!(iqr(&x, None).unwrap(), iqr(&x, Some(7)).unwrap());
    }

    #[test]
    fn test_validate_quantile_rejects_type_0_and_12() {
        let (_case, x, _) = sample("stat_quantile_types");
        for t in [0, 10, 12, usize::MAX] {
            for m in [
                spec_message(quantile(&x, 0.25, t)),
                spec_message(iqr(&x, Some(t))),
                spec_message(QuantileType::from_r(t)),
            ] {
                assert!(m.contains("from 1 to 9"), "type {t}: {m}");
            }
        }
    }

    #[test]
    fn test_validate_quantile_types_1_9() {
        let (case, x, y) = sample("stat_quantile_types");
        let probs = case.vector("probs");
        let s10 = one_to(10);
        let mut check = Check::new(&case);
        let mut seq10 = Vec::new();
        for t in 1..=9 {
            let qt = QuantileType::from_r(t).unwrap();
            check.vector(
                &format!("x_type{t}"),
                &quantiles(&x, &probs, qt).unwrap(),
                Tol::coef(28),
            );
            check.vector(
                &format!("y_type{t}"),
                &quantiles(&y, &probs, qt).unwrap(),
                Tol::coef(7),
            );
            seq10.push(quantile_sorted(&s10, 0.25, QuantileType::ALL[t - 1]));
        }
        check.vector("seq10_q25", &seq10, Tol::coef(10));
        check.finish();
    }

    /// Types 4-9 at p = (k - a)/(n + 1 - a - b), where a + p (n + 1 - a - b) lands on k only up
    /// to rounding; R's 4 * eps fuzz picks the order statistic k. Exact: the answer is a data
    /// value.
    #[test]
    fn test_validate_quantile_fuzz_rule() {
        let (case, x, _) = sample("stat_quantile_types");
        let mut check = Check::new(&case);
        for (t, qt) in [
            (4, QuantileType::T4),
            (5, QuantileType::T5),
            (6, QuantileType::T6),
            (8, QuantileType::T8),
            (9, QuantileType::T9),
        ] {
            let probs = case.vector(&format!("fuzz_type{t}_probs"));
            check.vector(
                &format!("x_fuzz_type{t}"),
                &quantiles(&x, &probs, qt).unwrap(),
                Tol::exact(),
            );
        }
        // Seven values of -1e20 below 1..21: with the fuzz, type 9 at this p is x_(8) = 1;
        // without it (j = 7, h = 1 - 9e-16) the result would be -88816.841970012523.
        let mut gap = vec![-1e20; 7];
        gap.extend(one_to(21));
        let p = case.vector("gap_type9_p")[0];
        check.scalar(
            "gap_type9_q",
            quantile_sorted(&gap, p, QuantileType::T9),
            Tol::exact(),
        );
        check.finish();
    }

    #[test]
    fn test_quantile_type_from_r_round_trip() {
        for t in 1..=9 {
            assert_eq!(QuantileType::from_r(t).unwrap().r_type(), t);
        }
        assert_eq!(QuantileType::default(), QuantileType::T7);
        assert_eq!(QuantileType::ALL.len(), 9);
    }

    #[test]
    fn test_validate_quantiles_refuses_nan_and_bad_probabilities() {
        let (_case, x, _) = sample("stat_quantile_types");
        let t7 = QuantileType::T7;
        assert!(spec_message(quantiles(&[1.0, f64::NAN], &[0.5], t7)).contains("NaN"));
        for p in [-0.1, 1.1, f64::NAN] {
            let m = spec_message(quantiles(&x, &[p], t7));
            assert!(m.contains("probabilities"), "{p}: {m}");
        }
        assert!(matches!(
            quantiles(&[], &[0.5], t7),
            Err(EconError::EmptyDataset)
        ));
        assert!(quantile_sorted(&[], 0.5, t7).is_nan());
        assert_eq!(quantile_sorted(&[3.0], 0.7, QuantileType::T4), 3.0);
        // R: quantile(c(1, Inf), c(0, 0.5, 1)) is 1, Inf, Inf
        assert_eq!(
            quantiles(&[1.0, f64::INFINITY], &[0.0, 0.5, 1.0], t7).unwrap(),
            vec![1.0, f64::INFINITY, f64::INFINITY]
        );
    }

    fn fivenum_vec(f: &FivenumResult) -> Vec<f64> {
        vec![f.minimum, f.lower_hinge, f.median, f.upper_hinge, f.maximum]
    }

    #[test]
    fn test_validate_fivenum() {
        let (case, x, y) = sample("stat_quantile_types");
        Check::new(&case)
            .vector(
                "x_fivenum",
                &fivenum_vec(&fivenum(&x).unwrap()),
                Tol::coef(28),
            )
            .vector(
                "y_fivenum",
                &fivenum_vec(&fivenum(&y).unwrap()),
                Tol::coef(7),
            )
            .finish();
    }

    #[test]
    fn test_validate_boxplot_stats() {
        let (case, x, y) = sample("stat_quantile_types");
        let b = boxplot_stats(&x, 1.5).unwrap();
        let b05 = boxplot_stats(&x, 0.5).unwrap();
        let by = boxplot_stats(&y, 1.5).unwrap();
        let b62 = boxplot_stats(&[1.0, 7.0, 12.0, 15.0, 15.0, 16.0, 18.0, 18.0], 1.5).unwrap();
        let b0 = boxplot_stats(&x, 0.0).unwrap();
        Check::new(&case)
            .vector("x_boxplot_stats", &b.stats, Tol::coef(28))
            .scalar("x_boxplot_n", b.n as f64, Tol::exact())
            .scalar("x_boxplot_n_out", b.out.len() as f64, Tol::exact())
            .vector("x_boxplot05_stats", &b05.stats, Tol::coef(28))
            .vector("x_boxplot05_out", &b05.out, Tol::exact())
            .vector("y_boxplot_stats", &by.stats, Tol::coef(7))
            .scalar("y_boxplot_n_out", by.out.len() as f64, Tol::exact())
            .vector("data62_boxplot_stats", &b62.stats, Tol::coef(8))
            .scalar("data62_boxplot_n_out", b62.out.len() as f64, Tol::exact())
            .vector("x_boxplot_stats", &b0.stats, Tol::coef(28))
            .finish();
        assert!(b0.out.is_empty());
        assert_eq!(boxplot_stats(&[x[0], f64::NAN, x[1]], 1.5).unwrap().n, 2);
    }

    #[test]
    fn test_boxplot_stats_rejects_bad_coef_and_empty() {
        for coef in [-1.0, f64::NAN, f64::INFINITY] {
            let m = spec_message(boxplot_stats(&[1.0, 2.0, 3.0], coef));
            assert!(m.contains("coef"), "{coef}: {m}");
        }
        assert!(matches!(
            boxplot_stats(&[], 1.5),
            Err(EconError::EmptyDataset)
        ));
        assert!(matches!(
            boxplot_stats(&[f64::NAN], 1.5),
            Err(EconError::EmptyDataset)
        ));
    }

    #[test]
    fn test_validate_percentile_interval() {
        let (case, x, y) = sample("stat_quantile_types");
        let mut check = Check::new(&case);
        for (lv, tag) in [(0.95, "95"), (0.9, "90"), (0.5, "50")] {
            let (a, b) = percentile_interval(&x, lv).unwrap();
            check.vector(&format!("x_pct_{tag}"), &[a, b], Tol::coef(28));
            let (a, b) = percentile_interval(&y, lv).unwrap();
            check.vector(&format!("y_pct_{tag}"), &[a, b], Tol::coef(7));
        }
        check.finish();
    }

    #[test]
    fn test_percentile_interval_rejects_bad_level_and_nan() {
        for lv in [0.0, 1.0, 95.0, f64::NAN] {
            let m = spec_message(percentile_interval(&[1.0, 2.0, 3.0], lv));
            assert!(m.contains("level"), "{lv}: {m}");
        }
        assert!(spec_message(percentile_interval(&[1.0, f64::NAN], 0.95)).contains("NaN"));
        assert!(matches!(
            percentile_interval(&[], 0.95),
            Err(EconError::EmptyDataset)
        ));
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
    fn test_density_structure_properties() {
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
    fn test_density_integrates_properties() {
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

    #[test]
    fn test_density_kernel_parses_aliases_and_rejects_unknown() {
        for (s, k) in [
            ("Gaussian", DensityKernel::Gaussian),
            ("normal", DensityKernel::Gaussian),
            (" NORMAL ", DensityKernel::Gaussian),
            ("uniform", DensityKernel::Rectangular),
            ("Rectangular", DensityKernel::Rectangular),
            ("quartic", DensityKernel::Biweight),
            ("Epanechnikov", DensityKernel::Epanechnikov),
            ("triangular", DensityKernel::Triangular),
            ("cosine", DensityKernel::Cosine),
            ("OptCosine", DensityKernel::OptCosine),
        ] {
            assert_eq!(s.parse::<DensityKernel>().unwrap(), k, "{s:?}");
        }
        assert_eq!(
            spec_message("parzen".parse::<DensityKernel>()),
            "unknown kernel 'parzen'; expected one of: gaussian, epanechnikov, rectangular, \
             triangular, biweight, cosine, optcosine"
        );
        assert_eq!(DensityKernel::choices().validate(), Ok(()));
        assert_eq!(DensityKernel::Biweight.to_string(), "biweight");
        assert!(matches!(
            run_density(&[1.0, 2.0, 3.5, 4.0], Some(0.5), "bogus", Some(16)),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    /// STAT-67 (robust): mad with the default constant, constant = 1 and a given centre, pinned to R on noisy data.
    #[test]
    fn test_validate_mad_t3() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_robust_weighted_t3");
        let x = numeric_column(ds.df(), "x1", NaPolicy::Fail).unwrap().0;
        Check::new(&case)
            .scalar("mad", mad(&x, None, None).unwrap(), Tol::coef(25))
            .scalar(
                "mad_const1",
                mad(&x, None, Some(1.0)).unwrap(),
                Tol::coef(25),
            )
            .scalar(
                "mad_center10",
                mad(&x, Some(10.0), None).unwrap(),
                Tol::coef(25),
            )
            .finish();
    }
}
