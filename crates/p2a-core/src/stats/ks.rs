//! Kolmogorov-Smirnov Test for Distribution Comparison.
//!
//! The Kolmogorov-Smirnov test is a nonparametric test that compares:
//! - A sample with a reference probability distribution (one-sample test)
//! - Two samples to determine if they come from the same distribution (two-sample test)
//!
//! # References
//!
//! - Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione".
//!   *Giornale dell'Istituto Italiano degli Attuari*, 4, 83-91.
//! - Smirnov, N. V. (1939). "On the estimation of the discrepancy between empirical curves
//!   of distribution for two independent samples". *Bulletin of Moscow University*, 2(2), 3-16.
//! - Marsaglia, G., Tsang, W. W., & Wang, J. (2003). "Evaluating Kolmogorov's distribution".
//!   *Journal of Statistical Software*, 8(18), 1-4.
//! - R Core Team. `stats::ks.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html>
//!
//! # Mathematical Background
//!
//! ## Test Statistic
//!
//! For the two-sample case with samples of size n and m:
//!
//! ```text
//! D = sup_x |F_n(x) - G_m(x)|
//! ```
//!
//! Where F_n and G_m are the empirical CDFs of the two samples.
//!
//! For one-sided alternatives:
//! - D+ = sup_x [F_n(x) - G_m(x)]: alternative "greater", the CDF of x lies above that of y
//!   (x tends to be smaller)
//! - D- = sup_x [G_m(x) - F_n(x)]: alternative "less", the CDF of x lies below that of y
//!   (x tends to be larger)
//!
//! ## Asymptotic Distribution
//!
//! Under the null hypothesis, √(nm/(n+m)) × D converges to the Kolmogorov
//! distribution K, with CDF:
//!
//! ```text
//! P(K ≤ x) = 1 - 2∑_{k=1}^∞ (-1)^{k+1} e^{-2k²x²}
//! ```
//!
//! For one-sided tests, √(nm/(n+m)) × D± converges to half of this distribution.

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{MissingSummary, sample_column};
use crate::errors::{EconError, EconResult};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;

// ═══════════════════════════════════════════════════════════════════════════════
// Result Structs
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a Kolmogorov-Smirnov test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KsTestResult {
    /// Test name
    pub test_name: String,

    /// Test statistic D (maximum absolute difference between CDFs)
    pub statistic: f64,

    /// P-value
    pub p_value: f64,

    /// Significance level
    pub significance: SignificanceLevel,

    /// Alternative hypothesis
    pub alternative: Alternative,

    /// Whether exact or asymptotic p-value was computed
    pub exact: bool,

    /// Sample size (first sample or only sample)
    pub n: usize,

    /// Second sample size (for two-sample test)
    pub n_2: Option<usize>,

    /// Whether the null hypothesis is rejected at α = 0.05
    pub reject_null: bool,
    /// Rows dropped because of missing values (dataset wrapper; nothing for slices). Each
    /// sample drops its own missing values; the summary sums the two columns.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for KsTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(
            f,
            "D = {:.6}, p-value = {:.6} {}",
            self.statistic,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        let alt_str = match self.alternative {
            Alternative::TwoSided => "two-sided",
            Alternative::Greater => "greater (CDF of x not below CDF of y)",
            Alternative::Less => "less (CDF of x not above CDF of y)",
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        if let Some(n2) = self.n_2 {
            writeln!(f, "Sample sizes: n = {}, m = {}", self.n, n2)?;
        } else {
            writeln!(f, "Sample size: n = {}", self.n)?;
        }

        writeln!(f)?;
        writeln!(
            f,
            "P-value method: {}",
            if self.exact { "exact" } else { "asymptotic" }
        )?;
        writeln!(f)?;

        if self.reject_null {
            writeln!(
                f,
                "Conclusion: Reject H₀ at α = 0.05 (distributions differ)"
            )?;
        } else {
            writeln!(
                f,
                "Conclusion: Fail to reject H₀ at α = 0.05 (no evidence of difference)"
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        // Two samples drop their missing values separately.
        let note = if self.n_2.is_some() {
            self.missing.two_sample_note()
        } else {
            self.missing.note()
        };
        if let Some(note) = note {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// One-Sample KS Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Distribution to test against in one-sample KS test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TheoreticalDistribution {
    /// Standard normal distribution N(0, 1)
    Normal,
    /// Normal distribution with specified mean and standard deviation
    NormalParams { mean: f64, sd: f64 },
    /// Uniform distribution on [0, 1]
    Uniform,
    /// Uniform distribution on [a, b]
    UniformParams { a: f64, b: f64 },
    /// Exponential distribution with rate λ
    Exponential { rate: f64 },
}

/// Kind of reference distribution, for the CLI and MCP option strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KsDistribution {
    /// Normal distribution (`pnorm`)
    Normal,
    /// Uniform distribution (`punif`)
    Uniform,
    /// Exponential distribution (`pexp`)
    Exponential,
}

crate::option_enum!(
    KsDistribution,
    crate::options::Choices {
        option: "distribution",
        values: &[
            (
                &["normal", "pnorm", "norm", "gaussian"],
                KsDistribution::Normal
            ),
            (&["uniform", "punif", "unif"], KsDistribution::Uniform),
            (&["exponential", "pexp", "exp"], KsDistribution::Exponential),
        ],
    }
);

/// The reference distribution, built once after its parameters were validated.
enum ReferenceCdf {
    Normal(statrs::distribution::Normal),
    Uniform(statrs::distribution::Uniform),
    Exp(statrs::distribution::Exp),
}

impl ReferenceCdf {
    fn cdf(&self, x: f64) -> f64 {
        use statrs::distribution::ContinuousCDF;

        match self {
            ReferenceCdf::Normal(d) => d.cdf(x),
            ReferenceCdf::Uniform(d) => d.cdf(x),
            ReferenceCdf::Exp(d) => d.cdf(x),
        }
    }
}

impl TheoreticalDistribution {
    /// Check the parameters: mean finite; sd and rate finite and above 0; a and b finite with
    /// a below b. Stricter than `ks.test`, which evaluates a point mass at `sd = 0` and warns
    /// and returns NaN for a negative rate. The error names the parameter.
    pub fn validate(&self) -> EconResult<()> {
        let refuse = |message: String| Err(EconError::InvalidSpecification { message });
        match *self {
            TheoreticalDistribution::Normal | TheoreticalDistribution::Uniform => Ok(()),
            TheoreticalDistribution::NormalParams { mean, sd } => {
                if !mean.is_finite() {
                    return refuse(format!(
                        "KS reference normal: mean must be finite, got {mean}"
                    ));
                }
                if !(sd.is_finite() && sd > 0.0) {
                    return refuse(format!(
                        "KS reference normal: sd must be finite and > 0, got {sd}"
                    ));
                }
                Ok(())
            }
            TheoreticalDistribution::UniformParams { a, b } => {
                if !a.is_finite() {
                    return refuse(format!(
                        "KS reference uniform: a (min) must be finite, got {a}"
                    ));
                }
                if !b.is_finite() {
                    return refuse(format!(
                        "KS reference uniform: b (max) must be finite, got {b}"
                    ));
                }
                if a >= b {
                    return refuse(format!(
                        "KS reference uniform: a (min) must be less than b (max), got a = {a}, \
                         b = {b}"
                    ));
                }
                Ok(())
            }
            TheoreticalDistribution::Exponential { rate } => {
                if !(rate.is_finite() && rate > 0.0) {
                    return refuse(format!(
                        "KS reference exponential: rate must be finite and > 0, got {rate}"
                    ));
                }
                Ok(())
            }
        }
    }

    /// The reference distribution from a kind and optional parameters, with R's defaults
    /// (mean 0, sd 1, min 0, max 1, rate 1); validated. The standard normal and U(0, 1) give
    /// the parameter-free variants. A parameter the kind does not take is an error naming both,
    /// as R's `ks.test` stops with "unused argument".
    pub fn from_kind(
        kind: KsDistribution,
        mean: Option<f64>,
        sd: Option<f64>,
        min: Option<f64>,
        max: Option<f64>,
        rate: Option<f64>,
    ) -> EconResult<Self> {
        let others: &[(&str, Option<f64>)] = match kind {
            KsDistribution::Normal => &[("min", min), ("max", max), ("rate", rate)],
            KsDistribution::Uniform => &[("mean", mean), ("sd", sd), ("rate", rate)],
            KsDistribution::Exponential => {
                &[("mean", mean), ("sd", sd), ("min", min), ("max", max)]
            }
        };
        if let Some((param, _)) = others.iter().find(|(_, value)| value.is_some()) {
            return Err(EconError::InvalidSpecification {
                message: format!("KS reference {kind}: parameter '{param}' does not apply"),
            });
        }
        let dist = match kind {
            KsDistribution::Normal => {
                let (mean, sd) = (mean.unwrap_or(0.0), sd.unwrap_or(1.0));
                if mean == 0.0 && sd == 1.0 {
                    TheoreticalDistribution::Normal
                } else {
                    TheoreticalDistribution::NormalParams { mean, sd }
                }
            }
            KsDistribution::Uniform => {
                let (a, b) = (min.unwrap_or(0.0), max.unwrap_or(1.0));
                if a == 0.0 && b == 1.0 {
                    TheoreticalDistribution::Uniform
                } else {
                    TheoreticalDistribution::UniformParams { a, b }
                }
            }
            KsDistribution::Exponential => TheoreticalDistribution::Exponential {
                rate: rate.unwrap_or(1.0),
            },
        };
        dist.validate()?;
        Ok(dist)
    }

    /// R's name for the distribution: "pnorm", "pnorm(mean = 2, sd = 2)", "punif",
    /// "punif(min = -1.5, max = 3)", "pexp(rate = 0.8)".
    pub fn label(&self) -> String {
        match self {
            TheoreticalDistribution::Normal => "pnorm".to_string(),
            TheoreticalDistribution::NormalParams { mean, sd } => {
                format!("pnorm(mean = {mean}, sd = {sd})")
            }
            TheoreticalDistribution::Uniform => "punif".to_string(),
            TheoreticalDistribution::UniformParams { a, b } => {
                format!("punif(min = {a}, max = {b})")
            }
            TheoreticalDistribution::Exponential { rate } => format!("pexp(rate = {rate})"),
        }
    }

    /// Validate the parameters and build the distribution.
    fn reference_cdf(&self) -> EconResult<ReferenceCdf> {
        use statrs::distribution as sd;

        self.validate()?;
        let unreachable = |e: &dyn std::fmt::Display| {
            EconError::Internal(format!("KS reference distribution after validation: {e}"))
        };
        Ok(match *self {
            TheoreticalDistribution::Normal => {
                ReferenceCdf::Normal(sd::Normal::new(0.0, 1.0).map_err(|e| unreachable(&e))?)
            }
            TheoreticalDistribution::NormalParams { mean, sd: s } => {
                ReferenceCdf::Normal(sd::Normal::new(mean, s).map_err(|e| unreachable(&e))?)
            }
            TheoreticalDistribution::Uniform => {
                ReferenceCdf::Uniform(sd::Uniform::new(0.0, 1.0).map_err(|e| unreachable(&e))?)
            }
            TheoreticalDistribution::UniformParams { a, b } => {
                ReferenceCdf::Uniform(sd::Uniform::new(a, b).map_err(|e| unreachable(&e))?)
            }
            TheoreticalDistribution::Exponential { rate } => {
                ReferenceCdf::Exp(sd::Exp::new(rate).map_err(|e| unreachable(&e))?)
            }
        })
    }
}

/// Perform one-sample Kolmogorov-Smirnov test.
///
/// Tests whether a sample comes from a specified theoretical distribution.
///
/// # Arguments
/// * `x` - Sample data
/// * `distribution` - The theoretical distribution to test against
/// * `alternative` - Direction of alternative hypothesis
///
/// # Returns
/// * `KsTestResult` containing the D statistic and p-value
///
/// # Example
/// ```ignore
/// // Test if data is normally distributed
/// let x = vec![0.1, -0.5, 0.3, 0.8, -0.2, 0.6, -0.1];
/// let result = ks_test_one_sample(&x, TheoreticalDistribution::Normal, Alternative::TwoSided)?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `ks.test(x, "pnorm")` for standard normal
pub fn ks_test_one_sample(
    x: &[f64],
    distribution: TheoreticalDistribution,
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    let cdf = distribution.reference_cdf()?;

    // Filter out NaN values
    let mut data: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = data.len();

    if n < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n,
            context: "KS test requires at least 1 observation".to_string(),
        });
    }

    // Sort data
    data.sort_by(f64::total_cmp);

    // Compute D statistic
    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;

    for (i, &xi) in data.iter().enumerate() {
        let f_emp_upper = (i + 1) as f64 / n as f64; // F_n(x_i)
        let f_emp_lower = i as f64 / n as f64; // F_n(x_i^-)
        let f_theo = cdf.cdf(xi);

        d_plus = d_plus.max(f_emp_upper - f_theo);
        d_minus = d_minus.max(f_theo - f_emp_lower);
    }

    let (statistic, p_value) = match alternative {
        Alternative::TwoSided => {
            let d = d_plus.max(d_minus);
            let p = kolmogorov_p_value_one_sample(d, n, true);
            (d, p)
        }
        Alternative::Greater => (d_plus, kolmogorov_p_value_one_sample(d_plus, n, false)),
        Alternative::Less => (d_minus, kolmogorov_p_value_one_sample(d_minus, n, false)),
    };

    // Determine if using exact or asymptotic
    let exact = n < 100;

    Ok(KsTestResult {
        test_name: "One-sample Kolmogorov-Smirnov test".to_string(),
        statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        exact,
        n,
        n_2: None,
        reject_null: p_value < 0.05,
        missing: MissingSummary::none(n),
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// Two-Sample KS Test
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform two-sample Kolmogorov-Smirnov test.
///
/// Tests whether two samples come from the same (unknown) distribution.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Second sample data
/// * `alternative` - Direction of alternative hypothesis
///
/// # Returns
/// * `KsTestResult` containing the D statistic and p-value
///
/// # Mathematical Details
///
/// The test statistic is:
/// ```text
/// D = sup_x |F_n(x) - G_m(x)|
/// ```
///
/// For the two-sided test, the asymptotic p-value uses the Kolmogorov distribution
/// with effective sample size √(n*m/(n+m)).
///
/// # Example
/// ```ignore
/// let x = vec![1.2, 1.5, 1.8, 2.1, 2.4];
/// let y = vec![2.0, 2.5, 3.0, 3.5, 4.0];
/// let result = ks_test_two_sample(&x, &y, Alternative::TwoSided)?;
/// if result.reject_null {
///     println!("Samples appear to come from different distributions");
/// }
/// ```
///
/// # References
/// - R equivalent: `ks.test(x, y)`
/// - SciPy equivalent: `scipy.stats.ks_2samp(x, y)`
pub fn ks_test_two_sample(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    // Filter out NaN values
    let x_clean: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let y_clean: Vec<f64> = y.iter().copied().filter(|v| !v.is_nan()).collect();

    let n = x_clean.len();
    let m = y_clean.len();

    if n < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n,
            context: "First sample requires at least 1 observation".to_string(),
        });
    }
    if m < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: m,
            context: "Second sample requires at least 1 observation".to_string(),
        });
    }

    // Compute D statistic using the combined sorted approach
    let (d_plus, d_minus) = compute_two_sample_d(&x_clean, &y_clean);

    let (statistic, p_value) = match alternative {
        Alternative::TwoSided => {
            let d = d_plus.max(d_minus);
            let p = kolmogorov_p_value_two_sample(d, n, m, true);
            (d, p)
        }
        Alternative::Greater => (d_plus, kolmogorov_p_value_two_sample(d_plus, n, m, false)),
        Alternative::Less => (d_minus, kolmogorov_p_value_two_sample(d_minus, n, m, false)),
    };

    // Use exact for small samples (product < 10000)
    let exact = n * m < 10000;

    Ok(KsTestResult {
        test_name: "Two-sample Kolmogorov-Smirnov test".to_string(),
        statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        exact,
        n,
        n_2: Some(m),
        reject_null: p_value < 0.05,
        missing: MissingSummary::none(n + m),
    })
}

/// Compute D+ and D- statistics for two-sample test.
///
/// Uses the efficient algorithm of tracking both CDFs simultaneously.
/// Following R's convention:
/// - D+ = max[F_n(x) - G_m(x)]: alternative "greater", the CDF of x lies above that of y
///   (x tends to be smaller)
/// - D- = max[G_m(x) - F_n(x)]: alternative "less", the CDF of x lies below that of y
///   (x tends to be larger)
fn compute_two_sample_d(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let m = y.len() as f64;

    // Combined sorted array with labels; tied values are processed together below
    let mut combined: Vec<(f64, bool)> = x.iter().map(|&v| (v, true)).collect();
    combined.extend(y.iter().map(|&v| (v, false)));
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut d_plus = 0.0_f64;
    let mut d_minus = 0.0_f64;
    let mut fn_x = 0.0; // F_n(x) - CDF of first sample
    let mut fm_y = 0.0; // G_m(x) - CDF of second sample

    // Group by value to handle ties properly
    let mut i = 0;
    while i < combined.len() {
        let current_val = combined[i].0;

        // Count how many from each sample at this value (handling ties)
        let mut x_count = 0usize;
        let mut y_count = 0usize;

        while i < combined.len() && combined[i].0 == current_val {
            if combined[i].1 {
                x_count += 1;
            } else {
                y_count += 1;
            }
            i += 1;
        }

        // Update CDFs
        fn_x += x_count as f64 / n;
        fm_y += y_count as f64 / m;

        // Compute differences after processing all observations at this value
        d_plus = d_plus.max(fn_x - fm_y);
        d_minus = d_minus.max(fm_y - fn_x);
    }

    (d_plus, d_minus)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Dataset Interface
// ═══════════════════════════════════════════════════════════════════════════════

/// Perform Kolmogorov-Smirnov test using dataset columns.
///
/// Each column drops its own missing values (null or NaN), as `ks.test` does, and the drop is
/// reported in `missing` (summed over both columns for the two-sample test). Integer, decimal
/// and boolean columns are accepted; a text column is refused (`EconError::NonNumericColumn`),
/// as are infinite values (`EconError::NonFiniteValues`).
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `x_col` - Name of first variable column
/// * `y_col` - Optional name of second variable column (for two-sample test)
/// * `distribution` - Theoretical distribution (for one-sample test, ignored if y_col is Some)
/// * `alternative` - Direction of alternative hypothesis
///
/// # Example
/// ```ignore
/// // One-sample test against normal distribution
/// let result = ks_test(&dataset, "x", None, Some(TheoreticalDistribution::Normal), Alternative::TwoSided)?;
///
/// // Two-sample test
/// let result = ks_test(&dataset, "x", Some("y"), None, Alternative::TwoSided)?;
/// ```
pub fn ks_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: Option<&str>,
    distribution: Option<TheoreticalDistribution>,
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    let df = dataset.df();
    let (x, missing_x) = sample_column(df, x_col, "Kolmogorov-Smirnov")?;

    match y_col {
        Some(y_name) => {
            // Two-sample test
            let (y, missing_y) = sample_column(df, y_name, "Kolmogorov-Smirnov")?;
            let mut result = ks_test_two_sample(&x, &y, alternative)?;
            result.missing = MissingSummary::merge_samples(&missing_x, &missing_y);
            Ok(result)
        }
        None => {
            // One-sample test
            let dist = distribution.unwrap_or(TheoreticalDistribution::Normal);
            let mut result = ks_test_one_sample(&x, dist, alternative)?;
            result.missing = missing_x;
            Ok(result)
        }
    }
}

/// Convenience function for two-sample KS test from dataset.
pub fn run_ks_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    alternative: Alternative,
) -> EconResult<KsTestResult> {
    ks_test(dataset, x_col, Some(y_col), None, alternative)
}

// ═══════════════════════════════════════════════════════════════════════════════
// P-Value Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute p-value for one-sample KS test using asymptotic approximation.
///
/// Uses the Kolmogorov distribution with the transformation √n × D.
fn kolmogorov_p_value_one_sample(d: f64, n: usize, two_sided: bool) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let sqrt_n = (n as f64).sqrt();
    let z = sqrt_n * d;

    if two_sided {
        kolmogorov_cdf_complement(z)
    } else {
        // One-sided: P(D+ > d) = exp(-2 * n * d^2) for large n
        // This is the asymptotic formula
        (-2.0 * (n as f64) * d * d).exp().min(1.0)
    }
}

/// Compute p-value for two-sample KS test using asymptotic approximation.
///
/// Uses effective sample size √(n*m/(n+m)).
fn kolmogorov_p_value_two_sample(d: f64, n: usize, m: usize, two_sided: bool) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }

    let nf = n as f64;
    let mf = m as f64;

    // Effective sample size
    let en = (nf * mf / (nf + mf)).sqrt();
    let z = (en + 0.12 + 0.11 / en) * d; // Stephens (1970) correction

    if two_sided {
        kolmogorov_cdf_complement(z)
    } else {
        // One-sided asymptotic formula
        let lambda = en * d;
        (-2.0 * lambda * lambda).exp().min(1.0)
    }
}

/// Compute the complement of the Kolmogorov CDF: P(K > x).
///
/// Uses the series expansion from Marsaglia, Tsang & Wang (2003):
/// P(K > x) = 2 * sum_{k=1}^∞ (-1)^{k+1} * exp(-2 * k^2 * x^2)
///
/// This converges very quickly for x > 0.
fn kolmogorov_cdf_complement(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x >= 3.0 {
        // For large x, the tail probability is essentially 0
        return 0.0;
    }

    let x_sq = x * x;
    let mut sum = 0.0;
    let mut sign = 1.0;

    // Series converges very fast - typically 3-5 terms are enough
    for k in 1..=100 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * x_sq).exp();

        if term.abs() < 1e-15 {
            break;
        }

        sum += term;
        sign = -sign;
    }

    (2.0 * sum).clamp(0.0, 1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    #[test]
    fn test_ks_two_sample_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));
        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_ks_two_sample_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // Identical samples should have D = 0 (when handling ties properly)
        assert!(
            result.statistic < 0.01,
            "D should be 0 or very small for identical samples: {}",
            result.statistic
        );
        // P-value should be high (not reject)
        assert!(
            result.p_value > 0.9,
            "p-value should be high for identical samples: {}",
            result.p_value
        );
    }

    #[test]
    fn test_ks_two_sample_different() {
        // Clearly different distributions
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // Should have D = 1.0 (maximum difference)
        assert!((result.statistic - 1.0).abs() < 1e-10);
        assert!(result.p_value < 0.05);
        assert!(result.reject_null);
    }

    #[test]
    fn test_ks_one_sample_uniform() {
        // Data from uniform(0,1) - should not reject
        let x = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let result =
            ks_test_one_sample(&x, TheoreticalDistribution::Uniform, Alternative::TwoSided)
                .unwrap();

        assert_eq!(result.n, 9);
        assert!(result.statistic >= 0.0 && result.statistic <= 1.0);
        // Fairly uniform data should have high p-value
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_ks_one_sample_non_normal() {
        // Clearly non-normal data (all positive, skewed)
        let x = vec![0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0];

        let result =
            ks_test_one_sample(&x, TheoreticalDistribution::Normal, Alternative::TwoSided).unwrap();

        // Should detect deviation from normality
        assert!(result.statistic > 0.3);
    }

    #[test]
    fn test_ks_alternatives() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0]; // Shifted up (y is stochastically greater than x)

        let result_two = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();
        let result_less = ks_test_two_sample(&x, &y, Alternative::Less).unwrap();
        let result_greater = ks_test_two_sample(&x, &y, Alternative::Greater).unwrap();

        // When y > x (stochastically):
        // - For any value v, more x's are below v than y's
        // - So CDF_x(v) > CDF_y(v) for most values
        // - D+ = max(CDF_x - CDF_y) is large
        // - D- = max(CDF_y - CDF_x) is small or zero
        //
        // The "greater" alternative tests D+, which should be significant here
        // The "less" alternative tests D-, which should NOT be significant

        // D+ (greater statistic) should be larger than D- (less statistic)
        assert!(
            result_greater.statistic >= result_less.statistic,
            "D_plus should be >= D_minus when x < y: greater={}, less={}",
            result_greater.statistic,
            result_less.statistic
        );

        // Two-sided should equal max(D+, D-)
        assert!(
            (result_two.statistic - result_greater.statistic.max(result_less.statistic)).abs()
                < 1e-10,
            "Two-sided D should equal max(D+, D-): two={}, max={}",
            result_two.statistic,
            result_greater.statistic.max(result_less.statistic)
        );

        // P-value for "greater" should be smaller (more evidence)
        assert!(
            result_greater.p_value < result_less.p_value,
            "greater p-value should be smaller when x < y: greater={}, less={}",
            result_greater.p_value,
            result_less.p_value
        );
    }

    #[test]
    fn test_ks_handles_nan() {
        let x = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
        let y = vec![1.5, f64::NAN, 2.5, 3.5, 4.5];

        let result = ks_test_two_sample(&x, &y, Alternative::TwoSided).unwrap();

        // NaNs should be filtered out
        assert_eq!(result.n, 4);
        assert_eq!(result.n_2, Some(4));
    }

    #[test]
    fn test_ks_from_dataset() {
        let df = df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "y" => [1.5, 2.5, 3.5, 4.5, 5.5]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Two-sample test
        let result = ks_test(&dataset, "x", Some("y"), None, Alternative::TwoSided).unwrap();
        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));

        // One-sample test
        let result = ks_test(
            &dataset,
            "x",
            None,
            Some(TheoreticalDistribution::Normal),
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(result.n, 5);
        assert!(result.n_2.is_none());
    }

    #[test]
    fn test_kolmogorov_cdf() {
        // Test the Kolmogorov CDF complement
        // For z = 0, P(K > 0) = 1
        assert!((kolmogorov_cdf_complement(0.0) - 1.0).abs() < 1e-10);

        // For very large z, P(K > z) ≈ 0
        assert!(kolmogorov_cdf_complement(3.0) < 1e-10);

        // For z = 1.36, P(K > z) ≈ 0.05 (critical value at α = 0.05)
        let p = kolmogorov_cdf_complement(1.36);
        assert!(p > 0.04 && p < 0.06, "p={}", p);
    }

    // ========================================================================
    // Missing values and integer columns (case stat_two_sample_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// X-2/X-8: two-sample KS on Int64 columns with nulls, each sample dropping its own.
    #[test]
    fn test_validate_ks_test_two_sample_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_ks_test(&ds, "xi", "yi", Alternative::TwoSided).unwrap();
        Check::new(&case)
            .scalar("ks_d", r.statistic, Tol::stat(38))
            .scalar("n_xi", r.n as f64, Tol::exact())
            .scalar("n_yi", r.n_2.unwrap() as f64, Tol::exact())
            .finish();
        // deviation STAT-27: ks_p (R's exact p conditional on ties 0.15653867760174539;
        // p2a's asymptotic p with Stephens' correction 0.19563779265722) is not pinned.
    }

    #[test]
    fn test_validate_na_ks_test_missing_rows_and_errors() {
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let two = run_ks_test(&ds, "x", "y", Alternative::TwoSided).unwrap();
        assert_eq!((two.n, two.n_2), (7, Some(7)));
        assert_eq!(
            two.missing.by_column,
            vec![("x".to_string(), 15), ("y".to_string(), 15)]
        );
        assert_eq!(
            two.to_string().lines().last().unwrap(),
            "Missing values: 30 of 44 values dropped because of missing values (two samples of \
             22; x: 15, y: 15)"
        );
        let one = ks_test(
            &ds,
            "x",
            None,
            Some(TheoreticalDistribution::Normal),
            Alternative::TwoSided,
        )
        .unwrap();
        assert_eq!(one.n, 7);
        assert_eq!(one.missing.by_column, vec![("x".to_string(), 15)]);
        assert_eq!(
            one.to_string().lines().last().unwrap(),
            format!("Missing values: {}", one.missing.note().unwrap())
        );
        for (x, y) in [("g", "y"), ("x", "g")] {
            assert!(
                matches!(run_ks_test(&ds, x, y, Alternative::TwoSided), Err(EconError::NonNumericColumn { column }) if column == "g"),
                "{x} {y}"
            );
        }
        assert!(matches!(
            run_ks_test(&ds, "x", "nope", Alternative::TwoSided),
            Err(EconError::ColumnNotFound { column, .. }) if column == "nope"
        ));
    }

    /// STAT-26: invalid reference parameters are errors naming the parameter.
    #[test]
    fn test_ks_rejects_invalid_parameters() {
        use super::TheoreticalDistribution::*;
        let x = [0.31, -1.22, 0.84, 1.93, -0.47, 0.05];
        let bad = [
            (NormalParams { mean: 0.0, sd: 0.0 }, "sd"),
            (
                NormalParams {
                    mean: 0.0,
                    sd: -1.0,
                },
                "sd",
            ),
            (
                NormalParams {
                    mean: 0.0,
                    sd: f64::NAN,
                },
                "sd",
            ),
            (
                NormalParams {
                    mean: 0.0,
                    sd: f64::INFINITY,
                },
                "sd",
            ),
            (
                NormalParams {
                    mean: f64::NAN,
                    sd: 1.0,
                },
                "mean",
            ),
            (
                NormalParams {
                    mean: f64::INFINITY,
                    sd: 1.0,
                },
                "mean",
            ),
            (UniformParams { a: 1.0, b: 1.0 }, "less than b"),
            (UniformParams { a: 1.0, b: 0.0 }, "less than b"),
            (
                UniformParams {
                    a: f64::NAN,
                    b: 1.0,
                },
                "a (min)",
            ),
            (
                UniformParams {
                    a: 0.0,
                    b: f64::INFINITY,
                },
                "b (max)",
            ),
            (Exponential { rate: 0.0 }, "rate"),
            (Exponential { rate: -2.0 }, "rate"),
            (Exponential { rate: f64::NAN }, "rate"),
            (
                Exponential {
                    rate: f64::INFINITY,
                },
                "rate",
            ),
        ];
        for (dist, name) in bad {
            assert!(dist.validate().is_err(), "{dist:?}");
            for alt in [
                Alternative::TwoSided,
                Alternative::Greater,
                Alternative::Less,
            ] {
                let e = ks_test_one_sample(&x, dist, alt).unwrap_err();
                assert!(
                    matches!(&e, EconError::InvalidSpecification { message } if message.contains(name)),
                    "{dist:?}: {e}"
                );
            }
        }
        // the dataset wrapper reaches the same check
        let ds = Dataset::new(polars::prelude::df! { "x" => x.to_vec() }.unwrap());
        assert!(
            ks_test(
                &ds,
                "x",
                None,
                Some(Exponential { rate: 0.0 }),
                Alternative::TwoSided
            )
            .is_err()
        );
        // valid parameters still run
        assert!(
            ks_test_one_sample(
                &x,
                NormalParams { mean: 0.0, sd: 2.0 },
                Alternative::TwoSided
            )
            .is_ok()
        );
    }

    /// STAT-67: one-sample KS against R (ks.test(x, "pnorm"/"punif"/"pexp", ...)). The asymptotic p is R's
    /// exact = FALSE value (sqrt(n) D >= 1 in every case, where R's pKS2 with tol 1e-6 is exact to ~1e-15).
    #[test]
    fn test_validate_ks_t3_one_sample() {
        use super::TheoreticalDistribution::*;
        let (case, ds) = crate::test_support::load_case("stat_ks_t3");
        let mut c = Check::new(&case);
        for (tag, col, dist) in [
            ("norm", "xn", NormalParams { mean: 2.0, sd: 2.0 }),
            ("unif", "xu", UniformParams { a: -1.5, b: 3.0 }),
            ("exp", "xe", Exponential { rate: 0.8 }),
        ] {
            let two = ks_test(&ds, col, None, Some(dist), Alternative::TwoSided).unwrap();
            let gt = ks_test(&ds, col, None, Some(dist), Alternative::Greater).unwrap();
            let lt = ks_test(&ds, col, None, Some(dist), Alternative::Less).unwrap();
            let n = two.n;
            c.scalar(&format!("{tag}_d"), two.statistic, Tol::stat(n))
                .scalar(&format!("{tag}_p_asym"), two.p_value, Tol::p_value(n))
                .scalar(&format!("{tag}_dplus"), gt.statistic, Tol::stat(n))
                .scalar(
                    &format!("{tag}_p_greater_asym"),
                    gt.p_value,
                    Tol::p_value(n),
                )
                .scalar(&format!("{tag}_dminus"), lt.statistic, Tol::stat(n))
                .scalar(&format!("{tag}_p_less_asym"), lt.p_value, Tol::p_value(n))
                .scalar(&format!("n_{col}"), n as f64, Tol::exact());
            // deviation STAT-27: {tag}_p_exact (R's default exact p; norm 0.12705346811680995) is not pinned.
        }
        c.finish();
    }

    /// STAT-67, X-13: two-sample KS against R. D, D+ and D- and the one-sided asymptotic p-values are R's;
    /// the two-sided p-values are deviations.
    #[test]
    fn test_validate_ks_t3_two_sample() {
        let (case, ds) = crate::test_support::load_case("stat_ks_t3");
        let mut c = Check::new(&case);
        for (tag, x, y) in [("two5", "x5", "y5"), ("twol", "xa", "ya")] {
            let two = ks_test(&ds, x, Some(y), None, Alternative::TwoSided).unwrap();
            let gt = ks_test(&ds, x, Some(y), None, Alternative::Greater).unwrap();
            let lt = ks_test(&ds, x, Some(y), None, Alternative::Less).unwrap();
            let n = two.n.min(two.n_2.unwrap());
            c.scalar(&format!("{tag}_d"), two.statistic, Tol::stat(n))
                .scalar(&format!("{tag}_dplus"), gt.statistic, Tol::stat(n))
                .scalar(
                    &format!("{tag}_p_greater_asym"),
                    gt.p_value,
                    Tol::p_value(n),
                )
                .scalar(&format!("{tag}_dminus"), lt.statistic, Tol::stat(n))
                .scalar(&format!("{tag}_p_less_asym"), lt.p_value, Tol::p_value(n));
            // deviation STAT-27: {tag}_p_exact (R exact; two5 0.07936507936507936) and {tag}_p_asym
            // (R 0.08151888641220961; p2a applies Stephens' factor: 0.036146190769284631) are not pinned.
        }
        c.scalar(
            "n_xa",
            ks_test(&ds, "xa", Some("ya"), None, Alternative::TwoSided)
                .unwrap()
                .n as f64,
            Tol::exact(),
        )
        .scalar(
            "n_ya",
            ks_test(&ds, "xa", Some("ya"), None, Alternative::TwoSided)
                .unwrap()
                .n_2
                .unwrap() as f64,
            Tol::exact(),
        )
        .finish();
    }

    #[test]
    fn test_ks_distribution_parses_aliases_and_rejects_unknown() {
        assert_eq!(KsDistribution::choices().validate(), Ok(()));
        assert_eq!(
            "pnorm".parse::<KsDistribution>().unwrap(),
            KsDistribution::Normal
        );
        assert_eq!(
            " Gaussian ".parse::<KsDistribution>().unwrap(),
            KsDistribution::Normal
        );
        assert_eq!(
            "PUNIF".parse::<KsDistribution>().unwrap(),
            KsDistribution::Uniform
        );
        assert_eq!(
            "pexp".parse::<KsDistribution>().unwrap(),
            KsDistribution::Exponential
        );
        let e = "weibull".parse::<KsDistribution>().unwrap_err().to_string();
        assert!(
            e.contains("unknown distribution 'weibull'")
                && e.contains("normal, uniform, exponential"),
            "{e}"
        );
        assert_eq!(KsDistribution::Uniform.to_string(), "uniform");
    }

    #[test]
    fn test_ks_from_kind_uses_r_defaults_and_labels() {
        use super::TheoreticalDistribution as T;
        let n = T::from_kind(KsDistribution::Normal, None, None, None, None, None).unwrap();
        assert_eq!(n, T::Normal);
        assert_eq!(n.label(), "pnorm");
        let n2 = T::from_kind(
            KsDistribution::Normal,
            Some(2.0),
            Some(2.0),
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(n2, T::NormalParams { mean: 2.0, sd: 2.0 });
        assert_eq!(n2.label(), "pnorm(mean = 2, sd = 2)");
        assert_eq!(
            T::from_kind(KsDistribution::Uniform, None, None, None, None, None).unwrap(),
            T::Uniform
        );
        let u = T::from_kind(
            KsDistribution::Uniform,
            None,
            None,
            Some(-1.5),
            Some(3.0),
            None,
        )
        .unwrap();
        assert_eq!(u.label(), "punif(min = -1.5, max = 3)");
        assert_eq!(
            T::from_kind(KsDistribution::Exponential, None, None, None, None, None).unwrap(),
            T::Exponential { rate: 1.0 }
        );
        assert_eq!(T::Exponential { rate: 0.8 }.label(), "pexp(rate = 0.8)");
        assert!(T::from_kind(KsDistribution::Uniform, None, None, Some(1.0), None, None).is_err()); // min 1 >= default max 1
        assert!(
            T::from_kind(
                KsDistribution::Exponential,
                None,
                None,
                None,
                None,
                Some(-1.0)
            )
            .is_err()
        );
    }

    /// R's ks.test stops on an argument the distribution does not take ("unused argument");
    /// dropping it would test against a distribution other than the one asked for.
    #[test]
    fn test_ks_from_kind_refuses_parameters_of_another_kind() {
        use super::TheoreticalDistribution as T;
        use KsDistribution::{Exponential, Normal, Uniform};
        let v = Some(1.5);
        for (r, kind, param) in [
            (
                T::from_kind(Normal, None, None, v, None, None),
                "normal",
                "min",
            ),
            (
                T::from_kind(Normal, None, None, None, v, None),
                "normal",
                "max",
            ),
            (
                T::from_kind(Normal, None, None, None, None, v),
                "normal",
                "rate",
            ),
            (
                T::from_kind(Uniform, v, None, None, None, None),
                "uniform",
                "mean",
            ),
            (
                T::from_kind(Uniform, None, v, None, None, None),
                "uniform",
                "sd",
            ),
            (
                T::from_kind(Uniform, None, None, None, None, v),
                "uniform",
                "rate",
            ),
            (
                T::from_kind(Exponential, v, None, None, None, None),
                "exponential",
                "mean",
            ),
            (
                T::from_kind(Exponential, None, v, None, None, None),
                "exponential",
                "sd",
            ),
            (
                T::from_kind(Exponential, None, None, v, None, None),
                "exponential",
                "min",
            ),
            (
                T::from_kind(Exponential, None, None, None, v, None),
                "exponential",
                "max",
            ),
        ] {
            match r {
                Err(EconError::InvalidSpecification { ref message }) => assert!(
                    message.contains(kind) && message.contains(&format!("'{param}'")),
                    "{message}"
                ),
                other => panic!("{kind} {param}: {other:?}"),
            }
        }
    }

    #[test]
    fn test_ks_all_missing_column_is_named() {
        let ds = Dataset::new(
            polars::prelude::df! { "x" => [1.2, 0.4, 2.2], "y" => [None::<f64>, None, None] }
                .unwrap(),
        );
        let e = ks_test(&ds, "x", Some("y"), None, Alternative::TwoSided)
            .unwrap_err()
            .to_string();
        assert!(e.contains("'y'") && e.contains("Kolmogorov-Smirnov"), "{e}");
    }
}
