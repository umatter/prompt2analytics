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

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use statrs::function::gamma::ln_gamma;

use crate::data::Dataset;
use crate::data::columns::{MissingSummary, sample_column};
use crate::errors::{EconError, EconResult};
use crate::resource::{checkpoint, try_vec, try_with_capacity, try_zeros2};
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

    /// Whether the exact p-value was computed (R's `exact`; `test_name` says the same)
    pub exact: bool,

    /// Sample size (first sample or only sample)
    pub n: usize,

    /// Second sample size (for two-sample test)
    pub n_2: Option<usize>,

    /// Whether the null hypothesis is rejected at α = 0.05
    pub reject_null: bool,
    /// R's warning (ties in the one-sample test, or an asymptotic two-sample p with ties)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
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

        if let Some(ref warn) = self.warning {
            writeln!(f, "Warning: {warn}")?;
        }

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
    ks_test_one_sample_with(x, distribution, alternative, None)
}

/// One-sample Kolmogorov-Smirnov test with R's `exact` argument.
///
/// `exact = None` is R's `exact = NULL`: the exact distribution for n < 100 without ties
/// (Marsaglia-Tsang-Wang for the two-sided statistic, Birnbaum-Tingey for one-sided ones),
/// the asymptotic one otherwise. Ties carry R's warning.
///
/// # References
/// - R equivalent: `ks.test(x, "pnorm", alternative = , exact = )`
pub fn ks_test_one_sample_with(
    x: &[f64],
    distribution: TheoreticalDistribution,
    alternative: Alternative,
    exact: Option<bool>,
) -> EconResult<KsTestResult> {
    let cdf = distribution.reference_cdf()?;

    // Filter out NaN values
    let mut data: Vec<f64> = try_with_capacity(x.len(), "KS sample")?;
    data.extend(x.iter().copied().filter(|v| !v.is_nan()));
    let n = data.len();

    if n < 1 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: n,
            context: "KS test requires at least 1 observation".to_string(),
        });
    }

    data.sort_by(f64::total_cmp);
    let ties = data.windows(2).any(|w| w[0] == w[1]);

    // R: x <- F(sort(x)) - (0:(n-1))/n; D+ = max(1/n - x), D- = max(x)
    let nf = n as f64;
    let mut d_plus = f64::NEG_INFINITY;
    let mut d_minus = f64::NEG_INFINITY;
    for (i, &xi) in data.iter().enumerate() {
        let v = cdf.cdf(xi) - i as f64 / nf;
        d_plus = d_plus.max(1.0 / nf - v);
        d_minus = d_minus.max(v);
    }
    let statistic = match alternative {
        Alternative::TwoSided => d_plus.max(d_minus),
        Alternative::Greater => d_plus,
        Alternative::Less => d_minus,
    };

    let exact = exact.unwrap_or(n < 100 && !ties);
    let p_value = if exact {
        match alternative {
            Alternative::TwoSided => {
                // numeric-ok: R's exact two-sided p is the complement of the MTW cdf
                1.0 - pkolmogorov_two_exact(statistic, n)?
            }
            _ => pkolmogorov_one_exact_upper(statistic, n),
        }
    } else {
        match alternative {
            Alternative::TwoSided => pkstwo_upper(nf.sqrt() * statistic),
            _ => (-2.0 * nf * statistic * statistic).exp(),
        }
    }
    .clamp(0.0, 1.0);
    let warning =
        ties.then(|| "ties should not be present for the Kolmogorov-Smirnov test".to_string());

    Ok(KsTestResult {
        test_name: format!(
            "{} one-sample Kolmogorov-Smirnov test",
            if exact { "Exact" } else { "Asymptotic" }
        ),
        statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        exact,
        n,
        n_2: None,
        reject_null: p_value < 0.05,
        warning,
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
    ks_test_two_sample_with(x, y, alternative, None)
}

/// Two-sample Kolmogorov-Smirnov test with R's `exact` argument.
///
/// `exact = None` is R's `exact = NULL`: the exact distribution when n m < 10000, conditional
/// on the tie pattern of the pooled sample (R's `psmirnov`); the asymptotic one otherwise,
/// with R's warning under ties. For `alternative = less` with ties the exact p is that of D-
/// (R evaluates the D+ distribution there; without ties both agree).
///
/// # References
/// - R equivalent: `ks.test(x, y, alternative = , exact = )`
pub fn ks_test_two_sample_with(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    exact: Option<bool>,
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

    let (d_plus, d_minus) = compute_two_sample_d(&x_clean, &y_clean);
    let statistic = match alternative {
        Alternative::TwoSided => d_plus.max(d_minus),
        Alternative::Greater => d_plus,
        Alternative::Less => d_minus,
    };

    // block_end[k]: position k (1-based) of the sorted pooled sample ends a block of ties
    let total = n + m;
    let mut pooled: Vec<f64> = try_with_capacity(total, "KS pooled sample")?;
    pooled.extend_from_slice(&x_clean);
    pooled.extend_from_slice(&y_clean);
    pooled.sort_by(f64::total_cmp);
    let mut block_end: Vec<bool> = try_vec(total + 1, true, "KS pooled sample")?;
    block_end[0] = false;
    for k in 1..total {
        block_end[k] = pooled[k - 1] != pooled[k];
    }
    let ties = block_end[1..total].iter().any(|&b| !b);

    let exact = exact.unwrap_or((n as f64) * (m as f64) < 10000.0);
    let p_value = if exact {
        let side = match alternative {
            Alternative::TwoSided => SmirnovSide::Both,
            Alternative::Greater => SmirnovSide::Plus,
            Alternative::Less => SmirnovSide::Minus,
        };
        psmirnov_exact_upper(statistic, n, m, &block_end, side)?
    } else {
        let n_eff = (n as f64) * (m as f64) / (total as f64);
        match alternative {
            Alternative::TwoSided => {
                if statistic < f64::EPSILON {
                    1.0
                } else {
                    pkstwo_upper(n_eff.sqrt() * statistic)
                }
            }
            _ => (-2.0 * n_eff * statistic * statistic).exp(),
        }
    }
    .clamp(0.0, 1.0);
    let warning =
        (ties && !exact).then(|| "p-value will be approximate in the presence of ties".to_string());

    Ok(KsTestResult {
        test_name: format!(
            "{} two-sample Kolmogorov-Smirnov test",
            if exact { "Exact" } else { "Asymptotic" }
        ),
        statistic,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        exact,
        n,
        n_2: Some(m),
        reject_null: p_value < 0.05,
        warning,
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
    ks_test_with(dataset, x_col, y_col, distribution, alternative, None)
}

/// [`ks_test`] with R's `exact` argument (`None`: R's default rule).
pub fn ks_test_with(
    dataset: &Dataset,
    x_col: &str,
    y_col: Option<&str>,
    distribution: Option<TheoreticalDistribution>,
    alternative: Alternative,
    exact: Option<bool>,
) -> EconResult<KsTestResult> {
    let df = dataset.df();
    let (x, missing_x) = sample_column(df, x_col, "Kolmogorov-Smirnov")?;

    match y_col {
        Some(y_name) => {
            // Two-sample test
            let (y, missing_y) = sample_column(df, y_name, "Kolmogorov-Smirnov")?;
            let mut result = ks_test_two_sample_with(&x, &y, alternative, exact)?;
            result.missing = MissingSummary::merge_samples(&missing_x, &missing_y);
            Ok(result)
        }
        None => {
            // One-sample test
            let dist = distribution.unwrap_or(TheoreticalDistribution::Normal);
            let mut result = ks_test_one_sample_with(&x, dist, alternative, exact)?;
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

/// R's tolerance for the asymptotic Kolmogorov series (`C_pKS2`, `tol = 1e-06`).
const PKS2_TOL: f64 = 1e-6;

/// Upper tail P(K > x) of the limiting Kolmogorov distribution, with R's two series
/// (`pkstwo`, tolerance 1e-6): for x < 1 the complement of
/// `sqrt(2 pi) / x sum_{k odd < k_max} exp(-k^2 pi^2 / (8 x^2))`, else the alternating series
/// `2 sum (-1)^(k-1) exp(-2 k^2 x^2)` summed directly until a term is below the tolerance.
pub(crate) fn pkstwo_upper(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 1.0 {
        let k_max = (2.0 - PKS2_TOL.ln()).sqrt() as usize;
        let z = -(std::f64::consts::FRAC_PI_2 * std::f64::consts::FRAC_PI_4) / (x * x);
        let w = x.ln();
        let mut s = 0.0;
        let mut k = 1usize;
        while k < k_max {
            let kf = k as f64;
            s += (kf * kf * z - w).exp();
            k += 2;
        }
        let cdf = s * (2.0 * std::f64::consts::PI).sqrt();
        // numeric-ok: the x < 1 series is the cdf, at most 0.73 here, so no tail is lost
        (1.0 - cdf).clamp(0.0, 1.0)
    } else {
        let z = -2.0 * x * x;
        let mut sign = 1.0;
        let mut tail = 0.0;
        let mut k = 1u32;
        loop {
            let kf = f64::from(k);
            let term = 2.0 * sign * (z * kf * kf).exp();
            tail += term;
            if term.abs() <= PKS2_TOL || k > 100_000 {
                break;
            }
            sign = -sign;
            k += 1;
        }
        tail.clamp(0.0, 1.0)
    }
}

/// Marsaglia, Tsang and Wang (2003) exact cdf P(D_n < d) of the two-sided one-sample
/// statistic, as R's `C_pKolmogorov2x` (matrix power with decimal exponent rescaling; R omits
/// the quick right-tail approximation, and so does this).
pub(crate) fn pkolmogorov_two_exact(d: f64, n: usize) -> EconResult<f64> {
    let nf = n as f64;
    let k = (nf * d) as usize + 1;
    let m = 2 * k - 1;
    let h = k as f64 - nf * d;
    let mut hm = try_zeros2(m, m, "Kolmogorov exact distribution")?;
    for i in 0..m {
        for j in 0..m {
            hm[[i, j]] = if i + 1 < j { 0.0 } else { 1.0 };
        }
    }
    for i in 0..m {
        hm[[i, 0]] -= h.powi(i as i32 + 1);
        hm[[m - 1, i]] -= h.powi((m - i) as i32);
    }
    if 2.0 * h - 1.0 > 0.0 {
        hm[[m - 1, 0]] += (2.0 * h - 1.0).powi(m as i32);
    }
    for i in 0..m {
        for j in 0..m {
            if i + 1 > j {
                for g in 1..=(i + 1 - j) {
                    hm[[i, j]] /= g as f64;
                }
            }
        }
    }
    let (q, mut eq) = matrix_power(&hm, n)?;
    let mut s = q[[k - 1, k - 1]];
    for i in 1..=n {
        s = s * i as f64 / nf;
        if s < 1e-140 {
            s *= 1e140;
            eq -= 140;
        }
    }
    Ok(s * 10f64.powi(eq))
}

/// `A^n` with R's rescaling by 1e-140 when the centre element passes 1e140; returns the
/// matrix and its decimal exponent.
fn matrix_power(a: &Array2<f64>, n: usize) -> EconResult<(Array2<f64>, i32)> {
    checkpoint()?;
    if n == 1 {
        return Ok((a.clone(), 0));
    }
    let (v, ev) = matrix_power(a, n / 2)?;
    let b = v.dot(&v);
    let eb = 2 * ev;
    let (mut out, mut e) = if n.is_multiple_of(2) {
        (b, eb)
    } else {
        (a.dot(&b), eb)
    };
    let c = a.nrows() / 2;
    if out[[c, c]] > 1e140 {
        out.mapv_inplace(|x| x * 1e-140);
        e += 140;
    }
    Ok((out, e))
}

/// Exact upper tail P(D+_n >= d) of the one-sided one-sample statistic (Birnbaum and
/// Tingey), summed directly: `d sum_j exp(lchoose(n, j) + (n - j) log(1 - d - j/n) +
/// (j - 1) log(d + j/n))`, j = 0..floor(n (1 - d)).
pub(crate) fn pkolmogorov_one_exact_upper(d: f64, n: usize) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let nf = n as f64;
    let jmax = (nf * (1.0 - d)).floor() as usize;
    let lnf = ln_gamma(nf + 1.0);
    let s: f64 = (0..=jmax)
        .map(|j| {
            let jf = j as f64;
            let lchoose = lnf - ln_gamma(jf + 1.0) - ln_gamma(nf - jf + 1.0);
            (lchoose + (nf - jf) * (1.0 - d - jf / nf).ln() + (jf - 1.0) * (d + jf / nf).ln()).exp()
        })
        .sum();
    (d * s).clamp(0.0, 1.0)
}

/// Which lattice boundary the two-sample exact distribution counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SmirnovSide {
    /// |F_x - F_y| (two-sided D)
    Both,
    /// F_x - F_y (D+, alternative "greater")
    Plus,
    /// F_y - F_x (D-, alternative "less")
    Minus,
}

/// Exact P(D >= d) of the two-sample Smirnov statistic, conditional on the tie pattern of
/// the pooled sample (`block_end[k]`: position k of the sorted pooled sample, k = 1..N, ends
/// a block of equal values). Paths from (0, 0) to (m, n) are weighted uniformly; `h(i, j)`
/// is the share of paths to (i, j) that have reached the boundary
/// `|i n - j m| > floor(d m n - 1e-7)` (signed for one side) at a block end, so the tail is
/// accumulated directly.
pub(crate) fn psmirnov_exact_upper(
    d: f64,
    m: usize,
    n: usize,
    block_end: &[bool],
    side: SmirnovSide,
) -> EconResult<f64> {
    let (mi, ni) = (m as i128, n as i128);
    let bound = (d * m as f64 * n as f64 - 1e-7).floor() as i128;
    let hit = |i: usize, j: usize| -> bool {
        if !block_end[i + j] {
            return false;
        }
        let diff = i as i128 * ni - j as i128 * mi;
        match side {
            SmirnovSide::Both => diff.abs() > bound,
            SmirnovSide::Plus => diff > bound,
            SmirnovSide::Minus => -diff > bound,
        }
    };
    let mut row: Vec<f64> = try_vec(n + 1, 0.0f64, "Smirnov exact distribution")?;
    // i = 0
    row[0] = 0.0;
    for j in 1..=n {
        row[j] = if hit(0, j) { 1.0 } else { row[j - 1] };
    }
    for i in 1..=m {
        checkpoint()?;
        row[0] = if hit(i, 0) { 1.0 } else { row[0] };
        for j in 1..=n {
            row[j] = if hit(i, j) {
                1.0
            } else {
                let t = (i + j) as f64;
                (i as f64 / t) * row[j] + (j as f64 / t) * row[j - 1]
            };
        }
    }
    Ok(row[n].clamp(0.0, 1.0))
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
        // P(K > 0) = 1; the tail stays positive past x = 3 (it returned 0 there before)
        assert!((pkstwo_upper(0.0) - 1.0).abs() < 1e-10);
        let p3 = pkstwo_upper(3.0);
        assert!(
            p3 > 0.0 && (p3 - 2.0 * (-18.0f64).exp()).abs() < 1e-20,
            "{p3}"
        );
        assert!(pkstwo_upper(5.0) > 0.0);
        // For z = 1.36, P(K > z) ≈ 0.05 (critical value at α = 0.05)
        let p = pkstwo_upper(1.36);
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
            .scalar("ks_p", r.p_value, Tol::p_value(38))
            .finish();
        assert!(
            r.exact && r.warning.is_none(),
            "exact conditional on ties, as R"
        );
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

    /// STAT-67: one-sample KS against R (ks.test(x, "pnorm"/"punif"/"pexp", ...)): the asymptotic p
    /// through exact = false, the exact p through R's default.
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
            let asym = |alt| ks_test_with(&ds, col, None, Some(dist), alt, Some(false)).unwrap();
            let two = asym(Alternative::TwoSided);
            let gt = asym(Alternative::Greater);
            let lt = asym(Alternative::Less);
            let def = ks_test(&ds, col, None, Some(dist), Alternative::TwoSided).unwrap();
            assert!(def.exact && !two.exact);
            assert_eq!(def.test_name, "Exact one-sample Kolmogorov-Smirnov test");
            let n = two.n;
            c.scalar(&format!("{tag}_p_exact"), def.p_value, Tol::p_value(n));
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
        }
        c.finish();
    }

    /// STAT-67, X-13: two-sample KS against R: D, D+ and D-, the asymptotic p-values through exact =
    /// false, the exact p through R's default.
    #[test]
    fn test_validate_ks_t3_two_sample() {
        let (case, ds) = crate::test_support::load_case("stat_ks_t3");
        let mut c = Check::new(&case);
        for (tag, x, y) in [("two5", "x5", "y5"), ("twol", "xa", "ya")] {
            let asym = |alt| ks_test_with(&ds, x, Some(y), None, alt, Some(false)).unwrap();
            let two = asym(Alternative::TwoSided);
            let gt = asym(Alternative::Greater);
            let lt = asym(Alternative::Less);
            let def = ks_test(&ds, x, Some(y), None, Alternative::TwoSided).unwrap();
            assert!(def.exact);
            let n = two.n.min(two.n_2.unwrap());
            c.scalar(&format!("{tag}_d"), two.statistic, Tol::stat(n))
                .scalar(&format!("{tag}_p_asym"), two.p_value, Tol::p_value(n))
                .scalar(&format!("{tag}_p_exact"), def.p_value, Tol::p_value(n))
                .scalar(&format!("{tag}_dplus"), gt.statistic, Tol::stat(n))
                .scalar(
                    &format!("{tag}_p_greater_asym"),
                    gt.p_value,
                    Tol::p_value(n),
                )
                .scalar(&format!("{tag}_dminus"), lt.statistic, Tol::stat(n))
                .scalar(&format!("{tag}_p_less_asym"), lt.p_value, Tol::p_value(n));
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

    fn t4(ds: &Dataset, set: &str, group: &str) -> Vec<f64> {
        crate::stats::ranks::test_data::column_where(ds, "value", &[("set", set), ("group", group)])
    }

    /// Exact one-sample p (MTW two-sided, Birnbaum-Tingey one-sided) under R's default.
    /// Fails before: the asymptotic p was labelled exact.
    #[test]
    fn test_validate_ks_one_sample_exact_t4() {
        let (case, ds) = crate::test_support::load_case("stat_rank_ks_t4");
        let std = TheoreticalDistribution::Normal;
        let run = |set: &str, alt| ks_test_one_sample(&t4(&ds, set, "x"), std, alt).unwrap();
        let two = run("o30", Alternative::TwoSided);
        let gt = run("o30", Alternative::Greater);
        let lt = run("o30", Alternative::Less);
        let o99 = run("o99", Alternative::TwoSided);
        let o99b = run("o99b", Alternative::TwoSided);
        for r in [&two, &gt, &lt, &o99, &o99b] {
            assert!(r.exact && r.warning.is_none());
            assert_eq!(r.test_name, "Exact one-sample Kolmogorov-Smirnov test");
        }
        Check::new(&case)
            .scalar("o30_D", two.statistic, Tol::stat(30))
            .scalar("o30_p_exact", two.p_value, Tol::p_value(30))
            .scalar("o30_Dplus", gt.statistic, Tol::stat(30))
            .scalar("o30_p_greater_exact", gt.p_value, Tol::p_value(30))
            .scalar("o30_Dminus", lt.statistic, Tol::stat(30))
            .scalar("o30_p_less_exact", lt.p_value, Tol::p_value(30))
            .scalar("o99_D", o99.statistic, Tol::stat(99))
            .scalar("o99_p_exact", o99.p_value, Tol::p_value(99))
            .scalar("o99b_p_exact", o99b.p_value, Tol::p_value(99))
            .finish();
    }

    /// Exact two-sample p conditional on ties (R's psmirnov). `less` with ties is the D-
    /// distribution, checked against the enumeration in the case. Fails before: asymptotic
    /// p with Stephens' factor.
    #[test]
    fn test_validate_ks_two_sample_exact_t4() {
        let (case, ds) = crate::test_support::load_case("stat_rank_ks_t4");
        let run = |set: &str, alt| {
            ks_test_two_sample(&t4(&ds, set, "x"), &t4(&ds, set, "y"), alt).unwrap()
        };
        let (tt, gt, lt) = (
            Alternative::TwoSided,
            Alternative::Greater,
            Alternative::Less,
        );
        let t75 = run("t75", tt);
        let t75_l = run("t75", lt);
        let e99 = run("e99", tt);
        for r in [&t75, &t75_l, &e99] {
            assert!(r.exact && r.warning.is_none());
            assert_eq!(r.test_name, "Exact two-sample Kolmogorov-Smirnov test");
        }
        // deviation STAT-27: t75_p_less_r (R evaluates the D+ distribution for "less") is not
        // compared; t75_p_less_exact is the enumerated permutation p of D-.
        Check::new(&case)
            .scalar("t75_D", t75.statistic, Tol::stat(11))
            .scalar("t75_p_exact", t75.p_value, Tol::p_value(11))
            .scalar(
                "t75_p_greater_exact",
                run("t75", gt).p_value,
                Tol::p_value(11),
            )
            .scalar("t75_Dminus", t75_l.statistic, Tol::stat(11))
            .scalar("t75_p_less_exact", t75_l.p_value, Tol::p_value(11))
            .scalar("t57_p_exact", run("t57", tt).p_value, Tol::p_value(12))
            .scalar(
                "t57_p_greater_exact",
                run("t57", gt).p_value,
                Tol::p_value(12),
            )
            .scalar("e99_D", e99.statistic, Tol::stat(198))
            .scalar("e99_p_exact", e99.p_value, Tol::p_value(198))
            .scalar(
                "e99_p_greater_exact",
                run("e99", gt).p_value,
                Tol::p_value(198),
            )
            .scalar(
                "e99_p_less_exact",
                run("e99", lt).p_value,
                Tol::p_value(198),
            )
            .finish();
    }

    /// Asymptotic p-values: ties in the one-sample test (R's warning), R's pkstwo series with
    /// no cut-off at x >= 3, and no Stephens factor in the two-sample test.
    #[test]
    fn test_validate_ks_asymptotic_t4() {
        let (case, ds) = crate::test_support::load_case("stat_rank_ks_t4");
        let std = TheoreticalDistribution::Normal;
        let one = |set: &str, alt, exact| {
            ks_test_one_sample_with(&t4(&ds, set, "x"), std, alt, exact).unwrap()
        };
        let o20t = one("o20t", Alternative::TwoSided, None);
        assert!(!o20t.exact);
        assert_eq!(
            o20t.test_name,
            "Asymptotic one-sample Kolmogorov-Smirnov test"
        );
        assert_eq!(
            o20t.warning.as_deref(),
            Some("ties should not be present for the Kolmogorov-Smirnov test")
        );
        let a150 = one("a150", Alternative::TwoSided, Some(false));
        let a400 = one("a400", Alternative::TwoSided, Some(false));
        assert!(400f64.sqrt() * a400.statistic > 3.0 && a400.p_value > 0.0);
        let a400_g = one("a400", Alternative::Greater, Some(false));
        let two =
            |alt| ks_test_two_sample(&t4(&ds, "a100", "x"), &t4(&ds, "a100", "y"), alt).unwrap();
        let a100 = two(Alternative::TwoSided);
        assert!(!a100.exact && a100.warning.is_none());
        assert_eq!(
            a100.test_name,
            "Asymptotic two-sample Kolmogorov-Smirnov test"
        );
        Check::new(&case)
            .scalar("o20t_D", o20t.statistic, Tol::stat(20))
            .scalar("o20t_p_asym", o20t.p_value, Tol::p_value(20))
            .scalar("a150_p_asym", a150.p_value, Tol::p_value(150))
            .scalar("a400_p_asym", a400.p_value, Tol::p_value(400))
            .scalar("a400_p_greater_asym", a400_g.p_value, Tol::p_value(400))
            .scalar("a100_D", a100.statistic, Tol::stat(200))
            .scalar("a100_p_asym", a100.p_value, Tol::p_value(200))
            .scalar(
                "a100_p_less_asym",
                two(Alternative::Less).p_value,
                Tol::p_value(200),
            )
            .finish();
    }

    /// One-sided exact p-values on the stat_ks_t3 columns (R's default exact).
    #[test]
    fn test_validate_ks_t3_one_sided_exact_t4() {
        use super::TheoreticalDistribution::*;
        let (case, ds) = crate::test_support::load_case("stat_ks_t3_exact_t4");
        let mut c = Check::new(&case);
        for (tag, col, dist) in [
            ("norm", "xn", NormalParams { mean: 2.0, sd: 2.0 }),
            ("unif", "xu", UniformParams { a: -1.5, b: 3.0 }),
            ("exp", "xe", Exponential { rate: 0.8 }),
        ] {
            for (alt, side) in [
                (Alternative::Greater, "greater"),
                (Alternative::Less, "less"),
            ] {
                let r = ks_test(&ds, col, None, Some(dist), alt).unwrap();
                assert!(r.exact);
                c.scalar(
                    &format!("{tag}_p_{side}_exact"),
                    r.p_value,
                    Tol::p_value(r.n),
                );
            }
        }
        for (tag, x, y) in [("two5", "x5", "y5"), ("twol", "xa", "ya")] {
            for (alt, side) in [
                (Alternative::Greater, "greater"),
                (Alternative::Less, "less"),
            ] {
                let r = ks_test(&ds, x, Some(y), None, alt).unwrap();
                assert!(r.exact);
                c.scalar(
                    &format!("{tag}_p_{side}_exact"),
                    r.p_value,
                    Tol::p_value(r.n),
                );
            }
        }
        c.finish();
    }

    /// The exact two-sample distribution equals the permutation distribution of D, D+ and
    /// D- over all splits of small pooled samples with ties.
    #[test]
    fn test_psmirnov_exact_matches_enumeration() {
        let cases: [(&[f64], &[f64]); 3] = [
            (&[1.5, 2.0, 2.0, 3.5, 5.0, 5.5, 8.0], &[2.0, 3.0, 3.0, 4.0]),
            (
                &[1.0, 2.0, 2.0, 3.0, 5.0],
                &[2.0, 3.0, 3.0, 4.0, 6.0, 6.0, 7.0],
            ),
            (&[0.3, 1.1, 1.1, 2.4], &[0.3, 0.9, 1.1, 2.4, 2.4, 3.8]),
        ];
        for (x, y) in cases {
            let (m, n) = (x.len(), y.len());
            let pooled: Vec<f64> = x.iter().chain(y).copied().collect();
            let total = m + n;
            let stat = |sel: u32| {
                let xs: Vec<f64> = (0..total)
                    .filter(|i| sel >> i & 1 == 1)
                    .map(|i| pooled[i])
                    .collect();
                let ys: Vec<f64> = (0..total)
                    .filter(|i| sel >> i & 1 == 0)
                    .map(|i| pooled[i])
                    .collect();
                compute_two_sample_d(&xs, &ys)
            };
            let observed = compute_two_sample_d(x, y);
            let observed_sel: u32 = (1u32 << m) - 1;
            assert_eq!(stat(observed_sel), observed);
            let (mut hit_two, mut hit_plus, mut hit_minus, mut count) = (0, 0, 0, 0);
            for sel in 0u32..(1 << total) {
                if sel.count_ones() as usize != m {
                    continue;
                }
                count += 1;
                let (dp, dm) = stat(sel);
                if dp.max(dm) >= observed.0.max(observed.1) - 1e-12 {
                    hit_two += 1;
                }
                if dp >= observed.0 - 1e-12 {
                    hit_plus += 1;
                }
                if dm >= observed.1 - 1e-12 {
                    hit_minus += 1;
                }
            }
            for (alt, hits) in [
                (Alternative::TwoSided, hit_two),
                (Alternative::Greater, hit_plus),
                (Alternative::Less, hit_minus),
            ] {
                let r = ks_test_two_sample(x, y, alt).unwrap();
                let want = hits as f64 / count as f64;
                assert!(
                    (r.p_value - want).abs() < 1e-12,
                    "{alt:?}: {} vs {want}",
                    r.p_value
                );
            }
        }
    }
}
