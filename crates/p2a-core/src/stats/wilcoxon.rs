//! Wilcoxon Rank Sum and Signed Rank Tests.
//!
//! Non-parametric tests for comparing distributions without assuming normality.
//!
//! # References
//!
//! - Wilcoxon, F. (1945). "Individual Comparisons by Ranking Methods".
//!   *Biometrics Bulletin*, 1(6), 80-83.
//! - Mann, H. B. & Whitney, D. R. (1947). "On a Test of Whether one of Two
//!   Random Variables is Stochastically Larger than the Other".
//!   *Annals of Mathematical Statistics*, 18(1), 50-60.
//! - R Core Team. `stats::wilcox.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/wilcox.test.html>
//!
//! # Mathematical Background
//!
//! ## Wilcoxon Rank Sum Test (Mann-Whitney U)
//!
//! For two independent samples X and Y:
//!
//! ```text
//! 1. Pool both samples and rank all values from 1 to n₁+n₂
//! 2. W = sum of ranks in sample X
//! 3. U = n₁n₂ + n₁(n₁+1)/2 - W (Mann-Whitney U statistic)
//!
//! Under H₀ (same distribution):
//! E(W) = n₁(n₁+n₂+1) / 2
//! Var(W) = n₁n₂(n₁+n₂+1) / 12
//!
//! With ties (using average ranks):
//! Var(W) = n₁n₂/12 × [(N+1) - Σtᵢ(tᵢ²-1)/(N(N-1))]
//!
//! Normal approximation:
//! z = (W - E(W) ± 0.5) / √Var(W)
//! ```
//!
//! ## Wilcoxon Signed Rank Test
//!
//! For paired samples or one-sample location test:
//!
//! ```text
//! 1. Compute differences: dᵢ = xᵢ - μ (or xᵢ - yᵢ for paired)
//! 2. Remove zero differences
//! 3. Rank absolute differences |dᵢ|
//! 4. V = sum of ranks where dᵢ > 0 (V⁺)
//!
//! Under H₀ (median = μ or no difference):
//! E(V) = n(n+1) / 4
//! Var(V) = n(n+1)(2n+1) / 24
//!
//! With ties:
//! Var(V) = n(n+1)(2n+1)/24 - Σtᵢ(tᵢ-1)(tᵢ+1)/48
//!
//! Normal approximation:
//! z = (V - E(V) ± 0.5) / √Var(V)
//! ```

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy, numeric_column};
use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, try_vec, try_with_capacity};
use crate::stats::ranks::{midranks, tie_cube_sum};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::{
    normal_cdf, normal_p_value_two_sided, normal_quantile, validate_level,
};

/// Result of a Wilcoxon test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Test Type
    // ═══════════════════════════════════════════════════════════════════════
    /// Description of the test performed
    pub test_name: String,
    /// Alternative hypothesis type
    pub alternative: Alternative,

    // ═══════════════════════════════════════════════════════════════════════
    // Test Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Test statistic as R reports it: W = R1 - n1 (n1 + 1) / 2 for the rank-sum test, the
    /// Mann-Whitney U of `x` (R1 is the rank sum of `x - mu` in the pooled sample), and V, the
    /// sum of the ranks of the positive differences, for the signed-rank test.
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Whether exact p-value was computed (vs normal approximation)
    pub exact: bool,
    /// Whether continuity correction was applied
    pub continuity_correction: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Z-score (for normal approximation)
    pub z_score: Option<f64>,
    /// Mann-Whitney U of `x` for the rank-sum test (equal to `statistic`, R's W); `None` for
    /// the signed-rank test.
    pub u_statistic: Option<f64>,
    /// Rank sum R1 of `x - mu` in the pooled sample (rank-sum test); `None` for the signed-rank
    /// test.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_sum: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Estimates
    // ═══════════════════════════════════════════════════════════════════════
    /// Location estimate (pseudomedian or Hodges-Lehmann estimate)
    pub estimate: Option<f64>,
    /// Null hypothesis value
    pub null_value: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Confidence Interval (if requested)
    // ═══════════════════════════════════════════════════════════════════════
    /// Confidence level (e.g., 0.95)
    pub conf_level: Option<f64>,
    /// Lower bound of confidence interval
    pub conf_int_lower: Option<f64>,
    /// Upper bound of confidence interval
    pub conf_int_upper: Option<f64>,

    // ═══════════════════════════════════════════════════════════════════════
    // Sample Info
    // ═══════════════════════════════════════════════════════════════════════
    /// Observations used: for the rank-sum test the finite values of `x`; for the signed-rank
    /// test the finite values (one sample) or finite pairs (paired test), zero differences
    /// included.
    pub n: usize,
    /// Rank-sum test: the finite values of `y`. Paired test: the number of finite pairs (as
    /// `n`). One-sample test: `None`.
    pub n_2: Option<usize>,
    /// Tied observations: the size of every tie group of two or more among the pooled ranks
    /// (rank sum) or the ranks of `|d|` of the non-zero differences (signed rank), summed.
    /// The signed-rank test drops zero differences first, so its non-zero count is `n` minus
    /// the zeros.
    pub n_ties: usize,
    /// R's warnings, joined with "; " (ties or zeros on a requested or default exact path,
    /// intervals that cannot reach the requested level).
    pub warning: Option<String>,
    /// Values dropped because they are missing (R's NA handling of `wilcox.test`); empty
    /// for the array-level functions.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for WilcoxonResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Test statistic
        let signed_rank = self.rank_sum.is_none();
        let stat_name = if signed_rank { "V" } else { "W" };
        write!(
            f,
            "{} = {:.1}, p-value = {:.6} {}",
            stat_name,
            self.statistic,
            self.p_value,
            self.significance.stars()
        )?;

        if let Some(z) = self.z_score {
            write!(f, " (z = {:.4})", z)?;
        }
        writeln!(f)?;

        if let Some(u) = self.u_statistic {
            writeln!(f, "Mann-Whitney U = {:.1}", u)?;
        }

        if !self.exact {
            writeln!(
                f,
                "(Normal approximation{})",
                if self.continuity_correction {
                    " with continuity correction"
                } else {
                    ""
                }
            )?;
        }
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => {
                format!("true location shift is not equal to {}", self.null_value)
            }
            Alternative::Greater => {
                format!("true location shift is greater than {}", self.null_value)
            }
            Alternative::Less => format!("true location shift is less than {}", self.null_value),
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Confidence interval (if computed)
        if let (Some(cl), Some(lo), Some(hi)) =
            (self.conf_level, self.conf_int_lower, self.conf_int_upper)
        {
            writeln!(f, "{:.0}% confidence interval:", cl * 100.0)?;
            writeln!(f, "  ({:.6}, {:.6})", lo, hi)?;
            writeln!(f)?;
        }

        // Estimate
        if let Some(est) = self.estimate {
            writeln!(f, "Sample estimate:")?;
            if signed_rank {
                writeln!(f, "  (pseudo)median: {:.6}", est)?;
            } else {
                writeln!(f, "  difference in location: {:.6}", est)?;
            }
            writeln!(f)?;
        }

        // Warning
        if let Some(ref warn) = self.warning {
            writeln!(f, "Warning: {}", warn)?;
        }

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        // The rank-sum test's samples drop their missing values separately; paired rows are
        // dropped jointly.
        let note = if !signed_rank {
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

/// Configuration for Wilcoxon tests.
#[derive(Debug, Clone)]
pub struct WilcoxonConfig {
    /// Whether to compute exact p-value (None = auto-decide based on sample size)
    pub exact: Option<bool>,
    /// Whether to apply continuity correction for normal approximation
    pub correct: bool,
    /// Whether to compute confidence interval and location estimate
    pub conf_int: bool,
    /// Confidence level (default: 0.95)
    pub conf_level: f64,
}

impl Default for WilcoxonConfig {
    fn default() -> Self {
        Self {
            exact: None, // Auto-decide
            correct: true,
            conf_int: false,
            conf_level: 0.95,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wilcoxon Rank Sum Test (Mann-Whitney U)
// ═══════════════════════════════════════════════════════════════════════════

/// Perform Wilcoxon rank sum test (Mann-Whitney U test) for two independent samples.
///
/// Tests whether two independent samples come from the same distribution,
/// specifically whether one tends to have larger values than the other.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Second sample data
/// * `mu` - Hypothesized location shift (default: 0)
/// * `alternative` - Direction of alternative hypothesis
/// * `config` - Test configuration
///
/// # Example
/// ```ignore
/// let x = vec![1.2, 2.3, 3.1, 4.5, 5.2];
/// let y = vec![2.1, 3.4, 4.2, 5.8, 6.1];
/// let result = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &WilcoxonConfig::default())?;
/// println!("{}", result);
/// ```
///
/// # References
/// - R equivalent: `wilcox.test(x, y, paired = FALSE)`
/// - Also known as Mann-Whitney U test or Mann-Whitney-Wilcoxon test
pub fn wilcoxon_rank_sum(
    x: &[f64],
    y: &[f64],
    mu: f64,
    alternative: Alternative,
    config: &WilcoxonConfig,
) -> EconResult<WilcoxonResult> {
    if !mu.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!("mu must be finite, got {mu} (R: 'mu' must be a single number)"),
        });
    }
    if config.conf_int {
        validate_level(config.conf_level, "conf_level")?;
    }
    // Filter out non-finite values
    let x: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let y: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();

    let n1 = x.len();
    let n2 = y.len();

    if n1 == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "First sample is empty after removing non-finite values".to_string(),
        });
    }
    if n2 == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Second sample is empty after removing non-finite values".to_string(),
        });
    }

    // R: r <- rank(c(x - mu, y)); W = R1 - n1 (n1 + 1) / 2.
    let mut pooled: Vec<f64> = try_with_capacity(n1 + n2, "Wilcoxon pooled sample")?;
    pooled.extend(x.iter().map(|&v| v - mu));
    pooled.extend_from_slice(&y);
    let (ranks, sizes) = midranks(&pooled)?;
    let r1: f64 = ranks[..n1].iter().sum();
    let (m, n) = (n1 as f64, n2 as f64);
    let w = r1 - m * (m + 1.0) / 2.0;
    let n_ties: usize = sizes.iter().filter(|&&t| t > 1).sum();

    let ties = n_ties > 0;
    // R: exact <- (n.x < 50) && (n.y < 50) unless given; the exact path needs no ties.
    let want_exact = config.exact.unwrap_or(n1 < 50 && n2 < 50);
    let exact_used = want_exact && !ties;
    let mut warnings: Vec<&'static str> = Vec::new();

    let mut xs = x.clone();
    xs.sort_by(f64::total_cmp);
    let mut ys = y.clone();
    ys.sort_by(f64::total_cmp);

    let (p_value, z_score, interval) = if exact_used {
        let dist = wilcox_dist(n1, n2)?;
        let p = exact_p(
            w,
            m * n / 2.0,
            alternative,
            |q| dist.lower(q),
            |q| dist.upper(q),
        );
        let interval = config
            .conf_int
            .then(|| rank_sum_exact_interval(&xs, &ys, &dist, config.conf_level, alternative));
        (p, None, interval)
    } else {
        let nt = m + n;
        let sigma =
            ((m * n / 12.0) * ((nt + 1.0) - tie_cube_sum(&sizes) / (nt * (nt - 1.0)))).sqrt();
        let (p, z) = normal_p(w - m * n / 2.0, sigma, alternative, config.correct);
        let interval = if config.conf_int {
            Some(rank_sum_asymptotic_interval(
                &xs,
                &ys,
                config.conf_level,
                alternative,
                config.correct,
            )?)
        } else {
            None
        };
        (p, Some(z), interval)
    };
    if let Some(iv) = &interval {
        warnings.extend(iv.warnings.iter().copied());
    }
    if want_exact && ties {
        warnings.push("cannot compute exact p-value with ties");
        if config.conf_int {
            warnings.push("cannot compute exact confidence intervals with ties");
        }
    }

    Ok(WilcoxonResult {
        test_name: method_name("Wilcoxon rank sum", exact_used, config.correct),
        alternative,
        statistic: w,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        exact: exact_used,
        continuity_correction: config.correct && !exact_used,
        z_score,
        u_statistic: Some(w),
        rank_sum: Some(r1),
        estimate: interval.as_ref().and_then(|iv| iv.estimate),
        null_value: mu,
        conf_level: interval.as_ref().map(|iv| iv.conf_level),
        conf_int_lower: interval.as_ref().and_then(|iv| iv.lower),
        conf_int_upper: interval.as_ref().and_then(|iv| iv.upper),
        n: n1,
        n_2: Some(n2),
        n_ties,
        warning: join_warnings(&warnings),
        missing: MissingSummary::none(n1 + n2),
    })
}

/// R's warnings joined with "; ", or `None`.
fn join_warnings(warnings: &[&str]) -> Option<String> {
    if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    }
}

/// R's METHOD: "Wilcoxon rank sum exact test", "Wilcoxon rank sum test with continuity
/// correction" or "Wilcoxon rank sum test" (and the signed-rank forms).
fn method_name(kind: &str, exact: bool, correct: bool) -> String {
    if exact {
        format!("{kind} exact test")
    } else if correct {
        format!("{kind} test with continuity correction")
    } else {
        format!("{kind} test")
    }
}

/// R's normal approximation: `z = (dz - CORRECTION) / SIGMA` with `CORRECTION` =
/// `sign(dz) 0.5` (two-sided), `0.5` (greater), `-0.5` (less) when `correct`; returns
/// `(p, z)`.
fn normal_p(dz: f64, sigma: f64, alternative: Alternative, correct: bool) -> (f64, f64) {
    let correction = if correct {
        match alternative {
            Alternative::TwoSided => {
                if dz > 0.0 {
                    0.5
                } else if dz < 0.0 {
                    -0.5
                } else {
                    0.0
                }
            }
            Alternative::Greater => 0.5,
            Alternative::Less => -0.5,
        }
    } else {
        0.0
    };
    let z = (dz - correction) / sigma;
    let p = match alternative {
        Alternative::TwoSided => normal_p_value_two_sided(z),
        Alternative::Greater => normal_cdf(-z),
        Alternative::Less => normal_cdf(z),
    };
    (p.clamp(0.0, 1.0), z)
}

// ═══════════════════════════════════════════════════════════════════════════
// Wilcoxon Signed Rank Test
// ═══════════════════════════════════════════════════════════════════════════

/// Perform Wilcoxon signed rank test.
///
/// For one sample: tests whether the median differs from a hypothesized value.
/// For paired samples: tests whether the median difference is zero.
///
/// # Arguments
/// * `x` - First sample data
/// * `y` - Optional second sample (for paired test)
/// * `mu` - Hypothesized median (one-sample) or median difference (paired)
/// * `alternative` - Direction of alternative hypothesis
/// * `config` - Test configuration
///
/// # Example
/// ```ignore
/// // One-sample test
/// let x = vec![1.2, 2.3, 3.1, 4.5, 5.2];
/// let result = wilcoxon_signed_rank(&x, None, 2.5, Alternative::TwoSided, &WilcoxonConfig::default())?;
///
/// // Paired test
/// let before = vec![200.0, 190.0, 210.0, 180.0];
/// let after = vec![195.0, 188.0, 202.0, 175.0];
/// let result = wilcoxon_signed_rank(&before, Some(&after), 0.0, Alternative::TwoSided, &WilcoxonConfig::default())?;
/// ```
///
/// # References
/// - R equivalent: `wilcox.test(x, mu = 2.5)` (one-sample)
/// - R equivalent: `wilcox.test(x, y, paired = TRUE)` (paired)
pub fn wilcoxon_signed_rank(
    x: &[f64],
    y: Option<&[f64]>,
    mu: f64,
    alternative: Alternative,
    config: &WilcoxonConfig,
) -> EconResult<WilcoxonResult> {
    if !mu.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!("mu must be finite, got {mu} (R: 'mu' must be a single number)"),
        });
    }
    if config.conf_int {
        validate_level(config.conf_level, "conf_level")?;
    }
    // Compute differences
    let differences: Vec<f64> = match y {
        Some(y_vals) => {
            if x.len() != y_vals.len() {
                return Err(EconError::InvalidSpecification {
                    message: "Paired test requires samples of equal length".to_string(),
                });
            }
            x.iter()
                .zip(y_vals.iter())
                .filter(|(a, b)| a.is_finite() && b.is_finite())
                .map(|(a, b)| a - b - mu)
                .collect()
        }
        None => x
            .iter()
            .filter(|v| v.is_finite())
            .map(|&v| v - mu)
            .collect(),
    };

    if differences.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "No valid observations after removing non-finite values and zeros".to_string(),
        });
    }

    // Remove zeros (they provide no information about direction)
    let nonzero: Vec<f64> = differences.iter().copied().filter(|&d| d != 0.0).collect();
    let n_zeros = differences.len() - nonzero.len();

    if nonzero.is_empty() {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "All differences are zero".to_string(),
        });
    }

    let n = nonzero.len();
    let nf = n as f64;

    // R: r <- rank(abs(x)); V = sum(r[x > 0]).
    let mut abs_diffs: Vec<f64> = try_with_capacity(n, "Wilcoxon |d|")?;
    abs_diffs.extend(nonzero.iter().map(|d| d.abs()));
    let (ranks, sizes) = midranks(&abs_diffs)?;
    let v: f64 = ranks
        .iter()
        .zip(nonzero.iter())
        .filter(|(_, d)| **d > 0.0)
        .map(|(r, _)| *r)
        .sum();
    let n_ties: usize = sizes.iter().filter(|&&t| t > 1).sum();

    let ties = n_ties > 0;
    let zeros = n_zeros > 0;
    // R: exact <- (n < 50) unless given, n counted after removing zeros; the exact path needs
    // no ties and no zeros.
    let want_exact = config.exact.unwrap_or(n < 50);
    let exact_used = want_exact && !ties && !zeros;
    let mut warnings: Vec<&'static str> = Vec::new();

    // R adds mu back before the interval: the values whose Walsh averages are the jump points
    let mut vs: Vec<f64> = try_with_capacity(n, "Wilcoxon interval")?;
    vs.extend(nonzero.iter().map(|&d| d + mu));
    vs.sort_by(f64::total_cmp);

    let (p_value, z_score, interval) = if exact_used {
        let dist = SignrankDist(signrank_probs(n)?);
        let p = exact_p(
            v,
            nf * (nf + 1.0) / 4.0,
            alternative,
            |q| dist.lower(q),
            |q| dist.upper(q),
        );
        let interval = config
            .conf_int
            .then(|| signed_rank_exact_interval(&vs, &dist, config.conf_level, alternative));
        (p, None, interval)
    } else {
        let sigma =
            (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_cube_sum(&sizes) / 48.0).sqrt();
        let (p, z) = normal_p(
            v - nf * (nf + 1.0) / 4.0,
            sigma,
            alternative,
            config.correct,
        );
        let interval = if config.conf_int {
            Some(signed_rank_asymptotic_interval(
                &vs,
                config.conf_level,
                alternative,
                config.correct,
            )?)
        } else {
            None
        };
        (p, Some(z), interval)
    };
    if let Some(iv) = &interval {
        warnings.extend(iv.warnings.iter().copied());
    }
    if want_exact && ties {
        warnings.push("cannot compute exact p-value with ties");
        if config.conf_int {
            warnings.push("cannot compute exact confidence interval with ties");
        }
    }
    if want_exact && zeros {
        warnings.push("cannot compute exact p-value with zeroes");
        if config.conf_int {
            warnings.push("cannot compute exact confidence interval with zeroes");
        }
    }

    Ok(WilcoxonResult {
        test_name: method_name("Wilcoxon signed rank", exact_used, config.correct),
        alternative,
        statistic: v,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        exact: exact_used,
        continuity_correction: config.correct && !exact_used,
        z_score,
        u_statistic: None,
        rank_sum: None,
        estimate: interval.as_ref().and_then(|iv| iv.estimate),
        null_value: mu,
        conf_level: interval.as_ref().map(|iv| iv.conf_level),
        conf_int_lower: interval.as_ref().and_then(|iv| iv.lower),
        conf_int_upper: interval.as_ref().and_then(|iv| iv.upper),
        // The finite observations (pairs) used, zero differences included.
        n: differences.len(),
        n_2: y.map(|_| differences.len()),
        n_ties,
        warning: join_warnings(&warnings),
        missing: MissingSummary::none(differences.len()),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Dataset Interface
// ═══════════════════════════════════════════════════════════════════════════

/// Perform Wilcoxon test using dataset columns.
///
/// Missing values (null or NaN) are dropped as `wilcox.test` drops them and reported in
/// `missing`: the paired test keeps the complete pairs, the rank-sum test drops each
/// sample's missing values, the one-sample test drops them from the column. Integer
/// columns are accepted; a text column is `NonNumericColumn`.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `x_col` - Name of first variable column
/// * `y_col` - Optional name of second variable column
/// * `mu` - Hypothesized location shift or median
/// * `alternative` - Direction of alternative hypothesis
/// * `paired` - If true, perform paired (signed rank) test
/// * `config` - Test configuration
///
/// # Example
/// ```ignore
/// // Two-sample rank sum test
/// let result = wilcoxon_test(&dataset, "x", Some("y"), 0.0, Alternative::TwoSided, false, &config)?;
///
/// // Paired signed rank test
/// let result = wilcoxon_test(&dataset, "before", Some("after"), 0.0, Alternative::TwoSided, true, &config)?;
///
/// // One-sample signed rank test
/// let result = wilcoxon_test(&dataset, "x", None, 5.0, Alternative::Greater, false, &config)?;
/// ```
pub fn wilcoxon_test(
    dataset: &Dataset,
    x_col: &str,
    y_col: Option<&str>,
    mu: f64,
    alternative: Alternative,
    paired: bool,
    config: &WilcoxonConfig,
) -> EconResult<WilcoxonResult> {
    let df = dataset.df();
    match y_col {
        Some(y_name) if paired => {
            // Incomplete pairs are dropped jointly (R: complete.cases(x, y)).
            let cols = ColumnRequest::new(df)
                .numeric(x_col)
                .numeric(y_name)
                .na(NaPolicy::Omit)
                .extract()?;
            let mut result = wilcoxon_signed_rank(
                cols.numeric(x_col),
                Some(cols.numeric(y_name)),
                mu,
                alternative,
                config,
            )?;
            result.missing = cols.missing().clone();
            Ok(result)
        }
        Some(y_name) => {
            // Two independent samples: each drops its own missing values (R).
            let (x, missing_x) = numeric_column(df, x_col, NaPolicy::Omit)?;
            let (y, missing_y) = numeric_column(df, y_name, NaPolicy::Omit)?;
            let mut result = wilcoxon_rank_sum(&x, &y, mu, alternative, config)?;
            result.missing = MissingSummary::merge_samples(&missing_x, &missing_y);
            Ok(result)
        }
        None => {
            // One-sample signed rank test
            let (x, missing) = numeric_column(df, x_col, NaPolicy::Omit)?;
            let mut result = wilcoxon_signed_rank(&x, None, mu, alternative, config)?;
            result.missing = missing;
            Ok(result)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions: Exact Null Distributions
// ═══════════════════════════════════════════════════════════════════════════

/// Null distribution of the Mann-Whitney U of samples of sizes `m` and `n` (R's `dwilcox`),
/// as exact integer counts or, past the i128 range, as probabilities.
#[derive(Debug, Clone)]
pub(crate) enum WilcoxDist {
    /// Counts of U = 0..=m n and their total C(m + n, m).
    Counts(Vec<i128>, i128),
    /// Probabilities of U = 0..=m n.
    Probs(Vec<f64>),
}

impl WilcoxDist {
    fn len(&self) -> usize {
        match self {
            WilcoxDist::Counts(c, _) => c.len(),
            WilcoxDist::Probs(p) => p.len(),
        }
    }

    /// P(U = u).
    fn pmf(&self, u: usize) -> f64 {
        match self {
            WilcoxDist::Counts(c, total) => c[u] as f64 / *total as f64,
            WilcoxDist::Probs(p) => p[u],
        }
    }

    /// Sum of the mass of `range`, summed in integers before the one division when exact.
    fn mass(&self, range: std::ops::Range<usize>) -> f64 {
        match self {
            WilcoxDist::Counts(c, total) => {
                let s: i128 = c[range].iter().sum();
                s as f64 / *total as f64
            }
            WilcoxDist::Probs(p) => p[range].iter().sum::<f64>().min(1.0),
        }
    }
}

/// P(S <= q) of an integer distribution with mass on 0..len (R: `floor(q + 1e-7)`).
fn dist_lower(len: usize, mass: impl Fn(std::ops::Range<usize>) -> f64, q: f64) -> f64 {
    let q = (q + 1e-7).floor();
    if q < 0.0 {
        0.0
    } else if q >= (len - 1) as f64 {
        1.0
    } else {
        mass(0..q as usize + 1)
    }
}

/// P(S > q), summed directly over the upper tail.
fn dist_upper(len: usize, mass: impl Fn(std::ops::Range<usize>) -> f64, q: f64) -> f64 {
    let q = (q + 1e-7).floor();
    if q < 0.0 {
        1.0
    } else if q >= (len - 1) as f64 {
        0.0
    } else {
        mass(q as usize + 1..len)
    }
}

/// R's `qwilcox`/`qsignrank` search: the smallest q with P(S <= q) >= p, with R's fuzz of
/// `10 * DBL_EPSILON` and its upper-half branch.
fn dist_quantile(len: usize, pmf: impl Fn(usize) -> f64, p: f64) -> usize {
    let max = len - 1;
    let mut cum = 0.0;
    if p <= 0.5 {
        let x = p - 10.0 * f64::EPSILON;
        for q in 0..=max {
            cum += pmf(q);
            if cum >= x {
                return q;
            }
        }
        max
    } else {
        let x = 1.0 - p + 10.0 * f64::EPSILON;
        for q in 0..=max {
            cum += pmf(q);
            if cum > x {
                return max - q;
            }
        }
        0
    }
}

impl WilcoxDist {
    fn lower(&self, q: f64) -> f64 {
        dist_lower(self.len(), |r| self.mass(r), q)
    }
    fn upper(&self, q: f64) -> f64 {
        dist_upper(self.len(), |r| self.mass(r), q)
    }
    fn quantile(&self, p: f64) -> usize {
        dist_quantile(self.len(), |u| self.pmf(u), p)
    }
}

/// log2 C(n, k).
fn log2_choose(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (0..k)
        .map(|i| ((n - i) as f64 / (i + 1) as f64).log2())
        .sum()
}

/// Exact counts of the Mann-Whitney U (sizes `m`, `n`) as the coefficients of the Gaussian
/// binomial `prod_{i=1}^{a} (1 - q^{b+i}) / (1 - q^i)`, a = min(m, n), b = max(m, n), in
/// i128: each completed factor is itself a Gaussian binomial with coefficients at most
/// C(m + n, a), so nothing overflows while C(m + n, a) < 2^125. `None` above that.
pub(crate) fn wilcox_cdf_counts(m: usize, n: usize) -> EconResult<Option<(Vec<i128>, i128)>> {
    let (a, b) = (m.min(n), m.max(n));
    if log2_choose(a + b, a) >= 124.0 {
        return Ok(None);
    }
    let ab = checked_len(&[a, b], "Wilcoxon exact distribution")?;
    let len = ab
        .checked_add(a + 1)
        .ok_or_else(|| EconError::ResourceLimit("Wilcoxon exact distribution".to_string()))?;
    let mut poly: Vec<i128> = try_vec(len, 0i128, "Wilcoxon exact distribution")?;
    poly[0] = 1;
    let mut deg = 0usize;
    for i in 1..=a {
        checkpoint()?;
        let step = b + i;
        let top = deg + step;
        for k in (step..=top).rev() {
            poly[k] -= poly[k - step];
        }
        for k in i..=top {
            poly[k] += poly[k - i];
        }
        deg = i * b;
    }
    poly.truncate(ab + 1);
    let total: i128 = poly.iter().sum();
    Ok(Some((poly, total)))
}

/// Probabilities of the Mann-Whitney U (sizes `m`, `n`) by the recursion
/// `p(m, n, u) = m/(m+n) p(m-1, n, u-n) + n/(m+n) p(m, n-1, u)` in f64, for sizes whose
/// counts pass the i128 range (exact requested far above R's default range).
fn wilcox_probs_dp(m: usize, n: usize) -> EconResult<Vec<f64>> {
    let (a, b) = (m.min(n), m.max(n));
    // rows[r] holds P(U = u) for samples (r, n') at the current n'
    let mut rows: Vec<Vec<f64>> = try_with_capacity(a + 1, "Wilcoxon exact distribution")?;
    for r in 0..=a {
        let len = checked_len(&[r, b], "Wilcoxon exact distribution")? + 1;
        let mut row = try_vec(len, 0.0f64, "Wilcoxon exact distribution")?;
        row[0] = 1.0;
        rows.push(row);
    }
    for np in 1..=b {
        checkpoint()?;
        for r in 1..=a {
            let (lo, hi) = rows.split_at_mut(r);
            let prev = &lo[r - 1];
            let row = &mut hi[0];
            let tot = (r + np) as f64;
            let (c1, c2) = (r as f64 / tot, np as f64 / tot);
            let top = r * np;
            for u in (0..=top).rev() {
                let from_prev = if u >= np && u - np < prev.len() {
                    prev[u - np]
                } else {
                    0.0
                };
                row[u] = c1 * from_prev + c2 * row[u];
            }
        }
    }
    Ok(rows.pop().unwrap_or_default())
}

/// The null distribution of U for sizes `m`, `n`.
fn wilcox_dist(m: usize, n: usize) -> EconResult<WilcoxDist> {
    Ok(match wilcox_cdf_counts(m, n)? {
        Some((c, total)) => WilcoxDist::Counts(c, total),
        None => WilcoxDist::Probs(wilcox_probs_dp(m, n)?),
    })
}

/// Probabilities of the signed-rank statistic V for `n` non-zero differences (R's
/// `dsignrank`): the halving recursion `p[s] = (p[s] + p[s - i]) / 2`, i = 1..n.
pub(crate) fn signrank_probs(n: usize) -> EconResult<Vec<f64>> {
    let total = checked_len(&[n, n + 1], "signed-rank exact distribution")? / 2;
    let mut p = try_vec(total + 1, 0.0f64, "signed-rank exact distribution")?;
    p[0] = 1.0;
    for i in 1..=n {
        checkpoint()?;
        let top = i * (i + 1) / 2;
        for s in (0..=top).rev() {
            let add = if s >= i { p[s - i] } else { 0.0 };
            p[s] = (p[s] + add) / 2.0;
        }
    }
    Ok(p)
}

/// The exact signed-rank distribution with R's tail and quantile conventions.
struct SignrankDist(Vec<f64>);

impl SignrankDist {
    fn mass(&self, r: std::ops::Range<usize>) -> f64 {
        self.0[r].iter().sum::<f64>().min(1.0)
    }
    fn lower(&self, q: f64) -> f64 {
        dist_lower(self.0.len(), |r| self.mass(r), q)
    }
    fn upper(&self, q: f64) -> f64 {
        dist_upper(self.0.len(), |r| self.mass(r), q)
    }
    fn quantile(&self, p: f64) -> usize {
        dist_quantile(self.0.len(), |s| self.0[s], p)
    }
}

/// R's exact p of a statistic with a symmetric null distribution centred at `centre`.
fn exact_p(
    stat: f64,
    centre: f64,
    alternative: Alternative,
    lower: impl Fn(f64) -> f64,
    upper: impl Fn(f64) -> f64,
) -> f64 {
    match alternative {
        Alternative::TwoSided => {
            let p = if stat > centre {
                upper(stat - 1.0)
            } else {
                lower(stat)
            };
            (2.0 * p).min(1.0)
        }
        Alternative::Greater => upper(stat - 1.0),
        Alternative::Less => lower(stat),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions: Order Statistics Without Materialising
// ═══════════════════════════════════════════════════════════════════════════

/// Monotone u64 key of an f64 (the IEEE total order).
fn order_key(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 1 { !b } else { b | (1 << 63) }
}

/// Inverse of [`order_key`].
fn from_order_key(k: u64) -> f64 {
    if k >> 63 == 1 {
        f64::from_bits(k & !(1 << 63))
    } else {
        f64::from_bits(!k)
    }
}

/// The smallest value `t` in `[lo, hi]` with `count_le(t) >= k`, by bisection over the f64
/// total-order keys (at most 64 probes). When `count_le` counts computed values, the result
/// is the k-th smallest of them.
fn select_by_count(lo: f64, hi: f64, k: usize, count_le: impl Fn(f64) -> usize) -> f64 {
    let (mut a, mut b) = (order_key(lo), order_key(hi));
    while a < b {
        let mid = a + (b - a) / 2;
        if count_le(from_order_key(mid)) >= k {
            b = mid;
        } else {
            a = mid + 1;
        }
    }
    from_order_key(a)
}

/// The k-th smallest (1-based) of the pairwise differences `x_i - y_j` of two sorted
/// samples, as R's `sort(outer(x, y, "-"))[k]`, in O((n1 + n2) * 64) time and no extra
/// memory.
pub(crate) fn kth_pairwise_difference(xs: &[f64], ys: &[f64], k: usize) -> f64 {
    let (m, n) = (xs.len(), ys.len());
    let count_le = |t: f64| {
        let mut j = 0usize;
        let mut count = 0usize;
        for &xi in xs {
            while j < n && xi - ys[j] > t {
                j += 1;
            }
            count += n - j;
        }
        count
    };
    select_by_count(xs[0] - ys[n - 1], xs[m - 1] - ys[0], k, count_le)
}

/// The k-th smallest (1-based) of the Walsh averages `(x_i + x_j) / 2`, i <= j, of a sorted
/// sample, as R's `sort(outer(x, x, "+")[!lower.tri(...)]) / 2`, in O(n * 64) time.
pub(crate) fn kth_walsh_average(xs: &[f64], k: usize) -> f64 {
    let n = xs.len();
    let count_le = |t: f64| {
        let mut hi = n;
        let mut count = 0usize;
        for i in 0..n {
            while hi > i && (xs[i] + xs[hi - 1]) / 2.0 > t {
                hi -= 1;
            }
            if hi <= i {
                break;
            }
            count += hi - i;
        }
        count
    };
    select_by_count(xs[0], xs[n - 1], k, count_le)
}

/// Median of `total` order statistics given by `kth` (R's `median`: the mean of the two
/// middle values when `total` is even).
fn median_of(total: usize, kth: impl Fn(usize) -> f64) -> f64 {
    if total.is_multiple_of(2) {
        (kth(total / 2) + kth(total / 2 + 1)) / 2.0
    } else {
        kth(total.div_ceil(2))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions: Confidence Intervals and Estimates
// ═══════════════════════════════════════════════════════════════════════════

/// A confidence interval with its estimate, achieved level and R's warnings.
#[derive(Debug, Clone, Default)]
struct Interval {
    lower: Option<f64>,
    upper: Option<f64>,
    estimate: Option<f64>,
    conf_level: f64,
    warnings: Vec<&'static str>,
}

/// R's `CORRECTION` / `CORRECTION.CI`.
fn continuity(dz: f64, alternative: Alternative, correct: bool) -> f64 {
    if !correct {
        return 0.0;
    }
    match alternative {
        Alternative::TwoSided => {
            if dz > 0.0 {
                0.5
            } else if dz < 0.0 {
                -0.5
            } else {
                0.0
            }
        }
        Alternative::Greater => 0.5,
        Alternative::Less => -0.5,
    }
}

/// Upper normal quantile `qnorm(p, lower.tail = FALSE)`.
fn upper_quantile(p: f64) -> EconResult<f64> {
    Ok(-normal_quantile(p)?)
}

/// The first k in 1..=total with `f(k) < 0` for a non-increasing `f`; `None` if none.
fn first_negative(total: usize, f: impl Fn(usize) -> f64) -> Option<usize> {
    if total == 0 || f(total) >= 0.0 {
        return None;
    }
    let (mut lo, mut hi) = (1usize, total);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if f(mid) < 0.0 {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Some(lo)
}

/// R's `W(d)` of the rank-sum interval: the standardised statistic of `x - d` against `y`
/// with the ties of that pooled sample.
fn rank_sum_w_at(
    x: &[f64],
    y: &[f64],
    d: f64,
    alternative: Alternative,
    correct: bool,
) -> EconResult<f64> {
    let (m, n) = (x.len() as f64, y.len() as f64);
    let mut pooled: Vec<f64> = try_with_capacity(x.len() + y.len(), "Wilcoxon interval")?;
    pooled.extend(x.iter().map(|&v| v - d));
    pooled.extend_from_slice(y);
    let (ranks, sizes) = midranks(&pooled)?;
    let r1: f64 = ranks[..x.len()].iter().sum();
    let dz = r1 - m * (m + 1.0) / 2.0 - m * n / 2.0;
    let nt = m + n;
    let sigma = ((m * n / 12.0) * ((nt + 1.0) - tie_cube_sum(&sizes) / (nt * (nt - 1.0)))).sqrt();
    Ok((dz - continuity(dz, alternative, correct)) / sigma)
}

/// Tie-group sizes of a sorted slice.
fn run_sizes(sorted: &[f64]) -> Vec<usize> {
    let mut sizes = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j] == sorted[i] {
            j += 1;
        }
        sizes.push(j - i);
        i = j;
    }
    sizes
}

/// Exact rank-sum interval (R: order statistics `qwilcox(alpha/2)` and `m n - qu + 1` of the
/// sorted differences) and estimate (their median).
fn rank_sum_exact_interval(
    xs: &[f64],
    ys: &[f64],
    dist: &WilcoxDist,
    conf_level: f64,
    alternative: Alternative,
) -> Interval {
    let mn = xs.len() * ys.len();
    let alpha = 1.0 - conf_level;
    let kth = |k: usize| kth_pairwise_difference(xs, ys, k);
    let (lower, upper, achieved) = match alternative {
        Alternative::TwoSided => {
            let qu = dist.quantile(alpha / 2.0).max(1);
            let ql = mn - qu;
            (kth(qu), kth(ql + 1), 2.0 * dist.lower(qu as f64 - 1.0))
        }
        Alternative::Greater => {
            let qu = dist.quantile(alpha).max(1);
            (kth(qu), f64::INFINITY, dist.lower(qu as f64 - 1.0))
        }
        Alternative::Less => {
            let qu = dist.quantile(alpha).max(1);
            let ql = mn - qu;
            (f64::NEG_INFINITY, kth(ql + 1), dist.lower(qu as f64 - 1.0))
        }
    };
    let mut out = Interval {
        lower: Some(lower),
        upper: Some(upper),
        estimate: Some(median_of(mn, kth)),
        conf_level,
        warnings: Vec::new(),
    };
    if achieved - alpha > alpha / 2.0 {
        out.warnings.push("Requested conf.level not achievable");
        out.conf_level = 1.0 - achieved;
    }
    out
}

/// Asymptotic rank-sum interval as R's `uniroot` on `W(d)` finds it: between consecutive
/// differences `W(d)` is constant (only the ties within `x` and within `y` remain), so each
/// root is the order statistic where the corrected z crosses the quantile. Estimate: the
/// median of the differences (R's uncorrected root; the midpoint of a flat interval).
fn rank_sum_asymptotic_interval(
    xs: &[f64],
    ys: &[f64],
    conf_level: f64,
    alternative: Alternative,
    correct: bool,
) -> EconResult<Interval> {
    let (m, n) = (xs.len(), ys.len());
    let mn = m * n;
    let (mf, nf) = (m as f64, n as f64);
    let alpha = 1.0 - conf_level;
    let mumin = xs[0] - ys[n - 1];
    let mumax = xs[m - 1] - ys[0];
    let mut out = Interval {
        conf_level,
        ..Default::default()
    };
    if mumin == mumax {
        out.warnings
            .push("cannot compute confidence interval when all observations are tied");
        return Ok(out);
    }
    let nt = mf + nf;
    let within = tie_cube_sum(&run_sizes(xs)) + tie_cube_sum(&run_sizes(ys));
    let sigma = ((mf * nf / 12.0) * ((nt + 1.0) - within / (nt * (nt - 1.0)))).sqrt();
    let w_min = rank_sum_w_at(xs, ys, mumin, alternative, correct)?;
    let w_max = rank_sum_w_at(xs, ys, mumax, alternative, correct)?;
    let kth = |k: usize| kth_pairwise_difference(xs, ys, k);
    let root = |zq: f64| {
        if w_min - zq <= 0.0 {
            return mumin;
        }
        if w_max - zq >= 0.0 {
            return mumax;
        }
        let f = |k: usize| {
            let dz = (mn - k) as f64 - mf * nf / 2.0;
            (dz - continuity(dz, alternative, correct)) / sigma - zq
        };
        first_negative(mn, f).map_or(mumax, kth)
    };
    match alternative {
        Alternative::TwoSided => {
            out.lower = Some(root(upper_quantile(alpha / 2.0)?));
            out.upper = Some(root(normal_quantile(alpha / 2.0)?));
        }
        Alternative::Greater => {
            out.lower = Some(root(upper_quantile(alpha)?));
            out.upper = Some(f64::INFINITY);
        }
        Alternative::Less => {
            out.lower = Some(f64::NEG_INFINITY);
            out.upper = Some(root(normal_quantile(alpha)?));
        }
    }
    out.estimate = Some(median_of(mn, kth));
    Ok(out)
}

/// R's `signif(x, 2)`.
fn signif2(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let digits = 1 - x.abs().log10().floor() as i32;
    let scale = 10f64.powi(digits);
    (x * scale).round() / scale
}

/// Exact signed-rank interval (order statistics of the Walsh averages) and estimate.
fn signed_rank_exact_interval(
    vs: &[f64],
    dist: &SignrankDist,
    conf_level: f64,
    alternative: Alternative,
) -> Interval {
    let n = vs.len();
    let total = n * (n + 1) / 2;
    let alpha = 1.0 - conf_level;
    let kth = |k: usize| kth_walsh_average(vs, k);
    let (lower, upper, achieved) = match alternative {
        Alternative::TwoSided => {
            let qu = dist.quantile(alpha / 2.0).max(1);
            let ql = total - qu;
            (kth(qu), kth(ql + 1), 2.0 * dist.lower(qu as f64 - 1.0))
        }
        Alternative::Greater => {
            let qu = dist.quantile(alpha).max(1);
            (kth(qu), f64::INFINITY, dist.lower(qu as f64 - 1.0))
        }
        Alternative::Less => {
            let qu = dist.quantile(alpha).max(1);
            let ql = total - qu;
            (f64::NEG_INFINITY, kth(ql + 1), dist.lower(qu as f64 - 1.0))
        }
    };
    let mut out = Interval {
        lower: Some(lower),
        upper: Some(upper),
        estimate: Some(median_of(total, kth)),
        conf_level,
        warnings: Vec::new(),
    };
    if achieved - alpha > alpha / 2.0 {
        out.warnings.push("requested conf.level not achievable");
        out.conf_level = 1.0 - signif2(achieved);
    }
    out
}

/// R's `W(d)` of the signed-rank interval: zeros of `x - d` dropped, ties of `|x - d|`.
/// Returns the standardised statistic and whether its SIGMA was zero.
fn signed_rank_w_at(
    vs: &[f64],
    d: f64,
    alternative: Alternative,
    correct: bool,
) -> EconResult<(f64, bool)> {
    let mut abs_d: Vec<f64> = try_with_capacity(vs.len(), "Wilcoxon interval")?;
    let mut positive: Vec<bool> = try_with_capacity(vs.len(), "Wilcoxon interval")?;
    for &v in vs {
        let xd = v - d;
        if xd != 0.0 {
            abs_d.push(xd.abs());
            positive.push(xd > 0.0);
        }
    }
    let nx = abs_d.len() as f64;
    let (ranks, sizes) = midranks(&abs_d)?;
    let sum_pos: f64 = ranks
        .iter()
        .zip(&positive)
        .filter(|(_, p)| **p)
        .map(|(r, _)| *r)
        .sum();
    let zd = sum_pos - nx * (nx + 1.0) / 4.0;
    let sigma = (nx * (nx + 1.0) * (2.0 * nx + 1.0) / 24.0 - tie_cube_sum(&sizes) / 48.0).sqrt();
    Ok((
        (zd - continuity(zd, alternative, correct)) / sigma,
        sigma == 0.0,
    ))
}

/// R's median of a sorted slice.
fn sorted_median(vs: &[f64]) -> f64 {
    median_of(vs.len(), |k| vs[k - 1])
}

/// Asymptotic signed-rank interval, with R's doubling of alpha while the end points do not
/// bracket the quantile and its "requested conf.level not achievable" level. Estimate: the
/// median of the Walsh averages (R's uncorrected root).
fn signed_rank_asymptotic_interval(
    vs: &[f64],
    conf_level: f64,
    alternative: Alternative,
    correct: bool,
) -> EconResult<Interval> {
    const ZERO_OR_TIED: &str =
        "cannot compute confidence interval when all observations are zero or tied";
    let n = vs.len();
    let total = n * (n + 1) / 2;
    let nf = n as f64;
    let (mumin, mumax) = (vs[0], vs[n - 1]);
    let mut out = Interval {
        conf_level,
        ..Default::default()
    };
    let (w_min, zero_min) = signed_rank_w_at(vs, mumin, alternative, correct)?;
    if zero_min {
        out.warnings.push(ZERO_OR_TIED);
    }
    let w_max = if w_min.is_finite() {
        let (w, zero_max) = signed_rank_w_at(vs, mumax, alternative, correct)?;
        if zero_max && !zero_min {
            out.warnings.push(ZERO_OR_TIED);
        }
        w
    } else {
        f64::NAN
    };
    if !w_max.is_finite() {
        // R: NaN bounds at level 0 and the midrange as estimate.
        out.conf_level = 0.0;
        out.estimate = Some((mumin + mumax) / 2.0);
        return Ok(out);
    }
    let sigma =
        (nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - tie_cube_sum(&run_sizes(vs)) / 48.0).sqrt();
    let kth = |k: usize| kth_walsh_average(vs, k);
    let root = |zq: f64| {
        if w_min - zq == 0.0 {
            return mumin;
        }
        if w_max - zq == 0.0 {
            return mumax;
        }
        let f = |k: usize| {
            let zd = (total - k) as f64 - nf * (nf + 1.0) / 4.0;
            (zd - continuity(zd, alternative, correct)) / sigma - zq
        };
        first_negative(total, f).map_or(mumax, kth)
    };
    // qnorm(p, lower.tail = FALSE), NaN outside (0, 1) as R's comparisons then stop the loop
    let up = |p: f64| {
        if p < 1.0 {
            upper_quantile(p).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    };
    let lo = |p: f64| {
        if p < 1.0 {
            normal_quantile(p).unwrap_or(f64::NAN)
        } else {
            f64::NAN
        }
    };
    let mut alpha = 1.0 - conf_level;
    loop {
        let widen = match alternative {
            Alternative::TwoSided => w_min - up(alpha / 2.0) < 0.0 || w_max - lo(alpha / 2.0) > 0.0,
            Alternative::Greater => w_min - up(alpha) < 0.0,
            Alternative::Less => w_max - lo(alpha / 2.0) > 0.0,
        };
        if widen {
            alpha *= 2.0;
        } else {
            break;
        }
    }
    if alpha >= 1.0 || 1.0 - conf_level < alpha * 0.75 {
        out.conf_level = 1.0 - alpha.min(1.0);
        out.warnings.push("requested conf.level not achievable");
    }
    let med = sorted_median(vs);
    match alternative {
        Alternative::TwoSided => {
            if alpha < 1.0 {
                out.lower = Some(root(up(alpha / 2.0)));
                out.upper = Some(root(lo(alpha / 2.0)));
            } else {
                out.lower = Some(med);
                out.upper = Some(med);
            }
        }
        Alternative::Greater => {
            out.lower = Some(if alpha < 1.0 { root(up(alpha)) } else { med });
            out.upper = Some(f64::INFINITY);
        }
        Alternative::Less => {
            out.lower = Some(f64::NEG_INFINITY);
            out.upper = Some(if alpha < 1.0 { root(lo(alpha)) } else { med });
        }
    }
    out.estimate = Some(median_of(total, kth));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    // ========================================================================
    // Basic Functionality Tests
    // ========================================================================

    #[test]
    fn test_rank_sum_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        )
        .unwrap();

        // X has ranks 1-5: R1 = 15, W = R1 - 15 = 0
        assert_eq!(result.rank_sum, Some(15.0));
        assert_eq!(result.statistic, 0.0);
        assert!(result.p_value < 0.05); // Should be significant
        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));
    }

    #[test]
    fn test_rank_sum_no_difference() {
        let x = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        )
        .unwrap();

        // X has ranks 1, 3, 5, 7, 9: R1 = 25, W = 10
        assert_eq!(result.rank_sum, Some(25.0));
        assert_eq!(result.statistic, 10.0);
        // P-value should be moderate (not significant)
        assert!(result.p_value > 0.05);
    }

    #[test]
    fn test_signed_rank_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = WilcoxonConfig::default();
        let result = wilcoxon_signed_rank(&x, None, 0.0, Alternative::Greater, &config).unwrap();

        // All differences positive, all ranks contribute to V
        // V = 1 + 2 + ... + 10 = 55
        assert!((result.statistic - 55.0).abs() < 0.001);
        assert!(result.p_value < 0.01); // Highly significant
    }

    #[test]
    fn test_signed_rank_paired() {
        let before = vec![200.0, 190.0, 210.0, 180.0, 195.0];
        let after = vec![195.0, 188.0, 202.0, 175.0, 188.0];

        let config = WilcoxonConfig::default();
        let result =
            wilcoxon_signed_rank(&before, Some(&after), 0.0, Alternative::TwoSided, &config)
                .unwrap();

        // Differences: 5, 2, 8, 5, 7
        assert!(result.p_value < 0.1); // Should show some significance
        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));
    }

    #[test]
    fn test_with_ties() {
        let x = vec![1.0, 2.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 3.0, 4.0, 5.0];

        let config = WilcoxonConfig {
            exact: Some(false), // Force approximation
            ..Default::default()
        };
        let result = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &config).unwrap();

        assert!(result.n_ties > 0);
        assert!(!result.exact);
        assert!(result.z_score.is_some());
    }

    #[test]
    fn test_alternative_hypotheses() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let config = WilcoxonConfig::default();

        // Two-sided
        let r_two = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &config).unwrap();

        // Greater (x > y under H1) - should have high p-value since x < y
        let r_gt = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::Greater, &config).unwrap();

        // Less (x < y under H1) - should have low p-value
        let r_lt = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::Less, &config).unwrap();

        assert!(r_gt.p_value > r_two.p_value);
        assert!(r_lt.p_value < r_two.p_value);
    }

    #[test]
    fn test_from_dataset() {
        let df = df! {
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
            "y" => [2.0, 3.0, 4.0, 5.0, 6.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let config = WilcoxonConfig::default();

        // Rank sum test
        let result = wilcoxon_test(
            &dataset,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            false,
            &config,
        )
        .unwrap();
        assert_eq!(result.n, 5);
        assert_eq!(result.n_2, Some(5));

        // Paired test
        let result = wilcoxon_test(
            &dataset,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            true,
            &config,
        )
        .unwrap();
        assert!(result.test_name.contains("signed rank"));
    }

    #[test]
    fn test_confidence_interval() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let config = WilcoxonConfig {
            conf_int: true,
            conf_level: 0.95,
            ..Default::default()
        };
        let result = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &config).unwrap();

        assert!(result.estimate.is_some());
        assert!(result.conf_int_lower.is_some());
        assert!(result.conf_int_upper.is_some());

        let est = result.estimate.unwrap();
        let lo = result.conf_int_lower.unwrap();
        let hi = result.conf_int_upper.unwrap();

        assert!(lo < est);
        assert!(hi > est);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================
    //
    // These tests validate results against R's wilcox.test() function.

    #[test]
    fn test_validate_wilcoxon_n30_exact_flag() {
        // R's default is the exact test for each sample below 50 without ties (signed rank:
        // n < 50 non-zero differences), and the result says which path ran. Before: N = 30
        // ran the normal approximation. The normal keys stay pinned through exact = false.
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_wilcoxon");
        let x = column_where(&ds, "value", &[("set", "n30"), ("group", "x")]);
        let y = column_where(&ds, "value", &[("set", "n30"), ("group", "y")]);
        let v = column_where(&ds, "value", &[("set", "sr25"), ("group", "x")]);
        let tt = Alternative::TwoSided;
        let def = WilcoxonConfig::default();
        let normal = cfg(Some(false), true);
        let r = wilcoxon_rank_sum(&x, &y, 0.0, tt, &def).unwrap();
        let rn = wilcoxon_rank_sum(&x, &y, 0.0, tt, &normal).unwrap();
        assert_eq!((r.n, r.n_2, r.n_ties), (15, Some(15), 0));
        assert!(r.exact && !r.continuity_correction);
        assert_eq!(r.test_name, "Wilcoxon rank sum exact test");
        assert!(!rn.exact && rn.continuity_correction);
        let s = wilcoxon_signed_rank(&v, None, 0.0, tt, &def).unwrap();
        let sn = wilcoxon_signed_rank(&v, None, 0.0, tt, &normal).unwrap();
        assert_eq!((s.n, s.n_ties), (25, 0));
        assert!(s.exact);
        assert_eq!(s.test_name, "Wilcoxon signed rank exact test");
        assert!(!sn.exact && sn.continuity_correction);
        Check::new(&case)
            .scalar("n30_rank_sum", r.rank_sum.unwrap(), Tol::exact())
            .scalar("n30_W", r.statistic, Tol::exact())
            .scalar("n30_p_exact", r.p_value, Tol::p_value(30))
            .scalar("n30_p_normal", rn.p_value, Tol::p_value(30))
            .scalar("sr25_V", s.statistic, Tol::exact())
            .scalar("sr25_p_exact", s.p_value, Tol::p_value(25))
            .scalar("sr25_p_normal", sn.p_value, Tol::p_value(25))
            .finish();
    }

    #[test]
    fn test_validate_wilcoxon_examples() {
        // X-13: the old comments quoted "W = 7, p 0.08182" and "V = 40, p 0.1455"; R gives
        // W = 8, p 0.2353 and V = 33, p 0.2361. STAT-62: the old assertions accepted p in
        // (0.05, 0.35), (0.1, 0.3), (0.2, 0.8).
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_wilcoxon");
        let col = |set: &str, g: &str| column_where(&ds, "value", &[("set", set), ("group", g)]);
        let normal = WilcoxonConfig {
            exact: Some(false),
            correct: true,
            ..Default::default()
        };
        let exact = WilcoxonConfig {
            exact: Some(true),
            ..Default::default()
        };
        let a = wilcoxon_rank_sum(
            &col("x13a", "x"),
            &col("x13a", "y"),
            0.0,
            Alternative::TwoSided,
            &normal,
        )
        .unwrap();
        assert!(!a.exact && a.continuity_correction);
        let b = wilcoxon_signed_rank(&col("x13b", "x"), None, 1.5, Alternative::TwoSided, &normal)
            .unwrap();
        let e = wilcoxon_rank_sum(
            &col("exact3", "x"),
            &col("exact3", "y"),
            0.0,
            Alternative::TwoSided,
            &exact,
        )
        .unwrap();
        assert!(e.exact);
        let pr = wilcoxon_signed_rank(
            &col("paired", "x"),
            Some(&col("paired", "y")),
            0.0,
            Alternative::TwoSided,
            &normal,
        )
        .unwrap();
        Check::new(&case)
            .scalar("x13a_rank_sum", a.rank_sum.unwrap(), Tol::exact())
            .scalar("x13a_W", a.statistic, Tol::exact())
            .scalar("x13a_p", a.p_value, Tol::p_value(11))
            .scalar("x13b_V", b.statistic, Tol::exact())
            .scalar("x13b_p", b.p_value, Tol::p_value(9))
            .scalar("exact3_rank_sum", e.rank_sum.unwrap(), Tol::exact())
            .scalar("exact3_W", e.statistic, Tol::exact())
            .scalar("exact3_p", e.p_value, Tol::p_value(6))
            .scalar("paired_V", pr.statistic, Tol::exact())
            .scalar("paired_p", pr.p_value, Tol::p_value(pr.n))
            .finish();
    }

    #[test]
    fn test_validate_wilcoxon_small_scale() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_small_groups");
        let x = column_where(&ds, "ynt", &[("g", "a")]);
        let y = column_where(&ds, "ynt", &[("g", "b")]);
        let r = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        )
        .unwrap();
        assert!(r.exact && r.n_ties == 0);
        Check::new(&case)
            .scalar("wilcox_rank_sum", r.rank_sum.unwrap(), Tol::exact())
            .scalar("wilcox_W", r.statistic, Tol::exact())
            .scalar("wilcox_p_exact", r.p_value, Tol::p_value(14))
            .finish();
    }

    #[test]
    fn test_wilcoxon_rejects_invalid_conf_level_and_mu() {
        // R checks conf.level only when conf.int = TRUE, and refuses a non-finite mu. Before:
        // conf_level 1.5 or NaN panicked in statrs, 0.0 and a NaN mu returned Ok.
        let x = [1.1, 2.3, 2.9, 4.2, 5.0];
        let y = [3.4, 4.4, 5.9, 6.1, 7.7];
        for level in [1.5, 0.0, f64::NAN] {
            let cfg = WilcoxonConfig {
                conf_int: true,
                conf_level: level,
                ..Default::default()
            };
            for r in [
                wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &cfg),
                wilcoxon_signed_rank(&x, Some(&y), 0.0, Alternative::TwoSided, &cfg),
            ] {
                match r {
                    Err(EconError::InvalidSpecification { message }) => {
                        assert!(message.contains("conf_level"), "{message}")
                    }
                    other => panic!("{level}: expected InvalidSpecification, got {other:?}"),
                }
            }
        }
        let off = WilcoxonConfig {
            conf_int: false,
            conf_level: 1.5,
            ..Default::default()
        };
        assert!(wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &off).is_ok());
        for mu in [f64::NAN, f64::INFINITY] {
            for r in [
                wilcoxon_rank_sum(
                    &x,
                    &y,
                    mu,
                    Alternative::TwoSided,
                    &WilcoxonConfig::default(),
                ),
                wilcoxon_signed_rank(
                    &x,
                    None,
                    mu,
                    Alternative::TwoSided,
                    &WilcoxonConfig::default(),
                ),
            ] {
                match r {
                    Err(EconError::InvalidSpecification { message }) => {
                        assert!(message.contains("mu"), "{message}")
                    }
                    other => panic!("{mu}: expected InvalidSpecification, got {other:?}"),
                }
            }
        }
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_empty_sample() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        let result = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        );
        assert!(matches!(result, Err(EconError::InsufficientData { .. })));
    }

    #[test]
    fn test_all_zeros_signed_rank() {
        let x = vec![0.0, 0.0, 0.0];

        let result = wilcoxon_signed_rank(
            &x,
            None,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        );
        assert!(matches!(result, Err(EconError::InsufficientData { .. })));
    }

    #[test]
    fn test_single_observation() {
        let x = vec![5.0];
        let y = vec![3.0];

        let result = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        )
        .unwrap();
        assert_eq!(result.n, 1);
        assert_eq!(result.n_2, Some(1));
    }

    #[test]
    fn test_non_finite_values() {
        let x = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        let y = vec![2.0, 4.0, f64::NEG_INFINITY, 6.0];

        let result = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        )
        .unwrap();

        // Should only use finite values: x = [1, 3, 5], y = [2, 4, 6]
        assert_eq!(result.n, 3);
        assert_eq!(result.n_2, Some(3)); // y has 3 finite values: 2, 4, 6
    }

    /// The array-level functions drop non-finite values silently; their summary counts the
    /// values they used, so it agrees with `n` (and `n_2`). Fails before: the summary
    /// counted the input, and the signed-rank `n` did too.
    #[test]
    fn test_array_level_summary_counts_the_values_used() {
        let x = [1.3, f64::NAN, 3.1, f64::INFINITY, 5.2, 0.4];
        let y = [2.2, 4.1, f64::NEG_INFINITY, 6.3, 0.9, 2.8];
        let (tt, config) = (Alternative::TwoSided, WilcoxonConfig::default());

        let rs = wilcoxon_rank_sum(&x, &y, 0.0, tt, &config).unwrap();
        assert_eq!((rs.n, rs.n_2), (4, Some(5)));
        assert_eq!(
            (rs.missing.n_total, rs.missing.n_used, rs.missing.n_dropped),
            (9, 9, 0)
        );

        let one = wilcoxon_signed_rank(&x, None, 1.0, tt, &config).unwrap();
        assert_eq!(one.n, 4);
        assert_eq!((one.missing.n_total, one.missing.n_used), (4, 4));

        // Finite pairs: rows 1, 5 and 6 (x is NaN in row 2 and Inf in row 4, y -Inf in 3).
        let paired = wilcoxon_signed_rank(&x, Some(&y), 0.0, tt, &config).unwrap();
        assert_eq!((paired.n, paired.n_2), (3, Some(3)));
        assert_eq!((paired.missing.n_total, paired.missing.n_used), (3, 3));
        assert!(paired.missing.is_empty());
    }

    #[test]
    fn test_mann_whitney_u_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        let result = wilcoxon_rank_sum(
            &x,
            &y,
            0.0,
            Alternative::TwoSided,
            &WilcoxonConfig::default(),
        )
        .unwrap();

        // R1 = 1+2+3+4+5 = 15; U of x = R1 - n1 (n1 + 1) / 2 = 0 (R's W)
        assert_eq!(result.u_statistic, Some(0.0));
        assert_eq!(result.statistic, 0.0);
    }

    #[test]
    fn test_display_format() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 4.0, 5.0, 6.0, 7.0];

        let config = WilcoxonConfig {
            conf_int: true,
            ..Default::default()
        };
        let result = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &config).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Wilcoxon rank sum"), "{display}");
        assert!(display.contains("W ="));
        assert!(display.contains("p-value"));
        assert!(display.contains("confidence interval"));
        assert!(display.contains("difference in location"));

        let sr = wilcoxon_signed_rank(&x, Some(&y), 0.0, Alternative::TwoSided, &config).unwrap();
        let display = format!("{sr}");
        assert!(display.starts_with("Wilcoxon signed rank"), "{display}");
        assert!(display.contains("V = "), "{display}");
        assert!(!display.contains("W = "), "{display}");
        assert!(display.contains("(pseudo)median"), "{display}");
    }

    // ========================================================================
    // R-pinned: missing values (T2-WP07, case stat_wilcoxon_na)
    // ========================================================================
    //
    // x is missing in row 3 and y in row 8 (10 rows): the paired test keeps the 8 complete
    // pairs, the rank-sum test drops each sample's missing value (9 and 9). No ties (the
    // case checks it), so STAT-42 does not enter.

    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    fn normal_approx() -> WilcoxonConfig {
        WilcoxonConfig {
            exact: Some(false),
            correct: true,
            ..Default::default()
        }
    }

    #[test]
    fn test_validate_wilcoxon_na_paired() {
        let (case, ds) = load_case("stat_wilcoxon_na");
        let tt = Alternative::TwoSided;
        let p = wilcoxon_test(&ds, "x", Some("y"), 0.0, tt, true, &normal_approx()).unwrap();
        let e = wilcoxon_test(
            &ds,
            "x",
            Some("y"),
            0.0,
            tt,
            true,
            &WilcoxonConfig::default(),
        )
        .unwrap();
        Check::new(&case)
            .scalar("paired_v", p.statistic, Tol::exact())
            .scalar("paired_p_normal", p.p_value, Tol::p_value(8))
            .scalar("paired_z", p.z_score.unwrap(), Tol::stat(8))
            .scalar("paired_p_exact", e.p_value, Tol::p_value(8))
            .scalar("n_pairs", p.n as f64, Tol::exact())
            .finish();
        assert!(e.exact);
        assert_eq!(p.missing.n_dropped, 2);
        assert_eq!(
            p.missing.by_column,
            vec![("x".to_string(), 1), ("y".to_string(), 1)]
        );
        assert!(
            p.to_string()
                .contains("Missing values: 2 of 10 rows dropped because of missing values"),
            "{p}"
        );
    }

    #[test]
    fn test_validate_wilcoxon_na_rank_sum() {
        let (case, ds) = load_case("stat_wilcoxon_na");
        let r = wilcoxon_test(
            &ds,
            "x",
            Some("y"),
            0.0,
            Alternative::TwoSided,
            false,
            &normal_approx(),
        )
        .unwrap();
        let n_x = r.n as f64;
        Check::new(&case)
            .scalar("ranksum_r1", r.rank_sum.unwrap(), Tol::exact())
            .scalar("ranksum_w", r.statistic, Tol::exact())
            .scalar("ranksum_p_normal", r.p_value, Tol::p_value(18))
            .scalar("ranksum_z", r.z_score.unwrap(), Tol::stat(18))
            .scalar("n_x", n_x, Tol::exact())
            .scalar("n_y", r.n_2.unwrap() as f64, Tol::exact())
            .finish();
        assert_eq!(
            (r.missing.n_total, r.missing.n_used, r.missing.n_dropped),
            (20, 18, 2)
        );
        // each sample drops its own missing values: the note counts values
        assert_eq!(
            r.to_string().lines().last().map(str::to_string),
            r.missing
                .two_sample_note()
                .map(|note| format!("Missing values: {note}"))
        );
        assert!(r.to_string().contains("two samples of 10"), "{r}");
    }

    /// Column-type fixture of the dataset wrapper, on the Int64 case stat_friedman_quade: the
    /// one-sample test on the Int64 column equals the test on its Float64 cast; a text column
    /// and an absent column are refused.
    #[test]
    fn test_validate_na_wilcoxon_dtypes_and_errors() {
        let (_case, ds) = load_case("stat_friedman_quade");
        assert_eq!(ds.df().column("y").unwrap().dtype(), &DataType::Int64);
        let config = WilcoxonConfig::default();
        let tt = Alternative::TwoSided;
        let int = wilcoxon_test(&ds, "y", None, 10.0, tt, false, &config).unwrap();
        let mut df = ds.df().clone();
        let y = df.column("y").unwrap().cast(&DataType::Float64).unwrap();
        df.with_column(y).unwrap();
        let float = wilcoxon_test(&Dataset::new(df), "y", None, 10.0, tt, false, &config).unwrap();
        assert_eq!(int.statistic, float.statistic);
        assert_eq!(int.p_value, float.p_value);
        assert!(matches!(
            wilcoxon_test(&ds, "treat", None, 0.0, tt, false, &config),
            Err(EconError::NonNumericColumn { ref column }) if column == "treat"
        ));
        assert!(matches!(
            wilcoxon_test(&ds, "y", Some("nope"), 0.0, tt, true, &config),
            Err(EconError::ColumnNotFound { ref column, .. }) if column == "nope"
        ));
    }

    // ========================================================================
    // R-pinned: T4 cases (stat_rank_wilcoxon_t4)
    // ========================================================================

    fn t4_col(ds: &Dataset, set: &str, group: &str) -> Vec<f64> {
        crate::stats::ranks::test_data::column_where(ds, "value", &[("set", set), ("group", group)])
    }

    fn cfg(exact: Option<bool>, correct: bool) -> WilcoxonConfig {
        WilcoxonConfig {
            exact,
            correct,
            ..Default::default()
        }
    }

    /// W (not R1), the shift mu applied to x as R does, and the tie-corrected variance
    /// sum(t^3 - t). Fails before: the statistic was R1, mu shifted y the other way, and the
    /// variance counted tied values.
    #[test]
    fn test_validate_wilcoxon_rank_sum_t4() {
        let (case, ds) = load_case("stat_rank_wilcoxon_t4");
        let (tt, def) = (Alternative::TwoSided, WilcoxonConfig::default());
        let rs = |set: &str, mu: f64, alt: Alternative, c: &WilcoxonConfig| {
            wilcoxon_rank_sum(&t4_col(&ds, set, "x"), &t4_col(&ds, set, "y"), mu, alt, c).unwrap()
        };
        let d05 = rs("depr", 0.5, tt, &def);
        let d0 = rs("depr", 0.0, tt, &def);
        let lik = rs("lik", 0.0, tt, &def);
        let lik_nc = rs("lik", 0.0, tt, &cfg(None, false));
        let heavy = rs("heavy", 0.0, tt, &def);
        let n5510 = rs("n5510", 0.0, tt, &def);
        for r in [&d05, &d0, &lik, &heavy, &n5510] {
            assert!(!r.exact && r.continuity_correction);
            assert_eq!(
                r.test_name,
                "Wilcoxon rank sum test with continuity correction"
            );
            assert_eq!(r.u_statistic, Some(r.statistic));
        }
        assert_eq!(lik_nc.test_name, "Wilcoxon rank sum test");
        Check::new(&case)
            .scalar("depr_mu05_W", d05.statistic, Tol::exact())
            .scalar("depr_mu05_p", d05.p_value, Tol::p_value(18))
            .scalar("depr_mu0_W", d0.statistic, Tol::exact())
            .scalar("depr_mu0_p", d0.p_value, Tol::p_value(18))
            .scalar("lik_W", lik.statistic, Tol::exact())
            .scalar("lik_p", lik.p_value, Tol::p_value(24))
            .scalar("lik_p_nc", lik_nc.p_value, Tol::p_value(24))
            .scalar("lik_z", lik.z_score.unwrap(), Tol::stat(24))
            .scalar("heavy_W", heavy.statistic, Tol::exact())
            .scalar("heavy_p", heavy.p_value, Tol::p_value(36))
            .scalar("n5510_W", n5510.statistic, Tol::exact())
            .scalar("n5510_p_normal", n5510.p_value, Tol::p_value(65))
            .finish();

        // Part 2: R's per-sample exact rule (STAT-73): exact without ties for each sample
        // below 50, and when requested above that.
        let tf = rs("tf", 0.3, tt, &def);
        let tf_l = rs("tf", 0.3, Alternative::Less, &def);
        let tf_g = rs("tf", 0.3, Alternative::Greater, &def);
        let e4530 = rs("n4530", 0.0, tt, &def);
        let e4530_g = rs("n4530", 0.0, Alternative::Greater, &def);
        let e60 = rs("n60", 0.0, tt, &cfg(Some(true), true));
        for r in [&tf, &tf_l, &tf_g, &e4530, &e4530_g, &e60] {
            assert!(r.exact && r.z_score.is_none() && r.warning.is_none());
            assert_eq!(r.test_name, "Wilcoxon rank sum exact test");
        }
        assert!(!n5510.exact, "55 x 10 is outside R's default exact range");
        Check::new(&case)
            .scalar("tf_mu03_W", tf.statistic, Tol::exact())
            .scalar("tf_mu03_p", tf.p_value, Tol::p_value(18))
            .scalar("tf_mu03_p_less", tf_l.p_value, Tol::p_value(18))
            .scalar("tf_mu03_p_greater", tf_g.p_value, Tol::p_value(18))
            .scalar("n4530_W", e4530.statistic, Tol::exact())
            .scalar("n4530_p_exact", e4530.p_value, Tol::p_value(75))
            .scalar("n4530_p_greater_exact", e4530_g.p_value, Tol::p_value(75))
            .scalar("n60_W", e60.statistic, Tol::exact())
            .scalar("n60_p_exact", e60.p_value, Tol::p_value(120))
            .finish();
    }

    /// V, R's exact rule after removing zeros, the zero and tie fallbacks with R's warnings,
    /// and the tie-corrected variance. Fails before: zeros were ignored by the exact rule,
    /// the exact enumeration stopped at n = 20 and the variance counted tied values.
    #[test]
    fn test_validate_wilcoxon_signed_rank_t4() {
        let (case, ds) = load_case("stat_rank_wilcoxon_t4");
        let (tt, def) = (Alternative::TwoSided, WilcoxonConfig::default());
        let one = |set: &str, mu: f64, alt: Alternative, c: &WilcoxonConfig| {
            wilcoxon_signed_rank(&t4_col(&ds, set, "x"), None, mu, alt, c).unwrap()
        };
        let lik = wilcoxon_signed_rank(
            &t4_col(&ds, "lik", "x"),
            Some(&t4_col(&ds, "lik", "y")),
            0.0,
            tt,
            &def,
        )
        .unwrap();
        let srt = one("srt", 0.0, tt, &def);
        let srt_g = one("srt", 0.0, Alternative::Greater, &def);
        let sr40 = one("sr40", 0.0, tt, &def);
        let sr40_l = one("sr40", 0.0, Alternative::Less, &def);
        let sr60 = one("sr60", 0.0, tt, &cfg(Some(true), true));
        let srz = one("srz", 1.0, tt, &def);
        let srz_e = one("srz", 1.0, tt, &cfg(Some(true), false));
        let sr12 = one("sr12", 1.0, tt, &def);
        for r in [&sr40, &sr40_l, &sr60] {
            assert!(r.exact && r.warning.is_none());
            assert_eq!(r.test_name, "Wilcoxon signed rank exact test");
        }
        for r in [&lik, &srt, &srt_g, &srz, &sr12] {
            assert!(!r.exact && r.z_score.is_some());
            assert_eq!(
                r.test_name,
                "Wilcoxon signed rank test with continuity correction"
            );
        }
        assert_eq!(
            srz.warning.as_deref(),
            Some("cannot compute exact p-value with zeroes")
        );
        assert_eq!(
            srz_e.warning.as_deref(),
            Some("cannot compute exact p-value with zeroes")
        );
        assert_eq!(srz_e.test_name, "Wilcoxon signed rank test");
        assert_eq!(
            srt.warning.as_deref(),
            Some("cannot compute exact p-value with ties")
        );
        Check::new(&case)
            .scalar("lik_paired_V", lik.statistic, Tol::exact())
            .scalar("lik_paired_p", lik.p_value, Tol::p_value(12))
            .scalar("srt_V", srt.statistic, Tol::exact())
            .scalar("srt_p", srt.p_value, Tol::p_value(14))
            .scalar("srt_p_greater", srt_g.p_value, Tol::p_value(14))
            .scalar("sr40_V", sr40.statistic, Tol::exact())
            .scalar("sr40_p_exact", sr40.p_value, Tol::p_value(40))
            .scalar("sr40_p_less_exact", sr40_l.p_value, Tol::p_value(40))
            .scalar("sr60_V", sr60.statistic, Tol::exact())
            .scalar("sr60_p_exact", sr60.p_value, Tol::p_value(60))
            .scalar("srz_V", srz.statistic, Tol::exact())
            .scalar("srz_p", srz.p_value, Tol::p_value(11))
            .scalar("srz_p_exact_requested_nc", srz_e.p_value, Tol::p_value(11))
            .scalar("sr12_V", sr12.statistic, Tol::exact())
            .scalar("sr12_p", sr12.p_value, Tol::p_value(12))
            .finish();
        // 12 non-zero paired differences of 12 pairs
        assert_eq!(lik.n, 12);
    }

    /// Ties make the requested exact test fall back with R's warnings; the interval adds its
    /// own.
    #[test]
    fn test_wilcoxon_exact_fallback_warnings() {
        let x = [1.1, 2.0, 2.0, 3.4, 4.2];
        let y = [2.0, 3.3, 3.9, 4.4, 5.6];
        let c = WilcoxonConfig {
            conf_int: true,
            ..Default::default()
        };
        let r = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &c).unwrap();
        assert!(!r.exact);
        assert_eq!(
            r.warning.as_deref(),
            Some(
                "cannot compute exact p-value with ties; cannot compute exact confidence intervals with ties"
            )
        );
        let none =
            wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &cfg(Some(false), true)).unwrap();
        assert!(none.warning.is_none(), "no warning when exact = false");
    }

    /// The exact distributions stay probabilities at R's largest default sizes (u64 counts
    /// overflowed before): tails sum to 1 + P(W = w), and the i128 counts agree with the
    /// f64 recursion used past the i128 range.
    #[test]
    fn test_wilcoxon_exact_distribution_49_by_49() {
        let (c, total) = wilcox_cdf_counts(49, 49).unwrap().unwrap();
        assert_eq!(c.len(), 49 * 49 + 1);
        assert_eq!(c.iter().sum::<i128>(), total);
        assert!(c.iter().all(|&v| v >= 0));
        let dist = WilcoxDist::Counts(c.clone(), total);
        for w in [0.0, 700.0, 1200.0, 1500.0, 2401.0] {
            let (lo, hi) = (dist.lower(w), dist.upper(w - 1.0));
            assert!((0.0..=1.0).contains(&lo) && (0.0..=1.0).contains(&hi));
            let pw = dist.pmf(w as usize);
            assert!((lo + hi - 1.0 - pw).abs() < 1e-12, "{w}: {lo} + {hi}");
        }
        for (m, n) in [(3, 7), (12, 9), (20, 25), (49, 49)] {
            let (c, total) = wilcox_cdf_counts(m, n).unwrap().unwrap();
            let probs = wilcox_probs_dp(m, n).unwrap();
            assert_eq!(probs.len(), c.len());
            for (u, (&cu, &pu)) in c.iter().zip(&probs).enumerate() {
                let exact = cu as f64 / total as f64;
                assert!(
                    (exact - pu).abs() <= 1e-12 * exact.max(1e-300) + 1e-300,
                    "{m}x{n} u={u}: {exact} vs {pu}"
                );
            }
        }
        let sr = signrank_probs(49).unwrap();
        assert!((sr.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(
            wilcox_cdf_counts(70, 70).unwrap().is_none(),
            "past the i128 range"
        );
    }

    /// Intervals and estimates as R defines them: exact order statistics of the differences
    /// and Walsh averages, asymptotic roots at the jump points of R's step function (R's
    /// uniroot root lies within 1e-4 of each), estimates as medians. Fails before: the
    /// intervals were ad hoc.
    #[test]
    fn test_validate_wilcoxon_intervals_t4() {
        let (case, ds) = load_case("stat_rank_wilcoxon_t4");
        let ci = |level: f64, correct: bool| WilcoxonConfig {
            conf_int: true,
            conf_level: level,
            correct,
            ..Default::default()
        };
        let tt = Alternative::TwoSided;
        let rs = |set: &str, mu: f64, alt: Alternative, c: &WilcoxonConfig| {
            wilcoxon_rank_sum(&t4_col(&ds, set, "x"), &t4_col(&ds, set, "y"), mu, alt, c).unwrap()
        };
        let one = |set: &str, mu: f64, c: &WilcoxonConfig| {
            wilcoxon_signed_rank(&t4_col(&ds, set, "x"), None, mu, tt, c).unwrap()
        };
        let bounds = |r: &WilcoxonResult| [r.conf_int_lower.unwrap(), r.conf_int_upper.unwrap()];
        let tf = rs("tf", 0.3, tt, &ci(0.95, true));
        let tf90 = rs("tf", 0.3, tt, &ci(0.90, true));
        let tf_g = rs("tf", 0.3, Alternative::Greater, &ci(0.95, true));
        let tf_l = rs("tf", 0.3, Alternative::Less, &ci(0.95, true));
        assert!(tf.exact && tf.warning.is_none());
        assert_eq!(tf_g.conf_int_upper, Some(f64::INFINITY));
        assert_eq!(tf_l.conf_int_lower, Some(f64::NEG_INFINITY));
        let sr40 = one("sr40", 0.0, &ci(0.95, true));
        assert!(sr40.exact);
        let d05 = rs("depr", 0.5, tt, &ci(0.95, true));
        let d0 = rs("depr", 0.0, tt, &ci(0.95, true));
        let d0nc = rs("depr", 0.0, tt, &ci(0.95, false));
        let lik = rs("lik", 0.0, tt, &ci(0.95, true));
        let srt = one("srt", 0.0, &ci(0.95, true));
        let sr12 = one("sr12", 1.0, &ci(0.95, true));
        for r in [&d05, &d0, &d0nc, &lik, &srt, &sr12] {
            assert!(!r.exact);
            assert_eq!(r.conf_level, Some(0.95));
        }
        Check::new(&case)
            .vector("tf_ci", &bounds(&tf), Tol::stat(18))
            .vector("tf_ci90", &bounds(&tf90), Tol::stat(18))
            .scalar(
                "tf_ci_greater_lower",
                tf_g.conf_int_lower.unwrap(),
                Tol::stat(18),
            )
            .scalar(
                "tf_ci_less_upper",
                tf_l.conf_int_upper.unwrap(),
                Tol::stat(18),
            )
            .scalar("tf_est", tf.estimate.unwrap(), Tol::stat(18))
            .vector("sr40_ci", &bounds(&sr40), Tol::stat(40))
            .scalar("sr40_est", sr40.estimate.unwrap(), Tol::stat(40))
            .vector("depr_ci", &bounds(&d05), Tol::stat(18))
            .scalar("depr_est", d05.estimate.unwrap(), Tol::stat(18))
            .vector("depr_mu0_ci", &bounds(&d0), Tol::stat(18))
            .vector("depr_mu0_ci_nc", &bounds(&d0nc), Tol::stat(18))
            .vector("lik_ci", &bounds(&lik), Tol::stat(24))
            .vector("srt_ci", &bounds(&srt), Tol::stat(14))
            .vector("sr12_ci", &bounds(&sr12), Tol::stat(12))
            .finish();
        // estimates where the two middle order statistics coincide
        let mid = |key: &str| -> f64 {
            let v = case.vector(key);
            assert_eq!(v[0], v[1], "{key}");
            v[0]
        };
        for (r, key) in [
            (&lik, "lik_est_mid"),
            (&srt, "srt_est_mid"),
            (&sr12, "sr12_est_mid"),
        ] {
            assert!(Tol::stat(24).allows(r.estimate.unwrap(), mid(key)), "{key}");
        }
        // R's uniroot root (tol 1e-4) lies within 1e-4 of each jump point
        let scalar = |key: &str| case.scalar(key);
        let near = |a: f64, b: f64| (a - b).abs() < 1e-4;
        let r_ci = |key: &str| case.vector(key);
        assert!(near(bounds(&d05)[0], r_ci("depr_ci_r")[0]));
        assert!(near(bounds(&d05)[1], r_ci("depr_ci_r")[1]));
        assert!(near(d05.estimate.unwrap(), scalar("depr_est_r")));
        assert!(near(bounds(&sr12)[0], r_ci("sr12_ci_r")[0]));
        assert!(near(bounds(&sr12)[1], r_ci("sr12_ci_r")[1]));
        assert!(near(lik.estimate.unwrap(), scalar("lik_est_root")));
        assert!(near(srt.estimate.unwrap(), scalar("srt_est_root")));
        assert!(near(sr12.estimate.unwrap(), scalar("sr12_est_root")));
    }

    /// Deterministic noisy samples with ties (values rounded to one decimal).
    fn noisy(n: usize, seed: u64, shift: f64) -> Vec<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (state >> 11) as f64 / (1u64 << 53) as f64;
                (((u - 0.5) * 6.0 + shift) * 10.0).round() / 10.0
            })
            .collect()
    }

    /// The selections equal the sorted naive vectors R forms at every k.
    #[test]
    fn test_kth_selection_matches_naive_sort() {
        for (n1, n2) in [(7, 7), (7, 40), (40, 301), (301, 7)] {
            let mut xs = noisy(n1, n1 as u64, 0.3);
            let mut ys = noisy(n2, 1000 + n2 as u64, 0.0);
            xs.sort_by(f64::total_cmp);
            ys.sort_by(f64::total_cmp);
            let mut diffs: Vec<f64> = xs
                .iter()
                .flat_map(|a| ys.iter().map(move |b| a - b))
                .collect();
            diffs.sort_by(f64::total_cmp);
            let step = (diffs.len() / 97).max(1);
            for k in (1..=diffs.len()).step_by(step).chain([diffs.len()]) {
                assert_eq!(
                    kth_pairwise_difference(&xs, &ys, k),
                    diffs[k - 1],
                    "{n1}x{n2} k={k}"
                );
            }
            let mut walsh = Vec::new();
            for i in 0..n1 {
                for j in i..n1 {
                    walsh.push((xs[i] + xs[j]) / 2.0);
                }
            }
            walsh.sort_by(f64::total_cmp);
            let step = (walsh.len() / 97).max(1);
            for k in (1..=walsh.len()).step_by(step).chain([walsh.len()]) {
                assert_eq!(kth_walsh_average(&xs, k), walsh[k - 1], "{n1} k={k}");
            }
        }
    }

    /// The estimate is the midpoint of the flat interval when the two middle order
    /// statistics differ (R returns some point of it).
    #[test]
    fn test_wilcoxon_estimate_midpoint_of_flat_interval() {
        let x = [1.0, 1.0, 4.0, 4.0];
        let y = [2.0, 2.0, 3.0, 3.0];
        let c = WilcoxonConfig {
            conf_int: true,
            ..Default::default()
        };
        let r = wilcoxon_rank_sum(&x, &y, 0.0, Alternative::TwoSided, &c).unwrap();
        let mut d: Vec<f64> = x
            .iter()
            .flat_map(|a| y.iter().map(move |b| a - b))
            .collect();
        d.sort_by(f64::total_cmp);
        assert_ne!(d[7], d[8]);
        assert_eq!(r.estimate, Some((d[7] + d[8]) / 2.0));
    }

    /// Tiny exact samples cannot reach a 99% level: R's warning and the achieved level.
    #[test]
    fn test_wilcoxon_exact_level_not_achievable() {
        let c = WilcoxonConfig {
            conf_int: true,
            conf_level: 0.99,
            ..Default::default()
        };
        let r = wilcoxon_rank_sum(
            &[1.3, 2.9, 4.1],
            &[0.2, 0.8, 1.7],
            0.0,
            Alternative::TwoSided,
            &c,
        )
        .unwrap();
        assert!(r.exact);
        assert_eq!(
            r.warning.as_deref(),
            Some("Requested conf.level not achievable")
        );
        // qwilcox(0.005, 3, 3) = 0 -> 1, achieved alpha = 2 P(W <= 0) = 0.1
        assert!((r.conf_level.unwrap() - 0.9).abs() < 1e-12);
    }

    /// R 4.3.3: wilcox.test(c(1.1, 2.3, 2.3, 3.4, 4.2), conf.int = TRUE, conf.level = 0.99)
    /// reports conf.level 0.84 with "requested conf.level not achievable" and bounds
    /// 1.699956 and 3.799967 (uniroot; the jump points are 1.7 and 3.8).
    #[test]
    fn test_wilcoxon_signed_rank_asymptotic_level_doubling() {
        let c = WilcoxonConfig {
            conf_int: true,
            conf_level: 0.99,
            ..Default::default()
        };
        let r = wilcoxon_signed_rank(
            &[1.1, 2.3, 2.3, 3.4, 4.2],
            None,
            0.0,
            Alternative::TwoSided,
            &c,
        )
        .unwrap();
        assert!(!r.exact);
        assert!(
            (r.conf_level.unwrap() - 0.84).abs() < 1e-12,
            "{:?}",
            r.conf_level
        );
        assert!(
            r.warning
                .as_deref()
                .unwrap()
                .contains("requested conf.level not achievable")
        );
        assert!((r.conf_int_lower.unwrap() - 1.7).abs() < 1e-12);
        assert!((r.conf_int_upper.unwrap() - 3.8).abs() < 1e-12);
    }

    /// All observations tied: R warns and then fails in uniroot; p2a returns no interval with
    /// the warning.
    #[test]
    fn test_wilcoxon_interval_all_tied() {
        let c = WilcoxonConfig {
            conf_int: true,
            ..Default::default()
        };
        let r = wilcoxon_rank_sum(
            &[2.0, 2.0, 2.0],
            &[1.0, 1.0],
            0.0,
            Alternative::TwoSided,
            &c,
        )
        .unwrap();
        assert_eq!(
            (r.conf_int_lower, r.conf_int_upper, r.estimate),
            (None, None, None)
        );
        assert!(
            r.warning
                .as_deref()
                .unwrap()
                .contains("cannot compute confidence interval when all observations are tied")
        );
        let s =
            wilcoxon_signed_rank(&[3.0, 3.0, 3.0], None, 0.0, Alternative::TwoSided, &c).unwrap();
        assert_eq!((s.conf_int_lower, s.conf_int_upper), (None, None));
        assert!(
            s.warning
                .as_deref()
                .unwrap()
                .contains("all observations are zero or tied")
        );
    }
}
