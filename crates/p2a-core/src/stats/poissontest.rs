//! Exact Poisson Test for Rate Parameters.
//!
//! Provides exact tests for:
//! - One-sample: Testing a simple null about the rate parameter
//! - Two-sample: Comparing the ratio of two rate parameters
//!
//! # Mathematical Background
//!
//! For one-sample tests, given X ~ Poisson(λT):
//! - H₀: λ = r (rate equals hypothesized value)
//! - Test statistic is X itself
//! - P-value computed exactly from Poisson distribution
//!
//! For two-sample tests, given X₁ ~ Poisson(λ₁T₁) and X₂ ~ Poisson(λ₂T₂):
//! - H₀: λ₁/λ₂ = r (rate ratio equals hypothesized value)
//! - Conditioned on X₁ + X₂ = n, X₁ ~ Binomial(n, p) where
//!   p = λ₁T₁/(λ₁T₁ + λ₂T₂) = rT₁/(rT₁ + T₂) under H₀
//!
//! # References
//!
//! - Przyborowski, J. & Wilenski, H. (1940). "Homogeneity of Results in Testing
//!   Samples from Poisson Series". Biometrika, 31(3/4), 313-323.
//! - R Core Team. `stats::poisson.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/poisson.test.html>

use crate::errors::{EconError, EconResult};
use serde::{Deserialize, Serialize};
use statrs::distribution::{Binomial, ContinuousCDF, DiscreteCDF, Poisson};

/// Alternative hypothesis for the Poisson test.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PoissonAlternative {
    /// Two-sided test (rate ≠ hypothesized value)
    TwoSided,
    /// One-sided test (rate > hypothesized value)
    Greater,
    /// One-sided test (rate < hypothesized value)
    Less,
}

/// Result of the Poisson test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonTestResult {
    /// Test method name
    pub method: String,
    /// Test statistic (observed count)
    pub statistic: f64,
    /// Expected count under null hypothesis
    pub parameter: f64,
    /// P-value
    pub p_value: f64,
    /// Confidence interval for rate or rate ratio
    pub conf_int: (f64, f64),
    /// Confidence level used
    pub conf_level: f64,
    /// Estimated rate or rate ratio
    pub estimate: f64,
    /// Null hypothesis value
    pub null_value: f64,
    /// Alternative hypothesis
    pub alternative: PoissonAlternative,
    /// Number of samples (1 or 2)
    pub n_samples: usize,
}

impl std::fmt::Display for PoissonTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.method)?;
        writeln!(f, "{}", "=".repeat(self.method.len()))?;
        writeln!(f)?;
        writeln!(
            f,
            "Statistic: {:.0}, Expected: {:.4}",
            self.statistic, self.parameter
        )?;
        writeln!(f, "p-value: {:.6}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "Estimate: {:.4}", self.estimate)?;
        writeln!(
            f,
            "{:.0}% CI: ({:.4}, {:.4})",
            self.conf_level * 100.0,
            self.conf_int.0,
            self.conf_int.1
        )?;
        writeln!(f)?;
        writeln!(f, "H₀: rate = {:.4}", self.null_value)?;
        Ok(())
    }
}

/// Largest event count the one-sample test accepts: 2^52 = 4_503_599_627_370_496.
///
/// statrs evaluates the Poisson CDF and the chi-squared quantiles of the exact interval with an
/// incomplete gamma series whose index starts at the count and grows by 1.0 in `f64`. From about
/// 2^53 on, adding 1.0 no longer changes the index and the series never finishes, a loop no
/// deadline can interrupt. 2^52 leaves the index room to grow exactly; a test at the bound itself
/// still takes seconds. The bound guarantees only that the call returns: the interval can be NaN
/// from about 2e6 and the p-value drifts from about 2^40 (STAT-60 / T3-WP40). The two-sample test
/// does not use that series: its p-value and interval use the binomial CDF, whose continued
/// fraction statrs stops after 140 terms, and every loop of its interval search has a fixed
/// iteration cap. Its counts only need a total that fits in `u64`; that the test returns says
/// nothing about its accuracy at large totals.
pub const POISSON_MAX_ONE_SAMPLE_COUNT: u64 = 1 << 52;

/// Refuse a one-sample count above [`POISSON_MAX_ONE_SAMPLE_COUNT`] before statrs sees it.
fn check_one_sample_count(x: u64) -> EconResult<()> {
    if x > POISSON_MAX_ONE_SAMPLE_COUNT {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x[0] = {x} is above the largest event count the one-sample exact Poisson test \
                 supports ({POISSON_MAX_ONE_SAMPLE_COUNT} = 2^52); beyond it the incomplete \
                 gamma series behind the Poisson CDF and the exact interval cannot advance in f64 \
                 and does not finish"
            ),
        });
    }
    Ok(())
}

/// The two-sample test conditions on x[0] + x[1]; refuse a total that does not fit in `u64`.
fn total_count(x1: u64, x2: u64) -> EconResult<u64> {
    x1.checked_add(x2)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "x = [{x1}, {x2}]: the total x[0] + x[1] exceeds the largest count a 64-bit \
                 integer holds ({}); the two-sample test conditions on this total",
                u64::MAX
            ),
        })
}

/// The two-sample null's p = r·T₁ / (r·T₁ + T₂) for finite, positive `r`, `t1` and `t2`; it lies
/// in [0, 1] for every such input, so the null's binomial can always be built.
///
/// The direct form is used when r·T₁ is a normal number and the denominator is finite, which
/// covers every r and T that are not extreme. Otherwise (r·T₁ is subnormal, which leaves the
/// direct form's numerator with fewer digits, or underflows to 0, or overflows, or r·T₁ + T₂
/// overflows) it is 1 / (1 + T₂ / (r·T₁)) with the ratio formed from logarithms, accurate to about
/// 1e-13 relative. Its exponential overflows only for a p below 1 / `f64::MAX`, which then comes
/// out as 0, and a p within 2^-53 of 1 comes out as 1.
fn null_success_probability(r: f64, t1: f64, t2: f64) -> f64 {
    let r_t1 = r * t1;
    let denominator = r_t1 + t2;
    if r_t1.is_normal() && denominator.is_finite() {
        r_t1 / denominator
    } else {
        1.0 / (1.0 + (t2.ln() - t1.ln() - r.ln()).exp())
    }
}

/// The two-sample estimate, the ratio of the rates (x[0] / T[0]) / (x[1] / T[1]), or infinity
/// when x[1] = 0. When both rates are finite it is their quotient. When one of them overflows f64
/// on its own it is formed as (x[0] / x[1]) · (T[1] / T[0]) by [`rate_ratio_from_odds`], so only
/// a ratio that itself overflows is refused.
fn two_sample_estimate(x1: u64, x2: u64, t1: f64, t2: f64) -> EconResult<f64> {
    if x2 == 0 {
        return Ok(f64::INFINITY);
    }
    let (rate1, rate2) = (x1 as f64 / t1, x2 as f64 / t2);
    let ratio = if rate1.is_finite() && rate2.is_finite() {
        rate1 / rate2
    } else {
        rate_ratio_from_odds(x1 as f64 / x2 as f64, t1, t2)
    };
    if !ratio.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x = [{x1}, {x2}], T = [{t1:?}, {t2:?}]: the rate ratio (x[0]/T[0]) / \
                 (x[1]/T[1]) = ({x1}/{t1:?}) / ({x2}/{t2:?}) overflows f64"
            ),
        });
    }
    Ok(ratio)
}

/// Perform exact Poisson test.
///
/// # Arguments
///
/// * `x` - Number of events (single value for one-sample, two values for comparison)
/// * `t` - Time base (single value for one-sample, two values for comparison)
/// * `r` - Hypothesized rate (one-sample) or rate ratio (two-sample). Default: 1.0
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for the interval (default: 0.95)
///
/// # One-Sample Test
///
/// Tests H₀: λ = r where X ~ Poisson(λT).
///
/// # Two-Sample Test
///
/// Tests H₀: λ₁/λ₂ = r where X₁ ~ Poisson(λ₁T₁), X₂ ~ Poisson(λ₂T₂).
///
/// # Errors
///
/// `EconError::InvalidSpecification` when:
/// - `x` does not have length 1 or 2, or `t` has a different length;
/// - a time base in `t`, or `r`, is not finite and > 0;
/// - `conf_level` is not strictly between 0 and 1;
/// - one-sample: the count is above [`POISSON_MAX_ONE_SAMPLE_COUNT`], or the expected count
///   `r * T` is not finite and > 0 (it overflows, or underflows to 0);
/// - two-sample: the total `x[0] + x[1]` does not fit in `u64`, or, when `x[1]` is not 0, the
///   estimated rate ratio `(x[0] / T[0]) / (x[1] / T[1])` overflows `f64`. A rate `x[i] / T[i]`
///   that overflows on its own is not refused when the ratio is representable, and neither is an
///   `r * T[0]` outside `f64`: the null's p is then formed from logarithms.
///
/// `EconError::Computation` when a chi-squared distribution for the one-sample interval cannot be
/// built.
///
/// # Example
///
/// ```
/// use p2a_core::stats::poissontest::{poisson_test, PoissonAlternative};
///
/// // One-sample test: 137 events in 24.2 time units
/// let result = poisson_test(&[137], &[24.19893], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();
/// println!("Rate estimate: {:.4}, p = {:.4}", result.estimate, result.p_value);
///
/// // Two-sample test: compare rates
/// let result2 = poisson_test(&[11, 21], &[800.0, 3011.0], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();
/// println!("Rate ratio: {:.4}", result2.estimate);
/// ```
pub fn poisson_test(
    x: &[u64],
    t: &[f64],
    r: f64,
    alternative: PoissonAlternative,
    conf_level: f64,
) -> EconResult<PoissonTestResult> {
    // Validate inputs
    if x.is_empty() || x.len() > 2 {
        return Err(EconError::InvalidSpecification {
            message: "x must have length 1 or 2".to_string(),
        });
    }

    if t.len() != x.len() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x and T must have the same length, got {} and {}",
                x.len(),
                t.len()
            ),
        });
    }

    // Each check is written so that NaN fails it. A NaN conf_level and an infinite T or r used to
    // reach statrs and panic there (Gamma::inverse_cdf, the incomplete gamma function).
    if let Some(&ti) = t.iter().find(|&&ti| !(ti.is_finite() && ti > 0.0)) {
        return Err(EconError::InvalidSpecification {
            message: format!("time base T must be finite and > 0, got {ti:?}"),
        });
    }

    if !(r.is_finite() && r > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("rate (ratio) r must be finite and > 0, got {r:?}"),
        });
    }

    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("conf_level must be strictly between 0 and 1, got {conf_level:?}"),
        });
    }

    if x.len() == 1 {
        check_one_sample_count(x[0])?;
        poisson_test_one_sample(x[0], t[0], r, alternative, conf_level)
    } else {
        poisson_test_two_sample(x[0], x[1], t[0], t[1], r, alternative, conf_level)
    }
}

/// One-sample exact Poisson test.
fn poisson_test_one_sample(
    x: u64,
    t: f64,
    r: f64,
    alternative: PoissonAlternative,
    conf_level: f64,
) -> EconResult<PoissonTestResult> {
    // Expected count under H₀
    let expected = r * t;
    if !(expected.is_finite() && expected > 0.0) {
        // statrs accepts an infinite Poisson mean and then panics in its CDF; a mean that
        // underflows to 0 it refuses with a message that names neither r nor T.
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the expected count r * T = {r:?} * {t:?} = {expected:?} must be finite and > 0"
            ),
        });
    }

    // Create Poisson distribution under H₀
    let pois = Poisson::new(expected).map_err(|_| EconError::InvalidSpecification {
        message: format!("Invalid Poisson parameter: {}", expected),
    })?;

    // P-value calculation
    let p_value = match alternative {
        PoissonAlternative::Less => {
            // P(X <= x)
            pois.cdf(x)
        }
        PoissonAlternative::Greater => {
            // P(X >= x) = 1 - P(X < x) = 1 - P(X <= x-1)
            if x == 0 { 1.0 } else { 1.0 - pois.cdf(x - 1) }
        }
        PoissonAlternative::TwoSided => {
            // Two-sided: 2 * min(P(X <= x), P(X >= x))
            let p_left = pois.cdf(x);
            let p_right = if x == 0 { 1.0 } else { 1.0 - pois.cdf(x - 1) };
            2.0 * p_left.min(p_right).min(0.5)
        }
    };

    // Rate estimate
    let estimate = x as f64 / t;

    // Confidence interval for the rate
    let (ci_lower, ci_upper) = poisson_ci(x, conf_level)?;
    let conf_int = (ci_lower / t, ci_upper / t);

    Ok(PoissonTestResult {
        method: "Exact Poisson test".to_string(),
        statistic: x as f64,
        parameter: expected,
        p_value,
        conf_int,
        conf_level,
        estimate,
        null_value: r,
        alternative,
        n_samples: 1,
    })
}

/// Two-sample exact Poisson test (comparison of rates).
fn poisson_test_two_sample(
    x1: u64,
    x2: u64,
    t1: f64,
    t2: f64,
    r: f64,
    alternative: PoissonAlternative,
    conf_level: f64,
) -> EconResult<PoissonTestResult> {
    let n = total_count(x1, x2)?;

    // Under H₀: λ₁/λ₂ = r
    // X₁ | (X₁ + X₂ = n) ~ Binomial(n, p)
    // where p = λ₁T₁ / (λ₁T₁ + λ₂T₂) = rT₁ / (rT₁ + T₂), in [0, 1] for any finite, positive r, T
    let p = null_success_probability(r, t1, t2);
    let estimate = two_sample_estimate(x1, x2, t1, t2)?;

    // Expected value of X₁ under H₀
    let expected = n as f64 * p;

    // P-value from binomial distribution
    let p_value = if n == 0 {
        1.0 // No events, no evidence
    } else {
        let binom = Binomial::new(p, n).map_err(|_| EconError::InvalidSpecification {
            message: format!("Invalid Binomial parameters: p={}, n={}", p, n),
        })?;

        match alternative {
            PoissonAlternative::Less => {
                // P(X₁ <= x₁)
                binom.cdf(x1)
            }
            PoissonAlternative::Greater => {
                // P(X₁ >= x₁) = 1 - P(X₁ <= x₁ - 1)
                if x1 == 0 {
                    1.0
                } else {
                    1.0 - binom.cdf(x1 - 1)
                }
            }
            PoissonAlternative::TwoSided => {
                let p_left = binom.cdf(x1);
                let p_right = if x1 == 0 {
                    1.0
                } else {
                    1.0 - binom.cdf(x1 - 1)
                };
                2.0 * p_left.min(p_right).min(0.5)
            }
        }
    };

    // Exact conditional confidence interval for the rate ratio
    let conf_int = rate_ratio_ci(x1, x2, n, t1, t2, conf_level);

    Ok(PoissonTestResult {
        method: "Comparison of Poisson rates".to_string(),
        statistic: x1 as f64,
        parameter: expected,
        p_value,
        conf_int,
        conf_level,
        estimate,
        null_value: r,
        alternative,
        n_samples: 2,
    })
}

/// Compute confidence interval for Poisson count using exact method.
///
/// Uses the relationship between Poisson and chi-squared/gamma distributions.
/// `poisson_test` keeps `x` at or below [`POISSON_MAX_ONE_SAMPLE_COUNT`]; `x + 1` is still checked
/// before any distribution is built.
fn poisson_ci(x: u64, conf_level: f64) -> EconResult<(f64, f64)> {
    use statrs::distribution::ChiSquared;

    let x_plus_1 = x
        .checked_add(1)
        .ok_or_else(|| EconError::InvalidSpecification {
            message: format!(
                "x = {x}: x + 1 (the upper bound's shape) exceeds the largest 64-bit count ({})",
                u64::MAX
            ),
        })?;
    let chi_squared = |df: f64| {
        ChiSquared::new(df).map_err(|e| {
            EconError::Computation(format!(
                "chi-squared distribution with {df} degrees of freedom: {e}"
            ))
        })
    };

    let alpha = 1.0 - conf_level;

    // Lower bound: chi-squared quantile
    let lower = if x == 0 {
        0.0
    } else {
        chi_squared(2.0 * x as f64)?.inverse_cdf(alpha / 2.0) / 2.0
    };

    // Upper bound: chi-squared quantile
    let upper = chi_squared(2.0 * x_plus_1 as f64)?.inverse_cdf(1.0 - alpha / 2.0) / 2.0;

    Ok((lower, upper))
}

/// Smallest odds [`find_odds_bound`] resolves. statrs' binomial CDF evaluates its incomplete beta
/// function at 1 − p and treats an argument within a few ulps of 1 as 1, so below p ≈ 4.4e-16
/// the tail it returns is 0 or 1 whatever p is. Above that cutoff, 1 − p rounds p to a multiple
/// of 2^-53 ≈ 1.1e-16, so near ODDS_MIN a bound snaps to a boundary (k + 1/2) · 2^-53 between
/// them and is resolved only to about 2^-54 / s relative: 5.6% at 1e-15, 2.2e-4 at 2.5e-13.
const ODDS_MIN: f64 = 1e-15;

/// Largest odds [`find_odds_bound`] resolves. Near p = 1 the 1 − p that statrs works from has an
/// absolute error up to 2^-54, a relative error of about 5.6e-17 · (1 + s) at odds s: 6e-7 at
/// s = 1e10 and 6% at 1e15. From s = 2^53 ≈ 9e15 on, 1 + s rounds to s and p is exactly 1.
const ODDS_MAX: f64 = 1e15;

/// Cap on each bracketing loop in [`find_odds_bound`]. The search starts from the estimate
/// x[0] / max(x[1], 1), which is 0 or lies in [2^-64, 2^64]. Its longest walk, the upper bound's
/// halving from 2^64 down to [`ODDS_MIN`], takes at most 114 steps; its doubling from 100 up to
/// [`ODDS_MAX`] takes at most 44, and the lower bound's two loops at most 6 and 4. The cap leaves
/// every search as it is and only guarantees that each loop ends.
const MAX_BRACKET_STEPS: usize = 128;

/// Compute confidence interval for rate ratio.
///
/// Uses the exact conditional method: given n = x1 + x2 (checked by the caller), X₁ is binomial
/// with odds p / (1 - p) = r·T₁ / T₂. The bounds are searched on those odds, which depend on the
/// counts and `conf_level` only, and then multiplied by T₂ / T₁, as R's `poisson.test` does.
///
/// When both counts are > 0, exact tails put the lower odds bound strictly below the count ratio
/// x[0] / x[1] and the upper strictly above it: at p = x[0] / n the binomial's mean x[0] is also
/// its median, so both tails at x[0] are at least 1/2, more than alpha / 2. A bound from
/// [`find_odds_bound`] at or beyond the count ratio on the wrong side therefore means that statrs'
/// CDF misjudged the tail (near x[0] = x[1] from totals of about 4e7, for any counts from about
/// 3e14), and it is widened to 0 (lower) or infinity (upper). So, for every input, the odds
/// interval contains x[0] / x[1] and is never inverted, and the interval returned, those odds
/// times T₂ / T₁, contains the estimate. It need not contain the true interval: x = [2e7, 2e7]
/// gives (0, 1.00062) for R's (0.99938, 1.00062), and x = [5e15, 1] gives (1e15, ∞) for R's
/// (8.97e14, 1.97e17).
fn rate_ratio_ci(x1: u64, x2: u64, n: u64, t1: f64, t2: f64, conf_level: f64) -> (f64, f64) {
    let alpha = 1.0 - conf_level;

    if n == 0 {
        return (0.0, f64::INFINITY);
    }

    // Use numerical search to find CI bounds
    // The p-value equals alpha/2 at the CI bounds

    // Lower bound: find r such that P(X₁ >= x₁ | H₀: ratio=r) = alpha/2
    let mut lower = if x1 == 0 {
        0.0
    } else {
        find_odds_bound(x1, x2, n, alpha / 2.0, true)
    };

    // Upper bound: find r such that P(X₁ <= x₁ | H₀: ratio=r) = alpha/2
    let mut upper = if x2 == 0 {
        f64::INFINITY
    } else {
        find_odds_bound(x1, x2, n, alpha / 2.0, false)
    };

    // A bound on the wrong side of the count ratio comes only from a misjudged tail (see above).
    if x1 > 0 && x2 > 0 {
        let odds_est = x1 as f64 / x2 as f64;
        if lower >= odds_est {
            lower = 0.0;
        }
        if upper <= odds_est {
            upper = f64::INFINITY;
        }
    }

    (
        rate_ratio_from_odds(lower, t1, t2),
        rate_ratio_from_odds(upper, t1, t2),
    )
}

/// The rate ratio r = s·T₂ / T₁ for odds `s` in [0, ∞]: a bound found by [`find_odds_bound`], or
/// the count ratio x[0] / x[1] when [`two_sample_estimate`] cannot divide the rates. The product
/// is formed directly when T₂ / T₁ is a normal number and otherwise from logarithms, so it
/// overflows or underflows only when the result itself lies outside f64; s = 0 gives 0 and
/// s = ∞ gives ∞ either way.
fn rate_ratio_from_odds(s: f64, t1: f64, t2: f64) -> f64 {
    let time_ratio = t2 / t1;
    if time_ratio.is_normal() {
        s * time_ratio
    } else {
        (s.ln() + t2.ln() - t1.ln()).exp()
    }
}

/// Find a bound on the odds s = p / (1 - p) of X₁ | n by bisection.
///
/// For the lower bound, we find s such that P(X₁ >= x₁ | s) = alpha
/// As s increases, P(X₁ >= x₁) increases, so we want to find the smallest s
/// where this probability just reaches alpha.
///
/// For the upper bound, we find s such that P(X₁ <= x₁ | s) = alpha
/// As s increases, P(X₁ <= x₁) decreases, so we want to find the largest s
/// where this probability just reaches alpha.
///
/// `n` is `x1 + x2`, checked by the caller. The odds do not depend on the time bases. Only the
/// start points come from the counts (the estimate x[0] / max(x[1], 1)); the bracketing loops
/// walk from them no further than [`ODDS_MIN`] and [`ODDS_MAX`], fixed odds whatever the counts
/// are, and the tail is evaluated at odds clamped to that range. So the search resolves bounds
/// in [1e-15, 1e15] only. At 95% a bound can lie beyond that range from totals of about 2.5e13
/// (x = [m, 1] or [1, m]; for x[1] = 0 or x[0] = 0, from about 3.7e15), and earlier at higher
/// levels (at 0.9999 the upper odds of x = [m, 1] pass 1e15 from m ≈ 5e10).
///
/// A bound beyond the range leaves a bracket whose ends do not straddle alpha, and a limit comes
/// back instead, without an error: 0 for a lower bound below 1e-15, [`ODDS_MIN`] for an upper
/// bound below it, [`ODDS_MAX`] for a lower bound above 1e15, and infinity for an upper bound
/// above it. The limit lies on the conservative side of the true bound, so it never inverts the
/// interval and the interval still contains the estimate, as long as statrs' tail at the limit
/// is on the right side of alpha. It is at totals up to about 1e13: x = [1, 1e10] at 0.999999
/// gives a lower bound of 0 for R's 5.0e-17, and x = [1e10, 1] an upper bound of infinity for
/// R's 2.0e16. At larger totals a bound within the CDF's error (below) of a limit can come back
/// near the limit on either side of the true bound, and from about 3e14 a limit can come back
/// for a bound well inside the range. The other bound keeps the search's own error and can lie
/// inside the true interval (x = [3e13, 1]: lower bound 1.35% above R's). A lower bound comes
/// back as x[0] / max(x[1], 1) instead of [`ODDS_MAX`] when that is smaller. With x[1] > 0 that
/// is the estimate, and it happens only when the CDF misjudges the tail, since an accurate one
/// puts the lower bound below the estimate ([`rate_ratio_ci`] then widens it to 0).
///
/// Within the range four things limit a bound:
/// - Resolution degrades as s nears either limit (see [`ODDS_MIN`] and [`ODDS_MAX`]): at 0.9999,
///   x = [5e10, 1] has an upper bound of 9.48e14, 5.2% below R's 9.99975e14.
/// - The bisection stops once the tail is within 1e-10 · alpha of alpha, or once the bracket is
///   narrower than 1e-10 · (hi + lo + 1e-10). The second rule is coarser than 1e-10 relative only
///   for odds below about 1e-10, where it is still finer than the resolution near [`ODDS_MIN`].
/// - The lower bound's tail is 1 − F(x[0] − 1), and statrs' CDF F is close to 1 there and keeps
///   only 2^-53 absolute, so that tail is good only to about 2^-54 / alpha relative. That matters
///   only at levels close to 1: at conf_level 1 − 2e-12, x = [5, 5] has a lower bound 6.5e-6
///   above R's, and an upper bound 2e-11 off.
/// - Otherwise a bound is only as accurate as statrs' binomial CDF, whose incomplete beta
///   function loses digits as n grows. At n = 1e11 the upper bound is off by about 3e-5 and the
///   lower by 4.7e-4 (x = [1, 1e11]), 3.2e-4 of that from the CDF and 1.6e-4 from the rounding of
///   1 − p. At totals of 3e13 to 1e14 bounds are up to about 2% off when x[0] or x[1] is 1
///   (x = [5e13, 1]: lower bound 1.85% above R's), and more when a count is 0: about 2.5% at
///   3e13, 8% at 5e13 and 7% at 1e14 (x = [0, 5e13]: upper bound 8% below R's).
///
/// Near x[0] = x[1] from totals of about 4e7, and for any counts from about 2e14 (a zero count)
/// or 3e14, statrs' binomial CDF is off by tens of percent or more, and the result is unreliable
/// in every direction (STAT-60 / T3-WP40):
/// - bounds inside the range are tens of percent off (x = [0, 2e14]: upper bound 25% below R's;
///   x = [1, 3e14]: upper bound 28% above; x = [1e15, 1]: lower bound 30% above);
/// - from about 3e14 the limit returns above fire for bounds inside the range, and the limit is
///   then not conservative (x = [5e15, 1]: lower bound 1e15, R 8.97e14; x = [0, 3e15]: upper
///   bound 1e-15, R 1.23e-15; x = [3e14, 0]: lower bound 3e14, which is x[0], R 8.1e13);
/// - a bound can come back on the wrong side of the estimate, which would invert the interval or
///   leave the estimate outside it (x = [1e8, 1e8]: (1.0000038, 0.9999962); x = [1, 5e15]:
///   (5.6e-15, 1e-15) around 2e-16; x = [1000, 1e15]: (9.25e-13, 9.54e-13) around 1e-12).
///   [`rate_ratio_ci`] widens such a bound to 0 or infinity.
///
/// Every loop has a fixed iteration cap, and the search returns 0, a finite s > 0 or infinity
/// for any input.
fn find_odds_bound(x1: u64, x2: u64, n: u64, alpha: f64, find_lower: bool) -> f64 {
    // Compute the tail probability for given odds s; p = s / (1 + s), that is the null's p at
    // T = [1, 1], from odds kept in the range the binomial CDF resolves.
    let compute_tail_p = |s: f64| -> f64 {
        let p = null_success_probability(s.clamp(ODDS_MIN, ODDS_MAX), 1.0, 1.0);
        if let Ok(binom) = Binomial::new(p, n) {
            if find_lower {
                // For lower bound: we want P(X₁ >= x₁ | H₀)
                // = 1 - P(X₁ <= x₁ - 1) = 1 - F(x₁ - 1)
                if x1 == 0 {
                    1.0
                } else {
                    1.0 - binom.cdf(x1 - 1)
                }
            } else {
                // For upper bound: we want P(X₁ <= x₁ | H₀) = F(x₁)
                binom.cdf(x1)
            }
        } else {
            // Fallback for invalid parameters
            if find_lower { 1.0 } else { 0.0 }
        }
    };

    // Initial estimate for the odds
    let odds_est = x1 as f64 / x2.max(1) as f64;

    // Set up initial bounds
    let (mut lo, mut hi) = if find_lower {
        // For lower CI: search between very small value and the point estimate
        (1e-10, odds_est.max(1e-5))
    } else {
        // For upper CI: search between point estimate and a large value
        (odds_est.max(1e-10), odds_est * 100.0 + 100.0)
    };

    // Expand bounds if needed to bracket the solution, each loop at most MAX_BRACKET_STEPS times.
    if find_lower {
        // For lower bound: need compute_tail_p(lo) < alpha < compute_tail_p(hi)
        for _ in 0..MAX_BRACKET_STEPS {
            if !(lo > ODDS_MIN && compute_tail_p(lo) > alpha) {
                break;
            }
            lo /= 10.0;
        }
        for _ in 0..MAX_BRACKET_STEPS {
            if !(hi < (odds_est * 10.0).min(ODDS_MAX) && compute_tail_p(hi) < alpha) {
                break;
            }
            hi *= 2.0;
        }
    } else {
        // For upper bound: need compute_tail_p(hi) < alpha < compute_tail_p(lo)
        for _ in 0..MAX_BRACKET_STEPS {
            if !(lo > ODDS_MIN && compute_tail_p(lo) < alpha) {
                break;
            }
            lo /= 2.0;
        }
        for _ in 0..MAX_BRACKET_STEPS {
            if !(hi < ODDS_MAX && compute_tail_p(hi) > alpha) {
                break;
            }
            hi *= 2.0;
        }
    }

    // Each loop above stops short of its limit only once its end is on the right side of alpha.
    // So the bracket fails to straddle alpha only when a loop ran to ODDS_MIN or ODDS_MAX, or
    // started beyond it, with the bound further out (the lower bound's hi also stops at ten times
    // the estimate, which a lower bound does not reach). The tail there is evaluated at the
    // clamped odds, and the bisection would converge on a bracket end. Return the limit on the
    // conservative side instead, so the limit side lies outside the true bound. With exact tails
    // a bound inside the range never gets here. Where statrs' tail is off (by percents from
    // totals of about 3e13, by tens of percent from about 2e14) a bound inside the range near a
    // limit can, and from about 3e14 one well inside it can too; the limit returned is then not
    // conservative.
    if find_lower {
        if compute_tail_p(lo) > alpha {
            return 0.0;
        }
        if compute_tail_p(hi) < alpha {
            return ODDS_MAX.min(odds_est);
        }
    } else {
        if compute_tail_p(hi) > alpha {
            return f64::INFINITY;
        }
        if compute_tail_p(lo) < alpha {
            return ODDS_MIN;
        }
    }

    // Bisection search
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let p = compute_tail_p(mid);

        if (p - alpha).abs() <= 1e-10 * alpha {
            return mid;
        }

        if find_lower {
            // For lower bound: P increases with s
            // If P(mid) > alpha, we need smaller s: hi = mid
            // If P(mid) < alpha, we need larger s: lo = mid
            if p > alpha {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            // For upper bound: P decreases with s
            // If P(mid) > alpha, we need larger s: lo = mid
            // If P(mid) < alpha, we need smaller s: hi = mid
            if p > alpha {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Convergence check
        if (hi - lo) / (hi + lo + 1e-10) < 1e-10 {
            break;
        }
    }

    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_one_sample_basic() {
        // 137 events in 24.2 time units
        let result =
            poisson_test(&[137], &[24.19893], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();

        println!("One-sample: x={}, t={:.4}", 137, 24.19893);
        println!("Rate estimate: {:.4}", result.estimate);
        println!("Expected under H0: {:.4}", result.parameter);
        println!("p-value: {:.6}", result.p_value);
        println!(
            "95% CI: ({:.4}, {:.4})",
            result.conf_int.0, result.conf_int.1
        );

        assert_eq!(result.n_samples, 1);
        assert!((result.estimate - 137.0 / 24.19893).abs() < 0.01);
        // Rate should be significantly different from 1.0
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_poisson_two_sample_basic() {
        // Two-sample comparison
        let result = poisson_test(
            &[11, 21],
            &[800.0, 3011.0],
            1.0,
            PoissonAlternative::TwoSided,
            0.95,
        )
        .unwrap();

        println!("Two-sample: x1=11, x2=21, t1=800, t2=3011");
        println!("Rate ratio estimate: {:.4}", result.estimate);
        println!("p-value: {:.6}", result.p_value);
        println!(
            "95% CI: ({:.4}, {:.4})",
            result.conf_int.0, result.conf_int.1
        );

        assert_eq!(result.n_samples, 2);
        assert_eq!(result.method, "Comparison of Poisson rates");
    }

    #[test]
    fn test_poisson_ci() {
        // Test CI computation for known values
        let (lower, upper) = poisson_ci(10, 0.95).unwrap();
        println!("Poisson CI for x=10: ({:.4}, {:.4})", lower, upper);

        // For x=10, 95% CI should be approximately (4.8, 18.4)
        assert!(lower > 4.0 && lower < 6.0);
        assert!(upper > 17.0 && upper < 20.0);
    }

    #[test]
    fn test_poisson_alternatives() {
        let x = 15u64;
        let t = 10.0;
        let r = 1.0; // H0: rate = 1.0

        let two_sided = poisson_test(&[x], &[t], r, PoissonAlternative::TwoSided, 0.95).unwrap();
        let greater = poisson_test(&[x], &[t], r, PoissonAlternative::Greater, 0.95).unwrap();
        let less = poisson_test(&[x], &[t], r, PoissonAlternative::Less, 0.95).unwrap();

        // Rate estimate = 1.5, which is greater than H0 rate = 1.0
        // So "greater" alternative should have smaller p-value than "less"
        println!(
            "Two-sided p: {:.4}, Greater p: {:.4}, Less p: {:.4}",
            two_sided.p_value, greater.p_value, less.p_value
        );

        assert!(greater.p_value < less.p_value);
    }

    #[test]
    fn test_poisson_edge_cases() {
        // Zero events
        let result = poisson_test(&[0], &[10.0], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();
        assert_eq!(result.statistic, 0.0);
        assert!(result.conf_int.0 == 0.0);

        // Large count
        let result =
            poisson_test(&[1000], &[100.0], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();
        assert!((result.estimate - 10.0).abs() < 0.1);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_poisson_against_r() {
        // R code:
        // poisson.test(137, 24.19893)
        //
        // Result:
        // data:  137 time base: 24.19893
        // number of events = 137, time base = 24.199, p-value < 2.2e-16
        // alternative hypothesis: true event rate is not equal to 1
        // 95 percent confidence interval:
        //  4.739093 6.665835
        // sample estimates:
        // event rate
        //   5.661765

        let result =
            poisson_test(&[137], &[24.19893], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();

        println!("R validation (one-sample):");
        println!("Rate estimate: {:.6}", result.estimate);
        println!("p-value: {:.10}", result.p_value);
        println!(
            "95% CI: ({:.6}, {:.6})",
            result.conf_int.0, result.conf_int.1
        );

        // R gives: rate = 5.661765
        assert!(
            (result.estimate - 5.661765).abs() < 0.001,
            "Rate estimate mismatch: got {}",
            result.estimate
        );

        // R gives: p-value < 2.2e-16
        assert!(
            result.p_value < 1e-10,
            "p-value should be very small, got {}",
            result.p_value
        );

        // R gives: 95% CI (4.739093, 6.665835)
        // Note: Small differences in chi-squared quantile computation lead to ~0.02 differences
        assert!(
            (result.conf_int.0 - 4.739093).abs() < 0.02,
            "CI lower mismatch: got {}",
            result.conf_int.0
        );
        assert!(
            (result.conf_int.1 - 6.665835).abs() < 0.02,
            "CI upper mismatch: got {}",
            result.conf_int.1
        );
    }

    #[test]
    fn test_validate_poisson_two_sample_against_r() {
        // R code:
        // poisson.test(c(11, 21), c(800, 3011))
        //
        // Two-sample rate comparison test
        // Under H₀: λ₁/λ₂ = 1, we condition on X₁+X₂ and use binomial distribution
        // where p = T₁/(T₁+T₂) = 800/3811 ≈ 0.2099
        //
        // Our implementation follows the exact conditional method:
        // - Rate ratio estimate = (11/800) / (21/3011) ≈ 1.971
        // - Expected X₁ under H₀ = n × p = 32 × 0.2099 ≈ 6.72
        // - P-value computed from Binomial(32, 0.2099)

        let result = poisson_test(
            &[11, 21],
            &[800.0, 3011.0],
            1.0,
            PoissonAlternative::TwoSided,
            0.95,
        )
        .unwrap();

        println!("R validation (two-sample):");
        println!("Rate ratio: {:.6}", result.estimate);
        println!("Expected count1: {:.4}", result.parameter);
        println!("p-value: {:.4}", result.p_value);
        println!(
            "95% CI: ({:.6}, {:.6})",
            result.conf_int.0, result.conf_int.1
        );

        // Rate ratio = (11/800) / (21/3011) = 0.01375 / 0.00697 ≈ 1.9715
        assert!(
            (result.estimate - 1.9715).abs() < 0.01,
            "Rate ratio mismatch: got {}",
            result.estimate
        );

        // Expected count = 32 * (800/3811) ≈ 6.717
        assert!(
            (result.parameter - 6.717).abs() < 0.1,
            "Expected count mismatch: got {}",
            result.parameter
        );

        // Two-sample test: significant evidence that rates differ
        // CI should not include 1.0 if p < 0.05, or include 1.0 if p > 0.05
        assert!(
            result.p_value > 0.05,
            "p-value indicates non-significant result, got {}",
            result.p_value
        );

        // CI should include 1.0 since p > 0.05
        assert!(
            result.conf_int.0 < 1.0 && result.conf_int.1 > 1.0,
            "CI should include 1.0: ({:.4}, {:.4})",
            result.conf_int.0,
            result.conf_int.1
        );

        // R: 95 percent confidence interval 0.858426403391639 4.27726594365407
        assert_relative(result.conf_int.0, 0.858426403391639, 1e-7, "CI lower vs R");
        assert_relative(result.conf_int.1, 4.27726594365407, 1e-7, "CI upper vs R");
    }

    fn refusal(x: &[u64], t: &[f64], r: f64, conf_level: f64) -> String {
        match poisson_test(x, t, r, PoissonAlternative::TwoSided, conf_level) {
            Err(EconError::InvalidSpecification { message }) => message,
            other => panic!("x = {x:?}, t = {t:?}, r = {r}, conf_level = {conf_level}: {other:?}"),
        }
    }

    /// FU-1: the two-sample test conditions on x[0] + x[1]. A total that does not fit in u64 used
    /// to panic in debug builds ("attempt to add with overflow") and wrap in release builds.
    #[test]
    fn test_poisson_two_sample_refuses_overflowing_total() {
        for x in [
            [u64::MAX, 1],
            [1, u64::MAX],
            [1 << 63, 1 << 63],
            [u64::MAX, u64::MAX],
        ] {
            let message = refusal(&x, &[1.0, 2.5], 1.0, 0.95);
            assert!(message.contains("x[0] + x[1]"), "{message}");
            assert!(message.contains(&x[0].to_string()), "{message}");
            assert!(message.contains(&u64::MAX.to_string()), "{message}");
        }
    }

    /// The total may be u64::MAX itself and the call returns: statrs stops the binomial CDF's
    /// continued fraction after 140 terms, and the interval search's bisection (200 steps) and
    /// bracketing loops (MAX_BRACKET_STEPS each) have fixed caps. That says nothing about the
    /// accuracy of the result at such totals.
    #[test]
    fn test_poisson_two_sample_accepts_largest_total() {
        for x in [[u64::MAX - 1, 1], [u64::MAX, 0], [0, u64::MAX]] {
            let result = poisson_test(&x, &[1.0, 2.5], 1.0, PoissonAlternative::TwoSided, 0.95);
            assert!(result.is_ok(), "x = {x:?}: {result:?}");
        }
    }

    /// FU-1: a one-sample count above 2^52 is refused before statrs is called. From about 2^53 on,
    /// the incomplete gamma series behind the Poisson CDF and the chi-squared quantiles can stop
    /// advancing and never return; whether a given count hangs depends on r * T. At T = 3.5 (r = 1),
    /// 2^53, 2^53 + 2 and u64::MAX - 1 hung, while 2^53 - 2 returned at once with a NaN interval;
    /// 2^53 - 2 hung when r * T = x. The counts below are listed to check the refusal, not because
    /// each one hangs at the T used here; u64::MAX would also overflow x + 1 in the interval.
    #[test]
    fn test_poisson_one_sample_refuses_counts_above_bound() {
        for x in [
            POISSON_MAX_ONE_SAMPLE_COUNT + 1,
            (1 << 53) - 2,
            1 << 53,
            (1 << 53) + 2,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let message = refusal(&[x], &[3.5], 1.0, 0.95);
            assert!(message.contains(&format!("x[0] = {x}")), "{message}");
            assert!(
                message.contains(&POISSON_MAX_ONE_SAMPLE_COUNT.to_string()),
                "{message}"
            );
        }
    }

    /// The bound is inclusive. Checked on the validation step: a full test at 2^52 takes seconds.
    #[test]
    fn test_poisson_one_sample_bound_is_inclusive() {
        // Re-exported next to poisson_test, so callers (PR-01m's MCP cap) can name it.
        assert_eq!(
            crate::stats::POISSON_MAX_ONE_SAMPLE_COUNT,
            4_503_599_627_370_496
        );
        assert!(check_one_sample_count(POISSON_MAX_ONE_SAMPLE_COUNT).is_ok());
        assert!(check_one_sample_count(0).is_ok());
        assert!(check_one_sample_count(POISSON_MAX_ONE_SAMPLE_COUNT + 1).is_err());
    }

    /// FU-1: poisson_ci checks x + 1 before any distribution is built and no longer unwraps
    /// ChiSquared::new.
    #[test]
    fn test_poisson_ci_refuses_overflowing_count() {
        let e = poisson_ci(u64::MAX, 0.95).unwrap_err().to_string();
        assert!(e.contains("x + 1"), "{e}");
        // The limit, not only the parameter (which is u64::MAX itself here).
        assert!(e.contains(&format!("({})", u64::MAX)), "{e}");
        let (lower, upper) = poisson_ci(0, 0.95).unwrap();
        assert_eq!(lower, 0.0);
        assert!(upper > 3.68 && upper < 3.69, "{upper}"); // qgamma(0.975, 1) = 3.688879
    }

    /// Non-finite or out-of-range T, r and conf_level used to reach statrs and panic there
    /// (conf_level NaN in Gamma::inverse_cdf; T or r infinite, or r * T overflowing, in the
    /// incomplete gamma function). The wording matches T3-WP40's planned validation.
    #[test]
    fn test_poisson_refuses_non_finite_parameters() {
        for level in [f64::NAN, 95.0, 1.0, 0.0] {
            let message = refusal(&[10], &[5.0], 1.0, level);
            assert!(message.contains("conf_level"), "{message}");
        }
        for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0] {
            for (x, ts) in [(vec![10], vec![t]), (vec![11, 21], vec![800.0, t])] {
                let message = refusal(&x, &ts, 1.0, 0.95);
                assert!(
                    message.contains("time base T must be finite and > 0"),
                    "{message}"
                );
            }
        }
        for r in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            for (x, ts) in [(vec![10], vec![5.0]), (vec![11, 21], vec![800.0, 3011.0])] {
                let message = refusal(&x, &ts, r, 0.95);
                assert!(message.contains("r must be finite and > 0"), "{message}");
            }
        }
        // Debug formatting: Display wrote these as 301- and 21-digit numbers.
        let message = refusal(&[10], &[5.0], -1e300, 0.95);
        assert!(message.contains("got -1e300"), "{message}");
        let message = refusal(&[10], &[-1e300], 1.0, 0.95);
        assert!(message.contains("got -1e300"), "{message}");
        let message = refusal(&[10], &[5.0], 1.0, 1e20);
        assert!(message.contains("got 1e20"), "{message}");
        let message = refusal(&[10], &[1e200], 1e200, 0.95);
        assert!(message.contains("r * T"), "{message}");
        // One-sample: an expected count that underflows to 0 used to surface as statrs' "Invalid
        // Poisson parameter: 0", which names neither r nor T.
        let message = refusal(&[10], &[1e-200], 1e-200, 0.95);
        assert!(message.contains("r * T = 1e-200 * 1e-200"), "{message}");
    }

    /// PR-24: the two-sample null's p = r·T[0] / (r·T[0] + T[1]). An r·T[0] that overflowed
    /// made p NaN (an opaque "Invalid Binomial parameters" error, or Ok with a NaN expected count
    /// at x = [0, 0]); after that was fixed by refusing an r·T[0] outside f64, inputs whose p is
    /// representable were refused. p is formed from logarithms there now, and also when r·T[0] is
    /// subnormal, where the direct form's numerator kept fewer digits.
    #[test]
    fn test_poisson_two_sample_null_p_from_extreme_r_times_t() {
        let run = |x: [u64; 2], t: [f64; 2], r: f64| {
            poisson_test(&x, &t, r, PoissonAlternative::TwoSided, 0.95)
                .unwrap_or_else(|e| panic!("x = {x:?}, T = {t:?}, r = {r:?}: {e}"))
        };
        // r·T[0] underflows to 0: the same null as T = [1, 1], p = 1e-200. The logarithms round
        // differently from the direct form, so the expected count agrees to 1e-12, not bit for bit.
        let (tiny, unit) = (
            run([11, 21], [1e-200, 1e-200], 1e-200),
            run([11, 21], [1.0, 1.0], 1e-200),
        );
        assert_relative(unit.parameter, 32e-200, 1e-15, "T = [1, 1] expected count");
        assert_relative(tiny.parameter, unit.parameter, 1e-12, "expected count");
        assert_eq!(tiny.p_value, unit.p_value);
        assert_eq!(tiny.conf_int, unit.conf_int);
        assert_relative(tiny.estimate, unit.estimate, 1e-15, "estimate");
        // r·T[0] = 1e-400 underflows and T[1] = 1e-300: p = 1e-100.
        let result = run([11, 21], [1e-200, 1e-300], 1e-200);
        assert_relative(result.parameter, 32e-100, 1e-12, "T = [1e-200, 1e-300]");
        // r·T[0] = 1e400 overflows: p = 1 - 1e-400 rounds to 1, so the expected count is n and
        // P(X[0] <= 11) is 0; at x = [0, 0] the expected count is 0, not NaN.
        let result = run([11, 21], [1e200, 1.0], 1e200);
        assert_eq!((result.parameter, result.p_value), (32.0, 0.0));
        let result = run([0, 0], [1e200, 1.0], 1e200);
        assert_eq!((result.parameter, result.p_value), (0.0, 1.0));
        // r·T[0] = 1e-320 is subnormal; the direct form gave p = 1e-20 to only about 1e-5.
        let result = run([11, 21], [1e-160, 1e-300], 1e-160);
        assert_relative(result.parameter, 32e-20, 1e-12, "subnormal r * T[0]");
    }

    /// PR-24: the rate ratio's interval search started from the point estimate. When
    /// (x[0]/T[0]) / (x[1]/T[1]) overflowed f64, the upper bound's bracketing started at
    /// infinity, where halving does not move, and never returned; every T here is finite and > 0,
    /// so the T check let it through. An estimate that overflows is refused before any statrs
    /// call now; a rate that overflows on its own is not (see the test below).
    #[test]
    fn test_poisson_two_sample_refuses_overflowing_rate_ratio() {
        for (x, t) in [
            ([1, 1], [1e-160, 1e160]),
            ([11, 21], [1e-300, 1e300]),
            // The rate x[0]/T[0] = 1e309 overflows, and so does the ratio, 1e309.
            ([10_000_000_000_000_000_000, 1], [1e-290, 1.0]),
        ] {
            let message = refusal(&x, &t, 1.0, 0.95);
            assert!(message.contains("the rate ratio"), "{message}");
            assert!(message.contains("overflows f64"), "{message}");
            assert!(message.contains(&format!("x = {x:?}")), "{message}");
            assert!(message.contains(&format!("T = {t:?}")), "{message}");
        }
    }

    /// PR-24: a rate x[i]/T[i] that overflowed f64 was refused even when the rate ratio, the
    /// estimate, is representable. It is formed as (x[0] / x[1]) · (T[1] / T[0]) then. In both
    /// cases the rate x[1]/T[1] overflows (1e309, 2.1e309). For x = [11, 21] the interval is
    /// T[1] / T[0] times the one at T = [1, 1]. For x = [1, 1e19] both odds bounds lie below what
    /// the interval search resolves (R's are 2.5e-21 and 5.6e-19), so it returns the conservative
    /// (0, 1e-15), and the interval is (0, 1e-305) around the estimate 1e-309; it used to come
    /// back inverted, (1.0e-305, 7.6e-306). T = [1e-308, 1e-308], where both rates overflow, is in
    /// the test below.
    #[test]
    fn test_poisson_two_sample_accepts_overflowing_rate_with_representable_ratio() {
        let run = |x: [u64; 2], t: [f64; 2]| {
            poisson_test(&x, &t, 1.0, PoissonAlternative::TwoSided, 0.95)
                .unwrap_or_else(|e| panic!("x = {x:?}, T = {t:?}: {e}"))
        };
        let result = run([1, 10_000_000_000_000_000_000], [1.0, 1e-290]);
        assert_relative(result.estimate, 1e-309, 1e-12, "x = [1, 1e19] estimate");
        assert_eq!(result.conf_int.0, 0.0, "x = [1, 1e19] lower");
        assert_relative(result.conf_int.1, 1e-305, 1e-12, "x = [1, 1e19] upper");
        let (result, unit) = (run([11, 21], [1.0, 1e-308]), run([11, 21], [1.0, 1.0]));
        assert_relative(result.estimate, 11.0 / 21.0 * 1e-308, 1e-12, "estimate");
        assert_relative(result.conf_int.0, unit.conf_int.0 * 1e-308, 1e-12, "lower");
        assert_relative(result.conf_int.1, unit.conf_int.1 * 1e-308, 1e-12, "upper");
    }

    /// PR-24: the interval search has to return however it is called. Its start used to be the
    /// rate-ratio estimate, and from an infinite one the upper bound's bracketing halved infinity
    /// forever. poisson_test refuses the first two cases (their rate ratio overflows); called
    /// anyway, the search runs on the odds, which do not depend on T, and only the bound itself
    /// overflows: for x = [1, 1] it is 1e320 times (0.0127, 78.5), and for x = [5, 0] the lower
    /// bound is 1e310 times 0.916.
    #[test]
    fn test_rate_ratio_ci_overflows_only_with_the_bound() {
        for (x1, x2, t1, t2) in [(1, 1, 1e-160, 1e160), (11, 21, 1e-300, 1e300)] {
            assert_eq!(
                rate_ratio_ci(x1, x2, x1 + x2, t1, t2, 0.95),
                (f64::INFINITY, f64::INFINITY),
                "x = [{x1}, {x2}], T = [{t1:?}, {t2:?}]"
            );
        }
        assert_eq!(
            rate_ratio_ci(5, 0, 5, 1e-300, 1e10, 0.95),
            (f64::INFINITY, f64::INFINITY)
        );
    }

    fn interval(x: [u64; 2], t: [f64; 2]) -> (f64, f64) {
        poisson_test(&x, &t, 1.0, PoissonAlternative::TwoSided, 0.95)
            .unwrap_or_else(|e| panic!("x = {x:?}, T = {t:?}: {e}"))
            .conf_int
    }

    fn assert_relative(got: f64, want: f64, tolerance: f64, what: &str) {
        assert!(
            got.is_finite() && ((got - want) / want).abs() <= tolerance,
            "{what}: got {got:e}, want {want:e}"
        );
    }

    /// PR-24: the rate ratio's interval was searched on r itself, from fixed start points
    /// (1e-10; the estimate * 100 + 100), with the null's p formed as r * T[0] / (r * T[0] + T[1]).
    /// At time bases near the top of f64 the product, the sum and the start points overflowed, and
    /// the bounds came back wrong without an error. p depends on r only through the odds
    /// r * T[0] / T[1], so the interval at T is T[1] / T[0] times the one at T = [1, 1]. Expected
    /// values are R's poisson.test, which computes the interval that way.
    #[test]
    fn test_poisson_rate_ratio_interval_at_extreme_time_bases() {
        // Same p-value and estimate (1.336) as T = [1, 0.7]; the interval was (1.7977, 1.7977).
        let (lower, upper) = interval([21, 11], [1e308, 7e307]);
        let (lower_unit, upper_unit) = interval([21, 11], [1.0, 0.7]);
        assert_relative(lower, lower_unit, 1e-8, "x = [21, 11] lower");
        assert_relative(upper, upper_unit, 1e-8, "x = [21, 11] upper");
        assert_relative(lower_unit, 0.61596006297173, 1e-7, "R lower");
        assert_relative(upper_unit, 3.06913322981517, 1e-7, "R upper");
        // These lower bounds were infinite, above estimates of 1e308 and infinity.
        let lower = interval([1, 1], [1.0, 1e308]).0;
        assert_relative(lower, 0.0127393670836666e308, 1e-7, "x = [1, 1] lower");
        let lower = interval([100, 0], [1.0, 5e306]).0;
        assert_relative(lower, 26.6115770646767 * 5e306, 1e-7, "x = [100, 0] lower");
        // Both bounds were infinite; they bracket the estimate 1e308.
        let (lower, upper) = interval([1000, 1000], [1e-154, 1e154]);
        assert!(lower < 1e308 && 1e308 < upper, "({lower:e}, {upper:e})");
        assert_relative(lower, 9.15160512192926e307, 1e-7, "x = [1000, 1000] lower");
        assert_relative(upper, 1.09270448918713e308, 1e-7, "x = [1000, 1000] upper");
        // The lower bound was infinite; it is 1e298 times the one at T = [1e-10, 1].
        let lower = interval([5, 0], [1e-298, 1e10]).0;
        let lower_unit = interval([5, 0], [1e-10, 1.0]).0;
        assert_relative(lower, 1e298 * lower_unit, 1e-6, "x = [5, 0] lower");
    }

    /// PR-24: the same fixed start points broke the interval at ratios of time bases well inside
    /// f64. The upper bound's search never went below 1e-10, so at T = [1e12, 1] it returned 1e-10
    /// for a bound of 7.85e-11; its doubling stopped at 1e10, so at T = [1, 1e9] it returned
    /// 1.34e10 for a bound of 3.9e10. Expected values are R's poisson.test. Where T[1] / T[0]
    /// itself overflows (1e310) or is subnormal (1e-310), the bounds are still T[1] / T[0] times
    /// the odds bounds.
    #[test]
    fn test_poisson_rate_ratio_interval_scales_with_time_bases() {
        let (lower, upper) = interval([1, 1], [1e12, 1.0]);
        assert_relative(lower, 1.27393670836666e-14, 1e-7, "T = [1e12, 1] lower");
        assert_relative(upper, 7.84968353162628e-11, 1e-7, "T = [1e12, 1] upper");
        let (lower, upper) = interval([0, 1], [1.0, 1e9]);
        assert_eq!(lower, 0.0);
        assert_relative(upper, 3.9e10, 1e-7, "T = [1, 1e9] upper");
        for (x, t, t_unit, factor) in [
            ([1, 1000], [1e-10, 1e300], [1e-10, 1.0], 1e300),
            ([10_000, 1], [1e300, 1e-10], [1.0, 1e-10], 1e-300),
        ] {
            let ((lower, upper), (lower_unit, upper_unit)) = (interval(x, t), interval(x, t_unit));
            let what = format!("T = {t:?}");
            assert_relative(lower, lower_unit * factor, 1e-10, &what);
            assert_relative(upper, upper_unit * factor, 1e-10, &what);
        }
    }

    /// PR-24: the odds search kept the old search's fixed limits for the upper bound, a 1e-10
    /// floor and a 1e10 ceiling, and they truncated bounds without an error: x = [1e6, 1] at
    /// 0.9999 gave 1.28e10 (T = [1, 1]) and 1.28e8 (T = [100, 1]), x = [2e5, 1] at 0.99999 gave
    /// 1.024e10, x = [1e8, 1] at 0.99 with T = [100, 1] gave 1.00000001e8, and x = [1, 1e11] and
    /// x = [0, 1e11] with T = [1, 1000] gave 1e-10 and 1e-7. The limits are now the odds the tail
    /// can resolve, 1e-15 and 1e15. Expected values are R's qbeta. For the first four, which have
    /// x[1] = 1, that is qbeta(alpha/2, 1, x[0] + 1), the quantile of 1 - p, equal to the closed
    /// form 1/q - 1 with (1 - q)^n = 1 - alpha/2; poisson.test forms these bounds from p near 1
    /// and is 5.1e-7 to 1.9e-6 off them. The four come out 2.8e-7 to 6.0e-7 off, one of them
    /// below. They are limited by the 1 - p that statrs works from, good near odds s to about
    /// 5.6e-17 * (1 + s) relative (see ODDS_MAX): 1.1e-6 at 2e10, 2.2e-6 at 4e10. So 1e-5 is
    /// asserted. (The bisection used to stop once the tail was within 1e-10 of alpha/2, which
    /// bounded the error only at about 1e-10 / (alpha/2) relative, 2e-5 at 0.99999, and the four
    /// came out 2.8e-7 to 2.6e-6 above; the rule is relative now, see
    /// test_validate_poisson_rate_ratio_interval_at_levels_near_one.) At totals of 1e11 statrs'
    /// binomial CDF is itself off: the two bounds there are 2.9e-5 and 4.7e-5 from R, so 1e-4
    /// (STAT-60 / T3-WP40).
    #[test]
    fn test_poisson_rate_ratio_interval_beyond_old_odds_limits() {
        let upper = |x: [u64; 2], t: [f64; 2], conf_level: f64| {
            poisson_test(&x, &t, 1.0, PoissonAlternative::TwoSided, conf_level)
                .unwrap_or_else(|e| panic!("x = {x:?}, T = {t:?}: {e}"))
                .conf_int
                .1
        };
        let within = |tolerance: f64, cases: &[([u64; 2], [f64; 2], f64, f64)]| {
            for &(x, t, conf_level, want) in cases {
                let what = format!("x = {x:?}, T = {t:?}, conf_level = {conf_level}");
                assert_relative(upper(x, t, conf_level), want, tolerance, &what);
            }
        };
        within(
            1e-5,
            &[
                ([1_000_000, 1], [1.0, 1.0], 0.9999, 1.999951999484e10),
                ([1_000_000, 1], [100.0, 1.0], 0.9999, 1.999951999484e8),
                ([200_000, 1], [1.0, 1.0], 0.99999, 4.000009999910e10),
                ([100_000_000, 1], [100.0, 1.0], 0.99, 1.994995842784e8),
            ],
        );
        within(
            1e-4,
            &[
                ([1, 100_000_000_000], [1.0, 1.0], 0.95, 5.571643391e-11),
                ([0, 100_000_000_000], [1.0, 1000.0], 0.95, 3.688879454e-8),
            ],
        );
    }

    /// PR-24: a bound whose odds lie outside [1e-15, 1e15], the range the search resolves, came
    /// back wrong without an error. A bracketing loop had stopped at the limit with the bound
    /// further out, or its start was already beyond it; the tail there, evaluated at the clamped
    /// odds, did not straddle alpha / 2, and the bisection converged on a bracket end. Below
    /// 1e-15 the interval came back inverted, both ends above the estimate: x = [1, 1e16] gave
    /// (1.0e-15, 7.6e-16) around 1e-16, R (2.53e-18, 5.57e-16). Above 1e15 the lower bound came
    /// back as the estimate and the upper as the bracket's start, 100 * estimate + 100:
    /// x = [1e16, 1] gave (1.0e16, 1.0e18), R (1.79e15, 3.95e17), and x = [5e13, 1] an upper bound
    /// of 5.0e15, R 1.97e15. Where a loop stopped at the limit the bound came back near it, on
    /// the wrong side: at 0.9999, x = [1, 1e11] gave a lower bound of 1.0e-15, R 5.0e-16, and
    /// x = [1e11, 1] an upper bound of 1.28e15, R 2.0e15. Such a bound is now the limit, 0 or
    /// ODDS_MIN below the range and ODDS_MAX or infinity above it, so the limit lies outside R's
    /// bound and the interval contains the estimate; the resolved bound matches R's to the
    /// asserted tolerance. That holds while statrs' tail at the limit is on the right side of
    /// alpha / 2, which it is at totals up to about 1e13: x = [1, 1e10] and [1e10, 1] at 0.999999
    /// are such cases. Expected values are R's qbeta, and for x[1] = 1 or x[0] = 1 the closed form
    /// it agrees with.
    #[test]
    fn test_poisson_rate_ratio_interval_beyond_resolved_odds_is_conservative() {
        let test = |x: [u64; 2], level: f64| {
            poisson_test(&x, &[1.0, 1.0], 1.0, PoissonAlternative::TwoSided, level)
                .unwrap_or_else(|e| panic!("x = {x:?}: {e}"))
        };
        // One bound beyond the range at a total where statrs' CDF is accurate: R's lower bound of
        // x = [1, 1e10] is 5.0e-17 and its upper bound of x = [1e10, 1] 2.0e16. The resolved
        // bounds are 3.0e-7 from R's (the CDF at n = 1e10), hence 1e-6.
        let (lower, upper) = test([1, 10_000_000_000], 0.999999).conf_int;
        assert_eq!(lower, 0.0, "x = [1, 1e10] lower");
        assert_relative(upper, 1.742221502680587e-9, 1e-6, "x = [1, 1e10] upper");
        let (lower, upper) = test([10_000_000_000, 1], 0.999999).conf_int;
        assert_eq!(upper, f64::INFINITY, "x = [1e10, 1] upper");
        assert_relative(lower, 5.739798288916748e8, 1e-6, "x = [1e10, 1] lower");
        // Totals of 1e16 and 1e17 are beyond where statrs' CDF is accurate. These cases come out
        // as the limits because its tails at the limits happen to fall on the right side of
        // alpha / 2 there; exact tails would give the same, since the true bounds lie beyond the
        // range too. So they pin that result, and a change to the tail or the range (T3-WP40)
        // has to revisit them. At 5e15 to 9e15 statrs' tails fall on the wrong side, and
        // x = [1, 5e15] and [5e15, 1] came back inverted (see
        // test_poisson_rate_ratio_interval_contains_the_estimate).
        // Both bounds below 1e-15: x = [3, 1e17] has R's (6.19e-18, 8.77e-17).
        for x in [[1, 10_000_000_000_000_000], [3, 100_000_000_000_000_000]] {
            let result = test(x, 0.95);
            assert_eq!(result.conf_int, (0.0, ODDS_MIN), "x = {x:?}");
            assert!(result.estimate < ODDS_MIN, "x = {x:?}: {}", result.estimate);
        }
        // Both bounds above 1e15: for x = [1e17, 1] the closed form gives (1.79e16, 3.95e18). (R
        // itself loses the 1 in 1e17 + 1 and computes the bounds of x = [1e17, 0].)
        for x in [[10_000_000_000_000_000, 1], [100_000_000_000_000_000, 1]] {
            let result = test(x, 0.95);
            assert_eq!(result.conf_int, (ODDS_MAX, f64::INFINITY), "x = {x:?}");
            assert!(result.estimate > ODDS_MAX, "x = {x:?}: {}", result.estimate);
        }
        // One bound beyond the range; the other is resolved and keeps its value. Both resolved
        // bounds are 1.2e-5 from R's, inside R's interval (the CDF at n = 1e11).
        let (lower, upper) = test([1, 100_000_000_000], 0.9999).conf_int;
        assert_eq!(lower, 0.0, "x = [1, 1e11] lower");
        assert_relative(upper, 1.2506671286e-10, 1e-4, "x = [1, 1e11] upper");
        let (lower, upper) = test([100_000_000_000, 1], 0.9999).conf_int;
        assert_eq!(upper, f64::INFINITY, "x = [1e11, 1] upper");
        assert_relative(lower, 7.9957326545e9, 1e-4, "x = [1e11, 1] lower");
        // At 95% the upper bound of x = [m, 1] passes 1e15 from m of about 2.5e13 (R 1.18e15,
        // 1.97e15 and 3.95e15 here). The resolved lower bounds are 0.5% to 1.9% from R's, which is
        // how accurate statrs' binomial CDF is at these totals (STAT-60 / T3-WP40), hence 3e-2.
        // Two of them (3e13, 5e13) lie above R's, inside its interval.
        for (x, lower_r) in [
            ([30_000_000_000_000, 1], 5.3844077761e12),
            ([50_000_000_000_000, 1], 8.9740129602e12),
            ([100_000_000_000_000, 1], 1.7948025920e13),
        ] {
            let (lower, upper) = test(x, 0.95).conf_int;
            assert_eq!(upper, f64::INFINITY, "x = {x:?}");
            assert_relative(lower, lower_r, 3e-2, &format!("x = {x:?} lower"));
        }
    }

    /// PR-24: near x[0] = x[1] from totals of about 4e7, and for any counts from about 3e14,
    /// statrs' binomial CDF misjudges the tail, and the interval search returned a bound on the
    /// wrong side of the estimate. The interval came back inverted or without the estimate in it:
    /// x = [1, 5e15] gave (5.6e-15, 1e-15) around 2e-16, x = [5e15, 1] (1e15, 1.8e14) around 5e15,
    /// x = [1000, 1e15] (9.25e-13, 9.54e-13) around 1e-12, x = [2e7, 2e7] (1.0000009, 1.00062)
    /// around 1 and x = [1e8, 1e8] (1.0000038, 0.9999962). Exact tails never put a bound on that
    /// side (see rate_ratio_ci), so such a bound is widened to 0 or infinity now, and the interval
    /// contains the estimate and is not inverted. It need not contain R's interval, and which
    /// bound is widened depends on statrs' errors, which a T3-WP40 tail fix may change, so only
    /// the containment is asserted, for T = [1, 1] and T = [1, 3], plus the one bound statrs gets
    /// right in a case near x[0] = x[1] (x = [2e7, 2e7]: R's upper bound 1.000620037177023).
    #[test]
    fn test_poisson_rate_ratio_interval_contains_the_estimate() {
        for x in [
            [1, 5_000_000_000_000_000],
            [5_000_000_000_000_000, 1],
            [1000, 1_000_000_000_000_000],
            [1000, 500_000_000_000_000],
            [300_000_000_000_000, 1],
            [1, 7_000_000_000_000_000],
            [7_000_000_000_000_000, 1],
            [10, 10_000_000_000_000_000],
            [100_000_000_000_000_000, 99],
            [20_000_000, 20_000_000],
            [100_000_000, 100_000_000],
        ] {
            for t in [[1.0, 1.0], [1.0, 3.0]] {
                let result = poisson_test(&x, &t, 1.0, PoissonAlternative::TwoSided, 0.95)
                    .unwrap_or_else(|e| panic!("x = {x:?}, T = {t:?}: {e}"));
                let ((lower, upper), estimate) = (result.conf_int, result.estimate);
                assert!(
                    lower <= estimate && estimate <= upper,
                    "x = {x:?}, T = {t:?}: ({lower:e}, {upper:e}) around {estimate:e}"
                );
            }
        }
        let upper = interval([20_000_000, 20_000_000], [1.0, 1.0]).1;
        assert_relative(upper, 1.000620037177023, 1e-9, "x = [2e7, 2e7] upper");
    }

    /// PR-24: the bisection stopped once the tail was within 1e-10 of alpha / 2, an absolute rule
    /// that bounds a bound's error only at about 1e-10 / (alpha / 2) relative. At levels close to
    /// 1 that outweighed everything else: at conf_level 1 - 1e-12, x = [5, 5] gave (0.00195,
    /// 800.5) for R's (0.00115, 870.1), both bounds inside R's interval, and at 1 - 1e-9 it gave
    /// (0.00464, 225.4) and x = [50, 30] (0.4102, 8.136). The rule is relative now, within
    /// 1e-10 * alpha / 2. Expected values are R's qbeta (on p for the lower bound, on 1 - p for
    /// the upper). What limits these bounds now is the lower bound's tail, 1 - F(x[0] - 1) with
    /// statrs' F close to 1, which keeps only 2^-53 absolute (see find_odds_bound). At 1 - 1e-12
    /// and 1 - 1e-9, alpha / 2 lies halfway between multiples of 2^-53, so the rounding of
    /// 1 - F still decides the comparison with alpha / 2 correctly, and the bounds are within
    /// 4e-11 of R's. At 1 - 2e-12 it lies on one, and the lower bound is 6.5e-6 above R's, hence
    /// 1e-5 there.
    #[test]
    fn test_validate_poisson_rate_ratio_interval_at_levels_near_one() {
        for (x, level, lower_r, upper_r, lower_tolerance) in [
            (
                [5, 5],
                1.0 - 1e-12,
                1.149281586913931e-3,
                870.1087804645126,
                1e-6,
            ),
            (
                [5, 5],
                1.0 - 1e-9,
                4.604373078948037e-3,
                217.1848333863663,
                1e-6,
            ),
            (
                [50, 30],
                1.0 - 1e-9,
                0.4076236485346079,
                8.139273793300017,
                1e-6,
            ),
            (
                [5, 5],
                1.0 - 2e-12,
                1.320591987988113e-3,
                757.2361555240643,
                1e-5,
            ),
        ] {
            let (lower, upper) =
                poisson_test(&x, &[1.0, 1.0], 1.0, PoissonAlternative::TwoSided, level)
                    .unwrap_or_else(|e| panic!("x = {x:?}, conf_level = {level}: {e}"))
                    .conf_int;
            let what = format!("x = {x:?}, conf_level = {level}");
            assert_relative(lower, lower_r, lower_tolerance, &format!("{what}: lower"));
            assert_relative(upper, upper_r, 1e-6, &format!("{what}: upper"));
        }
    }

    /// PR-24: equal time bases at either end of f64 give the test at T = [1, 1], whose values are
    /// R's poisson.test(c(11, 21), c(1, 1)). At T = [1e308, 1e308] the null's r * T[0] + T[1]
    /// overflowed and was refused (and before that made the null's p 0); p is formed from
    /// logarithms there now. At T = [1e-308, 1e-308] both rates x[i]/T[i] overflow and the call was
    /// refused although the estimate is 11/21; it is formed from the counts and T[1] / T[0] now.
    #[test]
    fn test_poisson_two_sample_equal_time_bases_at_f64_limits() {
        let run = |t: [f64; 2]| {
            poisson_test(&[11, 21], &t, 1.0, PoissonAlternative::TwoSided, 0.95)
                .unwrap_or_else(|e| panic!("T = {t:?}: {e}"))
        };
        let unit = run([1.0, 1.0]);
        let large = run([1e308, 1e308]);
        assert_eq!(large.parameter, 16.0);
        assert_eq!(large.parameter, unit.parameter);
        assert_eq!(large.p_value, unit.p_value);
        assert_eq!(large.conf_int, unit.conf_int);
        assert_relative(large.estimate, 11.0 / 21.0, 1e-12, "estimate");
        let small = run([1e-308, 1e-308]);
        assert_eq!(small.parameter, unit.parameter);
        assert_eq!(small.p_value, unit.p_value);
        assert_eq!(small.conf_int, unit.conf_int);
        assert_eq!(small.estimate, unit.estimate);
        assert_eq!(small.estimate, 11.0 / 21.0);
        assert_relative(unit.p_value, 0.110184165183455, 1e-9, "p-value vs R");
        assert_relative(unit.conf_int.0, 0.228077423684261, 1e-7, "lower vs R");
        assert_relative(unit.conf_int.1, 1.13643731482008, 1e-7, "upper vs R");
    }

    #[test]
    fn test_poisson_display() {
        let result =
            poisson_test(&[137], &[24.19893], 1.0, PoissonAlternative::TwoSided, 0.95).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Exact Poisson test"));
        assert!(display.contains("p-value"));
    }
}
