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

/// The two-sample null's p = r·T₁ / (r·T₁ + T₂) for finite, positive `r`, `t1` and `t2`.
///
/// The direct form is used whenever r·T₁ > 0 and the denominator is finite, so every p in that
/// range is computed as before. Otherwise it is 1 / (1 + T₂ / (r·T₁)) with the ratio formed from
/// logarithms; its exponential overflows only for a p below 1 / `f64::MAX`, which then comes out
/// as 0.
fn null_success_probability(r: f64, t1: f64, t2: f64) -> f64 {
    let r_t1 = r * t1;
    let denominator = r_t1 + t2;
    if r_t1 > 0.0 && denominator.is_finite() {
        r_t1 / denominator
    } else {
        1.0 / (1.0 + (t2.ln() - t1.ln() - r.ln()).exp())
    }
}

/// The two-sample null is X₁ | n ~ Binomial(n, p) with p = r·T₁ / (r·T₁ + T₂). Refuse an `r * T[0]`
/// that is not finite and > 0 (p would be NaN, or 0 although r and T[0] are > 0).
fn two_sample_null_p(r: f64, t1: f64, t2: f64) -> EconResult<f64> {
    let r_t1 = r * t1;
    if !(r_t1.is_finite() && r_t1 > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("r * T[0] = {r:?} * {t1:?} = {r_t1:?} must be finite and > 0"),
        });
    }
    Ok(null_success_probability(r, t1, t2))
}

/// The two-sample estimate, the ratio of the rates x[i] / T[i], or infinity when x[1] = 0. Refuse
/// a rate that overflows f64 and, when x[1] > 0, a ratio that does.
fn two_sample_estimate(x1: u64, x2: u64, t1: f64, t2: f64) -> EconResult<f64> {
    let refuse = |what: String| EconError::InvalidSpecification {
        message: format!("x = [{x1}, {x2}], T = [{t1:?}, {t2:?}]: {what} overflows f64"),
    };
    let rate1 = x1 as f64 / t1;
    let rate2 = x2 as f64 / t2;
    if !rate1.is_finite() {
        return Err(refuse(format!("the rate x[0]/T[0] = {x1}/{t1:?}")));
    }
    if !rate2.is_finite() {
        return Err(refuse(format!("the rate x[1]/T[1] = {x2}/{t2:?}")));
    }
    if x2 == 0 {
        return Ok(f64::INFINITY);
    }
    let ratio = rate1 / rate2;
    if !ratio.is_finite() {
        return Err(refuse(format!(
            "the rate ratio (x[0]/T[0]) / (x[1]/T[1]) = ({x1}/{t1:?}) / ({x2}/{t2:?})"
        )));
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
/// - two-sample: the total `x[0] + x[1]` does not fit in `u64`; `r * T[0]` is not finite and
///   positive; a rate `x[0] / T[0]` or `x[1] / T[1]` overflows `f64`, or, when `x[1]` is not 0,
///   so does the rate ratio `(x[0] / T[0]) / (x[1] / T[1])`.
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
    // where p = λ₁T₁ / (λ₁T₁ + λ₂T₂) = rT₁ / (rT₁ + T₂)
    let p = two_sample_null_p(r, t1, t2)?;
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

/// Cap on each bracketing loop in [`find_odds_bound`]. The search starts from the estimate
/// x[0] / max(x[1], 1), which is 0 or lies in [2^-64, 2^64]; its longest walk, the upper bound's
/// halving from 2^64 down to its 1e-10 floor, takes 98 steps, and the other loops need at most
/// 27. The cap leaves every search as it is and only guarantees that each loop ends.
const MAX_BRACKET_STEPS: usize = 128;

/// Compute confidence interval for rate ratio.
///
/// Uses the exact conditional method: given n = x1 + x2 (checked by the caller), X₁ is binomial
/// with odds p / (1 - p) = r·T₁ / T₂. The bounds are searched on those odds, which depend on the
/// counts and `conf_level` only, and then multiplied by T₂ / T₁, as R's `poisson.test` does.
fn rate_ratio_ci(x1: u64, x2: u64, n: u64, t1: f64, t2: f64, conf_level: f64) -> (f64, f64) {
    let alpha = 1.0 - conf_level;

    if n == 0 {
        return (0.0, f64::INFINITY);
    }

    // Use numerical search to find CI bounds
    // The p-value equals alpha/2 at the CI bounds

    // Lower bound: find r such that P(X₁ >= x₁ | H₀: ratio=r) = alpha/2
    let lower = if x1 == 0 {
        0.0
    } else {
        rate_ratio_from_odds(find_odds_bound(x1, x2, n, alpha / 2.0, true), t1, t2)
    };

    // Upper bound: find r such that P(X₁ <= x₁ | H₀: ratio=r) = alpha/2
    let upper = if x2 == 0 {
        f64::INFINITY
    } else {
        rate_ratio_from_odds(find_odds_bound(x1, x2, n, alpha / 2.0, false), t1, t2)
    };

    (lower, upper)
}

/// The rate ratio r = s·T₂ / T₁ for odds `s` > 0 found by [`find_odds_bound`]. The product is
/// formed directly when T₂ / T₁ is a normal number and otherwise from logarithms, so it overflows
/// or underflows only when the bound itself lies outside f64.
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
/// `n` is `x1 + x2`, checked by the caller. The odds do not depend on the time bases, so the
/// start points and floors below are on the scale of the counts whatever T is; every loop has a
/// fixed iteration cap, and the search returns a finite s > 0 for any input.
fn find_odds_bound(x1: u64, x2: u64, n: u64, alpha: f64, find_lower: bool) -> f64 {
    // Compute the tail probability for given odds s; p = s / (1 + s), that is the null's p at
    // T = [1, 1].
    let compute_tail_p = |s: f64| -> f64 {
        let p = null_success_probability(s, 1.0, 1.0);
        // Ensure p is in valid range
        let p = p.clamp(1e-15, 1.0 - 1e-15);
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
            if !(lo > 1e-15 && compute_tail_p(lo) > alpha) {
                break;
            }
            lo /= 10.0;
        }
        for _ in 0..MAX_BRACKET_STEPS {
            if !(hi < odds_est * 10.0 && compute_tail_p(hi) < alpha) {
                break;
            }
            hi *= 2.0;
        }
    } else {
        // For upper bound: need compute_tail_p(hi) < alpha < compute_tail_p(lo)
        for _ in 0..MAX_BRACKET_STEPS {
            if !(lo > 1e-10 && compute_tail_p(lo) < alpha) {
                break;
            }
            lo /= 2.0;
        }
        for _ in 0..MAX_BRACKET_STEPS {
            if !(hi < 1e10 && compute_tail_p(hi) > alpha) {
                break;
            }
            hi *= 2.0;
        }
    }

    // Bisection search
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let p = compute_tail_p(mid);

        if (p - alpha).abs() < 1e-10 {
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
        // Two-sample: r * T[0] overflowing made the null's p NaN, which gave an opaque "Invalid
        // Binomial parameters" error, or Ok with a NaN expected count when x = [0, 0].
        for x in [[11, 21], [0, 0]] {
            let message = refusal(&x, &[1e200, 1.0], 1e200, 0.95);
            assert!(message.contains("r * T[0] = 1e200 * 1e200"), "{message}");
        }
        // Two-sample: r * T[0] underflowing to 0 would make the null's p 0 although r and T[0]
        // are > 0.
        let message = refusal(&[11, 21], &[1e-200, 1.0], 1e-200, 0.95);
        assert!(message.contains("r * T[0] = 1e-200 * 1e-200"), "{message}");
    }

    /// PR-24: the rate ratio's interval search started from the point estimate. When
    /// (x[0]/T[0]) / (x[1]/T[1]) overflowed f64, the upper bound's bracketing started at
    /// infinity, where halving does not move, and never returned; every T here is finite and > 0,
    /// so the T check let it through. Refused before any statrs call now. Each case pins one of
    /// the three refusals by its own fragment.
    #[test]
    fn test_poisson_two_sample_refuses_overflowing_rates() {
        for (x, t, fragment) in [
            ([1, 1], [1e-160, 1e160], "the rate ratio"),
            ([11, 21], [1e-300, 1e300], "the rate ratio"),
            (
                [10_000_000_000_000_000_000, 1],
                [1e-290, 1.0],
                "the rate x[0]/T[0]",
            ),
            (
                [1, 10_000_000_000_000_000_000],
                [1.0, 1e-290],
                "the rate x[1]/T[1]",
            ),
        ] {
            let message = refusal(&x, &t, 1.0, 0.95);
            assert!(message.contains(fragment), "{message}");
            assert!(message.contains("overflows f64"), "{message}");
            assert!(message.contains(&format!("x = {x:?}")), "{message}");
            assert!(message.contains(&format!("T = {t:?}")), "{message}");
        }
    }

    /// PR-24: the interval search has to return however it is called. Its start used to be the
    /// rate-ratio estimate, and from an infinite one the upper bound's bracketing halved infinity
    /// forever. poisson_test refuses these time bases; called anyway, the search runs on the
    /// odds, which do not depend on T, and only the bound itself overflows: for x = [1, 1] it is
    /// 1e320 times (0.0127, 78.5), and for x = [5, 0] the lower bound is 1e310 times 0.916.
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

    /// PR-24: an r * T[0] + T[1] that overflowed f64 was refused (and before that made the null's
    /// p 0). p is formed from logarithms there now, so T = [1e308, 1e308] gives the test at
    /// T = [1, 1], whose values are R's poisson.test(c(11, 21), c(1, 1)).
    #[test]
    fn test_poisson_two_sample_null_p_survives_overflowing_denominator() {
        let run = |t: [f64; 2]| {
            poisson_test(&[11, 21], &t, 1.0, PoissonAlternative::TwoSided, 0.95).unwrap()
        };
        let (large, unit) = (run([1e308, 1e308]), run([1.0, 1.0]));
        assert_eq!(large.parameter, 16.0);
        assert_eq!(large.parameter, unit.parameter);
        assert_eq!(large.p_value, unit.p_value);
        assert_eq!(large.conf_int, unit.conf_int);
        assert_relative(large.estimate, 11.0 / 21.0, 1e-12, "estimate");
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
