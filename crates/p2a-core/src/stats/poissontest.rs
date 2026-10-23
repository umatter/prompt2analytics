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
/// still takes seconds. The two-sample test does not use that series: its p-value and interval
/// use the binomial CDF, whose continued fraction statrs stops after 140 terms, and every loop of
/// its interval search has a fixed iteration cap. Its counts only need a total that fits in
/// `u64`; that the test returns says nothing about its accuracy at large totals.
pub const MAX_ONE_SAMPLE_COUNT: u64 = 1 << 52;

/// Refuse a one-sample count above [`MAX_ONE_SAMPLE_COUNT`] before statrs sees it.
fn check_one_sample_count(x: u64) -> EconResult<()> {
    if x > MAX_ONE_SAMPLE_COUNT {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "x[0] = {x} is above the largest event count the one-sample exact Poisson test \
                 supports ({MAX_ONE_SAMPLE_COUNT} = 2^52); beyond it the incomplete gamma series \
                 behind the Poisson CDF and the exact interval cannot advance in f64 and does not \
                 finish"
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

/// The two-sample null is X₁ | n ~ Binomial(n, p) with p = r·T₁ / (r·T₁ + T₂). Refuse an `r * T[0]`
/// that is not finite and > 0 (p would be NaN, or 0 although r and T[0] are > 0) and a
/// denominator that overflows (p would be 0 whatever the true value).
fn two_sample_null_p(r: f64, t1: f64, t2: f64) -> EconResult<f64> {
    let r_t1 = r * t1;
    if !(r_t1.is_finite() && r_t1 > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("r * T[0] = {r:?} * {t1:?} = {r_t1:?} must be finite and > 0"),
        });
    }
    let denominator = r_t1 + t2;
    if !denominator.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: format!("r * T[0] + T[1] = {r_t1:?} + {t2:?} overflows f64"),
        });
    }
    Ok(r_t1 / denominator)
}

/// The two-sample estimate is the ratio of the rates x[i] / T[i], and the interval search starts
/// from it. Refuse a rate that overflows f64 and, when x[1] > 0, a ratio that does: the estimate
/// would be infinite or NaN, and the upper bound's search would start at infinity.
fn check_two_sample_rates(x1: u64, x2: u64, t1: f64, t2: f64) -> EconResult<()> {
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
    if x2 > 0 && !(rate1 / rate2).is_finite() {
        return Err(refuse(format!(
            "the rate ratio (x[0]/T[0]) / (x[1]/T[1]) = ({x1}/{t1:?}) / ({x2}/{t2:?})"
        )));
    }
    Ok(())
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
/// - one-sample: the count is above [`MAX_ONE_SAMPLE_COUNT`], or the expected count `r * T` is not
///   finite and > 0 (it overflows, or underflows to 0);
/// - two-sample: the total `x[0] + x[1]` does not fit in `u64`; `r * T[0]` is not finite and
///   positive, or `r * T[0] + T[1]` overflows `f64`; a rate `x[0] / T[0]` or `x[1] / T[1]`
///   overflows `f64`, or, when `x[1]` is not 0, so does the rate ratio
///   `(x[0] / T[0]) / (x[1] / T[1])`.
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
            message: format!("time base T must be finite and > 0, got {ti}"),
        });
    }

    if !(r.is_finite() && r > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("rate (ratio) r must be finite and > 0, got {r}"),
        });
    }

    if !(conf_level > 0.0 && conf_level < 1.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("conf_level must be strictly between 0 and 1, got {conf_level}"),
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
    check_two_sample_rates(x1, x2, t1, t2)?;

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

    // Rate ratio estimate
    let rate1 = x1 as f64 / t1;
    let rate2 = x2 as f64 / t2;
    let estimate = if rate2 > 0.0 {
        rate1 / rate2
    } else {
        f64::INFINITY
    };

    // Confidence interval for rate ratio using profile likelihood
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
/// `poisson_test` keeps `x` at or below [`MAX_ONE_SAMPLE_COUNT`]; `x + 1` is still checked
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

/// Cap on each bracketing loop in [`find_rate_ratio_bound`]. The longest finite walk is the upper
/// bound's: halving from `f64::MAX` (about 2^1024) down to its 1e-10 floor (about 2^-33) takes
/// 1058 steps; the other loops need at most a few dozen. The cap therefore leaves every search
/// from a finite start as it was and only stops one that cannot finish.
const MAX_BRACKET_STEPS: usize = 1100;

/// Compute confidence interval for rate ratio.
///
/// Uses exact conditional method based on binomial distribution. `n` is `x1 + x2`, checked by
/// the caller.
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
        find_rate_ratio_bound(x1, x2, n, t1, t2, alpha / 2.0, true)
    };

    // Upper bound: find r such that P(X₁ <= x₁ | H₀: ratio=r) = alpha/2
    let upper = if x2 == 0 {
        f64::INFINITY
    } else {
        find_rate_ratio_bound(x1, x2, n, t1, t2, alpha / 2.0, false)
    };

    (lower, upper)
}

/// Find rate ratio bound using bisection search.
///
/// For the lower bound, we find r such that P(X₁ >= x₁ | H₀: ratio=r) = alpha
/// As r increases, P(X₁ >= x₁) increases, so we want to find the smallest r
/// where this probability just reaches alpha.
///
/// For the upper bound, we find r such that P(X₁ <= x₁ | H₀: ratio=r) = alpha
/// As r increases, P(X₁ <= x₁) decreases, so we want to find the largest r
/// where this probability just reaches alpha.
///
/// `n` is `x1 + x2`, checked by the caller.
///
/// Every loop has a fixed iteration cap, so the search returns for any input. `poisson_test`
/// refuses rates whose ratio overflows; called with one anyway, the upper bound's search starts
/// at infinity and returns infinity.
fn find_rate_ratio_bound(
    x1: u64,
    x2: u64,
    n: u64,
    t1: f64,
    t2: f64,
    alpha: f64,
    find_lower: bool,
) -> f64 {
    // Compute the tail probability for a given rate ratio r
    let compute_tail_p = |r: f64| -> f64 {
        let p = r * t1 / (r * t1 + t2);
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

    // Initial estimate for the rate ratio
    let rate_ratio_est = (x1 as f64 / t1) / (x2.max(1) as f64 / t2);

    // Set up initial bounds
    let (mut lo, mut hi) = if find_lower {
        // For lower CI: search between very small value and the point estimate
        (1e-10, rate_ratio_est.max(1e-5))
    } else {
        // For upper CI: search between point estimate and a large value
        (rate_ratio_est.max(1e-10), rate_ratio_est * 100.0 + 100.0)
    };

    // Expand bounds if needed to bracket the solution. Each loop runs at most
    // MAX_BRACKET_STEPS times: an end point at infinity no longer moves when halved, and the
    // upper bound's search used to halve one forever.
    if find_lower {
        // For lower bound: need compute_tail_p(lo) < alpha < compute_tail_p(hi)
        for _ in 0..MAX_BRACKET_STEPS {
            if !(lo > 1e-15 && compute_tail_p(lo) > alpha) {
                break;
            }
            lo /= 10.0;
        }
        for _ in 0..MAX_BRACKET_STEPS {
            if !(hi < rate_ratio_est * 10.0 && compute_tail_p(hi) < alpha) {
                break;
            }
            hi *= 2.0;
        }
    } else {
        // For upper bound: need compute_tail_p(hi) < alpha < compute_tail_p(lo)
        for _ in 0..MAX_BRACKET_STEPS {
            if !(lo.is_finite() && lo > 1e-10 && compute_tail_p(lo) < alpha) {
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
            // For lower bound: P increases with r
            // If P(mid) > alpha, we need smaller r: hi = mid
            // If P(mid) < alpha, we need larger r: lo = mid
            if p > alpha {
                hi = mid;
            } else {
                lo = mid;
            }
        } else {
            // For upper bound: P decreases with r
            // If P(mid) > alpha, we need larger r: lo = mid
            // If P(mid) < alpha, we need smaller r: hi = mid
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
            MAX_ONE_SAMPLE_COUNT + 1,
            (1 << 53) - 2,
            1 << 53,
            (1 << 53) + 2,
            u64::MAX - 1,
            u64::MAX,
        ] {
            let message = refusal(&[x], &[3.5], 1.0, 0.95);
            assert!(message.contains(&format!("x[0] = {x}")), "{message}");
            assert!(
                message.contains(&MAX_ONE_SAMPLE_COUNT.to_string()),
                "{message}"
            );
        }
    }

    /// The bound is inclusive. Checked on the validation step: a full test at 2^52 takes seconds.
    #[test]
    fn test_poisson_one_sample_bound_is_inclusive() {
        // Re-exported next to poisson_test, so callers (PR-01m's MCP cap) can name it.
        assert_eq!(crate::stats::MAX_ONE_SAMPLE_COUNT, 4_503_599_627_370_496);
        assert!(check_one_sample_count(MAX_ONE_SAMPLE_COUNT).is_ok());
        assert!(check_one_sample_count(0).is_ok());
        assert!(check_one_sample_count(MAX_ONE_SAMPLE_COUNT + 1).is_err());
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
        // Two-sample: r * T[0] + T[1] overflowing made p = 0 silently (the true p here is 1/2).
        let message = refusal(&[11, 21], &[1e308, 1e308], 1.0, 0.95);
        assert!(message.contains("r * T[0] + T[1]"), "{message}");
    }

    /// Round-2 review of PR-24: the rate ratio's interval search started from the point estimate.
    /// When (x[0]/T[0]) / (x[1]/T[1]) overflowed f64, the upper bound's bracketing started at
    /// infinity, where halving does not move, and never returned; every T here is finite and > 0,
    /// so the T check let it through. Refused before any statrs call now.
    #[test]
    fn test_poisson_two_sample_refuses_overflowing_rates() {
        for (x, t) in [
            ([1, 1], [1e-160, 1e160]),
            ([11, 21], [1e-300, 1e300]),
            ([10_000_000_000_000_000_000, 1], [1e-290, 1.0]),
        ] {
            let message = refusal(&x, &t, 1.0, 0.95);
            assert!(message.contains("overflows f64"), "{message}");
            assert!(message.contains(&format!("x = {x:?}")), "{message}");
            assert!(message.contains(&format!("T = {t:?}")), "{message}");
        }
    }

    /// Round-2 review of PR-24: find_rate_ratio_bound has to return however it is called. With an
    /// infinite starting estimate the upper bound's bracketing used to halve infinity forever.
    /// Here the true bounds overflow f64 too, so infinity is the right answer.
    #[test]
    fn test_find_rate_ratio_bound_returns_from_infinite_start() {
        for (x1, x2, t1, t2) in [(1, 1, 1e-160, 1e160), (11, 21, 1e-300, 1e300)] {
            let upper = find_rate_ratio_bound(x1, x2, x1 + x2, t1, t2, 0.025, false);
            assert_eq!(
                upper,
                f64::INFINITY,
                "x = [{x1}, {x2}], T = [{t1:?}, {t2:?}]"
            );
        }
        // With x[1] = 0 the lower bound's search starts from x[0] * T[1] / T[0], infinite here.
        let lower = find_rate_ratio_bound(5, 0, 5, 1e-300, 1e10, 0.025, true);
        assert_eq!(lower, f64::INFINITY);
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
