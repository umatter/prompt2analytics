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
/// still takes seconds. The two-sample test has no such loop (statrs caps its incomplete beta
/// iterations); its counts only need a total that fits in `u64`.
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
/// `EconError::InvalidSpecification` when `x` does not have length 1 or 2 or `t` a different
/// length; when a time base in `t` or `r` is not finite and > 0, or `r * T` overflows (one-sample);
/// when `conf_level` is not strictly between 0 and 1; when a one-sample count is above
/// [`MAX_ONE_SAMPLE_COUNT`]; and when the two counts' total does not fit in `u64`.
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
    if !expected.is_finite() {
        // statrs accepts an infinite Poisson mean and then panics in its CDF.
        return Err(EconError::InvalidSpecification {
            message: format!("the expected count r * T = {r} * {t} overflows f64"),
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
    let p = r * t1 / (r * t1 + t2);

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
            message: format!("x = {x}: x + 1 (the upper bound's shape) overflows a 64-bit count"),
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

    // Expand bounds if needed to bracket the solution
    if find_lower {
        // For lower bound: need compute_tail_p(lo) < alpha < compute_tail_p(hi)
        while compute_tail_p(lo) > alpha && lo > 1e-15 {
            lo /= 10.0;
        }
        while compute_tail_p(hi) < alpha && hi < rate_ratio_est * 10.0 {
            hi *= 2.0;
        }
    } else {
        // For upper bound: need compute_tail_p(hi) < alpha < compute_tail_p(lo)
        while compute_tail_p(lo) < alpha && lo > 1e-10 {
            lo /= 2.0;
        }
        while compute_tail_p(hi) > alpha && hi < 1e10 {
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

    /// The total may be u64::MAX itself; the two-sample path has no loop that depends on the counts
    /// (bisection and bracketing are capped), so it returns.
    #[test]
    fn test_poisson_two_sample_accepts_largest_total() {
        for x in [[u64::MAX - 1, 1], [u64::MAX, 0], [0, u64::MAX]] {
            let result = poisson_test(&x, &[1.0, 2.5], 1.0, PoissonAlternative::TwoSided, 0.95);
            assert!(result.is_ok(), "x = {x:?}: {result:?}");
        }
    }

    /// FU-1: a one-sample count above 2^52 is refused before statrs is called. Above about 2^53
    /// the incomplete gamma series behind the Poisson CDF and the chi-squared quantiles stops
    /// advancing and never returns (u64::MAX, u64::MAX - 1 and 2^53 - 2 all hung); u64::MAX would
    /// also overflow x + 1 in the interval.
    #[test]
    fn test_poisson_one_sample_refuses_counts_above_bound() {
        for x in [
            MAX_ONE_SAMPLE_COUNT + 1,
            (1 << 53) - 2,
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
        assert_eq!(MAX_ONE_SAMPLE_COUNT, 4_503_599_627_370_496);
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
