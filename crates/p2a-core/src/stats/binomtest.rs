//! Exact binomial test (binom.test).
//!
//! Performs an exact test of the null hypothesis that the probability of success
//! in a Bernoulli experiment equals a specified value.
//!
//! # References
//!
//! - Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial
//!   limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404-413.
//! - R Core Team. `stats::binom.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/binom.test.html>
//!
//! # Mathematical Background
//!
//! ## P-value Calculation
//!
//! For H₀: p = p₀, the exact p-value is calculated using the binomial distribution:
//!
//! ### Two-sided test
//! ```text
//! P = 2 × min(P(X ≤ x), P(X ≥ x))
//! ```
//! where X ~ Binomial(n, p₀)
//!
//! More precisely, R uses:
//! ```text
//! P = Σ P(X = k) for all k where P(X = k) ≤ P(X = x)
//! ```
//!
//! ### One-sided tests
//! ```text
//! P(greater) = P(X ≥ x) = 1 - P(X ≤ x-1)
//! P(less) = P(X ≤ x)
//! ```
//!
//! ## Confidence Interval (Clopper-Pearson)
//!
//! The exact confidence interval uses the relationship between binomial
//! and beta distributions:
//!
//! ```text
//! Lower: B_{α/2}(x, n-x+1)
//! Upper: B_{1-α/2}(x+1, n-x)
//! ```
//!
//! where B_q(a, b) is the q-th quantile of Beta(a, b).

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;

/// Result of an exact binomial test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinomTestResult {
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
    /// Number of successes
    pub successes: u64,
    /// Number of trials
    pub trials: u64,
    /// P-value (exact)
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Confidence Interval
    // ═══════════════════════════════════════════════════════════════════════
    /// Confidence level (e.g., 0.95)
    pub conf_level: f64,
    /// Lower bound of Clopper-Pearson confidence interval
    pub conf_int_lower: f64,
    /// Upper bound of Clopper-Pearson confidence interval
    pub conf_int_upper: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Estimates
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample proportion (estimate of probability of success)
    pub estimate: f64,
    /// Null hypothesis probability
    pub null_value: f64,
}

impl std::fmt::Display for BinomTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Data summary
        writeln!(
            f,
            "number of successes = {}, number of trials = {}, p-value = {:.6} {}",
            self.successes,
            self.trials,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => {
                format!("true probability of success is not equal to {}", self.null_value)
            }
            Alternative::Greater => {
                format!("true probability of success is greater than {}", self.null_value)
            }
            Alternative::Less => {
                format!("true probability of success is less than {}", self.null_value)
            }
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Confidence interval
        writeln!(f, "{:.0}% confidence interval:", self.conf_level * 100.0)?;
        writeln!(f, "  ({:.6}, {:.6})", self.conf_int_lower, self.conf_int_upper)?;
        writeln!(f)?;

        // Estimate
        writeln!(f, "Sample estimates:")?;
        writeln!(f, "  probability of success = {:.6}", self.estimate)?;
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Perform exact binomial test.
///
/// Tests the null hypothesis that the probability of success equals a specified value.
///
/// # Arguments
/// * `x` - Number of successes
/// * `n` - Number of trials
/// * `p` - Null hypothesis probability of success (default: 0.5)
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval (e.g., 0.95)
///
/// # Example
/// ```ignore
/// // Test if success probability differs from 0.5
/// let result = binom_test(15, 100, 0.5, Alternative::TwoSided, 0.95)?;
///
/// // Test if success probability exceeds 0.1
/// let result = binom_test(15, 100, 0.1, Alternative::Greater, 0.95)?;
/// ```
///
/// # References
/// - R equivalent: `binom.test(15, 100, p = 0.5)`
pub fn binom_test(
    x: u64,
    n: u64,
    p: f64,
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<BinomTestResult> {
    if n == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Number of trials must be positive".to_string(),
        });
    }
    if x > n {
        return Err(EconError::InvalidSpecification {
            message: "Number of successes cannot exceed number of trials".to_string(),
        });
    }
    if p <= 0.0 || p >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "Null probability must be between 0 and 1 (exclusive)".to_string(),
        });
    }
    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "Confidence level must be between 0 and 1".to_string(),
        });
    }

    let estimate = x as f64 / n as f64;

    // Compute exact p-value
    let p_value = exact_binom_p_value(x, n, p, alternative);

    // Compute Clopper-Pearson confidence interval
    let (ci_lower, ci_upper) = clopper_pearson_ci(x, n, conf_level, alternative);

    Ok(BinomTestResult {
        test_name: "Exact binomial test".to_string(),
        alternative,
        successes: x,
        trials: n,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        estimate,
        null_value: p,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute exact p-value for binomial test.
fn exact_binom_p_value(x: u64, n: u64, p0: f64, alternative: Alternative) -> f64 {
    use statrs::distribution::{Binomial, DiscreteCDF, Discrete};

    let binom = match Binomial::new(p0, n) {
        Ok(d) => d,
        Err(_) => return f64::NAN,
    };

    match alternative {
        Alternative::Less => {
            // P(X <= x)
            binom.cdf(x)
        }
        Alternative::Greater => {
            // P(X >= x) = 1 - P(X <= x-1)
            if x == 0 {
                1.0
            } else {
                1.0 - binom.cdf(x - 1)
            }
        }
        Alternative::TwoSided => {
            // Optimized two-sided p-value using binary search
            // Instead of iterating through all k, find tail boundaries
            two_sided_binom_p_value(&binom, x, n, p0)
        }
    }
}

/// Optimized two-sided binomial p-value calculation.
///
/// Uses the fact that binomial PMF is unimodal - we only need to find
/// where in the opposite tail the PMF drops below the observed value.
fn two_sided_binom_p_value(
    binom: &statrs::distribution::Binomial,
    x: u64,
    n: u64,
    p0: f64,
) -> f64 {
    use statrs::distribution::{DiscreteCDF, Discrete};

    let observed_prob = binom.pmf(x);
    let mean = n as f64 * p0;
    let x_f = x as f64;

    // Handle edge cases
    if x == 0 {
        // Lower tail is just P(X=0), find upper boundary
        let upper = find_upper_boundary(binom, n, observed_prob);
        return binom.cdf(0) + (1.0 - binom.cdf(upper.saturating_sub(1)));
    }
    if x == n {
        // Upper tail is just P(X=n), find lower boundary
        let lower = find_lower_boundary(binom, n, observed_prob);
        return binom.cdf(lower) + (1.0 - binom.cdf(n - 1));
    }

    if x_f < mean {
        // Observed is below mean - we have lower tail P(X <= x)
        // Find upper boundary where P(X = k) <= observed_prob
        let upper = find_upper_boundary(binom, n, observed_prob);

        // P-value = P(X <= x) + P(X >= upper)
        let p_lower = binom.cdf(x);
        let p_upper = if upper > n {
            0.0
        } else {
            1.0 - binom.cdf(upper.saturating_sub(1))
        };
        (p_lower + p_upper).min(1.0)
    } else if x_f > mean {
        // Observed is above mean - we have upper tail P(X >= x)
        // Find lower boundary where P(X = k) <= observed_prob
        let lower = find_lower_boundary(binom, n, observed_prob);

        // P-value = P(X <= lower) + P(X >= x)
        let p_lower = binom.cdf(lower);
        let p_upper = 1.0 - binom.cdf(x.saturating_sub(1));
        (p_lower + p_upper).min(1.0)
    } else {
        // x == mean (rare for discrete), p-value = 1
        1.0
    }
}

/// Binary search to find the smallest k >= mean where P(X=k) <= threshold.
fn find_upper_boundary(
    binom: &statrs::distribution::Binomial,
    n: u64,
    threshold: f64,
) -> u64 {
    use statrs::distribution::Discrete;
    use statrs::statistics::Distribution;

    // The mode is around n*p, search from there to n
    let mean = binom.mean().unwrap_or(n as f64 / 2.0) as u64;

    // Linear search from mean upward (PMF decreases monotonically after mode)
    // This is typically O(sqrt(n)) iterations
    for k in mean.max(1)..=n {
        if binom.pmf(k) <= threshold * (1.0 + 1e-10) {
            return k;
        }
    }
    n + 1 // No boundary found
}

/// Binary search to find the largest k <= mean where P(X=k) <= threshold.
fn find_lower_boundary(
    binom: &statrs::distribution::Binomial,
    n: u64,
    threshold: f64,
) -> u64 {
    use statrs::distribution::Discrete;
    use statrs::statistics::Distribution;

    let mean = binom.mean().unwrap_or(n as f64 / 2.0) as u64;

    // Linear search from mean downward (PMF decreases monotonically before mode)
    for k in (0..=mean.min(n)).rev() {
        if binom.pmf(k) <= threshold * (1.0 + 1e-10) {
            return k;
        }
    }
    0 // Return 0 if no boundary found (include nothing from lower tail)
}

/// Compute Clopper-Pearson exact confidence interval.
fn clopper_pearson_ci(x: u64, n: u64, conf_level: f64, alternative: Alternative) -> (f64, f64) {
    use statrs::distribution::{Beta, ContinuousCDF};

    let alpha = 1.0 - conf_level;
    let x_f = x as f64;
    let n_f = n as f64;

    match alternative {
        Alternative::TwoSided => {
            // Two-sided interval
            let lower = if x == 0 {
                0.0
            } else {
                let beta_lower = Beta::new(x_f, n_f - x_f + 1.0).unwrap();
                beta_lower.inverse_cdf(alpha / 2.0)
            };

            let upper = if x == n {
                1.0
            } else {
                let beta_upper = Beta::new(x_f + 1.0, n_f - x_f).unwrap();
                beta_upper.inverse_cdf(1.0 - alpha / 2.0)
            };

            (lower, upper)
        }
        Alternative::Greater => {
            // One-sided lower bound
            let lower = if x == 0 {
                0.0
            } else {
                let beta_lower = Beta::new(x_f, n_f - x_f + 1.0).unwrap();
                beta_lower.inverse_cdf(alpha)
            };

            (lower, 1.0)
        }
        Alternative::Less => {
            // One-sided upper bound
            let upper = if x == n {
                1.0
            } else {
                let beta_upper = Beta::new(x_f + 1.0, n_f - x_f).unwrap();
                beta_upper.inverse_cdf(1.0 - alpha)
            };

            (0.0, upper)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binom_test_basic() {
        let result = binom_test(15, 100, 0.5, Alternative::TwoSided, 0.95).unwrap();

        assert_eq!(result.successes, 15);
        assert_eq!(result.trials, 100);
        assert!((result.estimate - 0.15).abs() < 0.0001);
        assert!(result.p_value < 0.001); // Very significant difference from 0.5
    }

    #[test]
    fn test_binom_test_alternatives() {
        let x = 15;
        let n = 100;
        let p = 0.1;

        let result_two = binom_test(x, n, p, Alternative::TwoSided, 0.95).unwrap();
        let result_greater = binom_test(x, n, p, Alternative::Greater, 0.95).unwrap();
        let result_less = binom_test(x, n, p, Alternative::Less, 0.95).unwrap();

        // Greater should give smaller p-value than two-sided (since estimate > null)
        assert!(result_greater.p_value < result_two.p_value);

        // Less should give large p-value
        assert!(result_less.p_value > 0.9);
    }

    #[test]
    fn test_binom_test_invalid_inputs() {
        // x > n
        assert!(binom_test(101, 100, 0.5, Alternative::TwoSided, 0.95).is_err());

        // n = 0
        assert!(binom_test(0, 0, 0.5, Alternative::TwoSided, 0.95).is_err());

        // p out of range
        assert!(binom_test(15, 100, 0.0, Alternative::TwoSided, 0.95).is_err());
        assert!(binom_test(15, 100, 1.0, Alternative::TwoSided, 0.95).is_err());
    }

    #[test]
    fn test_binom_test_edge_cases() {
        // All successes
        let result = binom_test(100, 100, 0.5, Alternative::TwoSided, 0.95).unwrap();
        assert!((result.estimate - 1.0).abs() < 0.0001);
        assert!(result.conf_int_upper == 1.0);

        // No successes
        let result = binom_test(0, 100, 0.5, Alternative::TwoSided, 0.95).unwrap();
        assert!((result.estimate - 0.0).abs() < 0.0001);
        assert!(result.conf_int_lower == 0.0);
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    #[test]
    fn test_validate_binom_test_two_sided() {
        // R: binom.test(15, 100, p = 0.1)
        // p-value = 0.09628
        // 95% CI: (0.08645439, 0.23530750)

        let result = binom_test(15, 100, 0.1, Alternative::TwoSided, 0.95).unwrap();

        assert!(
            (result.p_value - 0.09628).abs() < 0.001,
            "p-value mismatch: Rust={}, R=0.09628",
            result.p_value
        );
        assert!(
            (result.conf_int_lower - 0.08645439).abs() < 0.001,
            "CI lower mismatch: Rust={}, R=0.08645439",
            result.conf_int_lower
        );
        assert!(
            (result.conf_int_upper - 0.23530750).abs() < 0.001,
            "CI upper mismatch: Rust={}, R=0.23530750",
            result.conf_int_upper
        );
    }

    #[test]
    fn test_validate_binom_test_greater() {
        // R: binom.test(15, 100, p = 0.1, alternative = "greater")
        // p-value = 0.07257
        // 95% CI: (0.09479401, 1.00000000)

        let result = binom_test(15, 100, 0.1, Alternative::Greater, 0.95).unwrap();

        assert!(
            (result.p_value - 0.07257).abs() < 0.001,
            "p-value mismatch: Rust={}, R=0.07257",
            result.p_value
        );
        assert!(
            (result.conf_int_lower - 0.09479401).abs() < 0.001,
            "CI lower mismatch: Rust={}, R=0.09479401",
            result.conf_int_lower
        );
        assert!(result.conf_int_upper == 1.0);
    }

    #[test]
    fn test_validate_binom_test_less() {
        // R: binom.test(15, 100, p = 0.1, alternative = "less")
        // p-value = 0.9601
        // 95% CI: (0.0000000, 0.2215369)

        let result = binom_test(15, 100, 0.1, Alternative::Less, 0.95).unwrap();

        assert!(
            (result.p_value - 0.9601).abs() < 0.001,
            "p-value mismatch: Rust={}, R=0.9601",
            result.p_value
        );
        assert!(result.conf_int_lower == 0.0);
        assert!(
            (result.conf_int_upper - 0.2215369).abs() < 0.001,
            "CI upper mismatch: Rust={}, R=0.2215369",
            result.conf_int_upper
        );
    }
}
