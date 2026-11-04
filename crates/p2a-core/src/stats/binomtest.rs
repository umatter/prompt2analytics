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
use crate::traits::estimator::validate_level;

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
                format!(
                    "true probability of success is not equal to {}",
                    self.null_value
                )
            }
            Alternative::Greater => {
                format!(
                    "true probability of success is greater than {}",
                    self.null_value
                )
            }
            Alternative::Less => {
                format!(
                    "true probability of success is less than {}",
                    self.null_value
                )
            }
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Confidence interval
        writeln!(f, "{:.0}% confidence interval:", self.conf_level * 100.0)?;
        writeln!(
            f,
            "  ({:.6}, {:.6})",
            self.conf_int_lower, self.conf_int_upper
        )?;
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
    validate_level(p, "p")?;
    validate_level(conf_level, "conf_level")?;

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
    use statrs::distribution::{Binomial, DiscreteCDF};

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
            // P(X >= x) = P(X > x - 1), the survival function (no 1 - cdf cancellation)
            if x == 0 { 1.0 } else { binom.sf(x - 1) }
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
fn two_sided_binom_p_value(binom: &statrs::distribution::Binomial, x: u64, n: u64, p0: f64) -> f64 {
    use statrs::distribution::{Discrete, DiscreteCDF};

    let observed_prob = binom.pmf(x);
    let mean = n as f64 * p0;
    let x_f = x as f64;

    // Handle edge cases
    if x == 0 {
        // Lower tail is just P(X=0), find upper boundary
        let upper = find_upper_boundary(binom, n, observed_prob);
        return binom.cdf(0) + binom.sf(upper.saturating_sub(1));
    }
    if x == n {
        // Upper tail is just P(X=n), find lower boundary
        let lower = find_lower_boundary(binom, n, observed_prob);
        return binom.cdf(lower) + binom.sf(n - 1);
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
            binom.sf(upper.saturating_sub(1))
        };
        (p_lower + p_upper).min(1.0)
    } else if x_f > mean {
        // Observed is above mean - we have upper tail P(X >= x)
        // Find lower boundary where P(X = k) <= observed_prob
        let lower = find_lower_boundary(binom, n, observed_prob);

        // P-value = P(X <= lower) + P(X >= x)
        let p_lower = binom.cdf(lower);
        let p_upper = binom.sf(x.saturating_sub(1));
        (p_lower + p_upper).min(1.0)
    } else {
        // x == mean (rare for discrete), p-value = 1
        1.0
    }
}

/// Binary search to find the smallest k >= mean where P(X=k) <= threshold.
fn find_upper_boundary(binom: &statrs::distribution::Binomial, n: u64, threshold: f64) -> u64 {
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
fn find_lower_boundary(binom: &statrs::distribution::Binomial, n: u64, threshold: f64) -> u64 {
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

    #[test]
    fn test_validate_binom_grid() {
        use crate::data::columns::{ColumnRequest, NaPolicy};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_binom");
        let cols = ColumnRequest::new(ds.df())
            .factor("label")
            .numerics(&["x", "n", "p", "stat15"])
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        let (mut all, mut ok) = (Vec::new(), Vec::new());
        let mut v: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
        for i in 0..cols.n() {
            let label = cols.factor("label").label(i).to_string();
            let (x, n, p) = (
                cols.numeric("x")[i] as u64,
                cols.numeric("n")[i] as u64,
                cols.numeric("p")[i],
            );
            let two = binom_test(x, n, p, Alternative::TwoSided, 0.95).unwrap();
            let less = binom_test(x, n, p, Alternative::Less, 0.95).unwrap();
            let greater = binom_test(x, n, p, Alternative::Greater, 0.95).unwrap();
            if cols.numeric("stat15")[i] == 0.0 {
                ok.push(label.clone());
                v.entry("p_two").or_default().push(two.p_value);
            } // deviation STAT-15: the stat15 cells keep R's two-sided p under p_two_stat15
            all.push(label);
            for (k, val) in [
                ("p_less", less.p_value),
                ("p_greater", greater.p_value),
                ("ci_two_lower", two.conf_int_lower),
                ("ci_two_upper", two.conf_int_upper),
                ("ci_greater_lower", greater.conf_int_lower),
                ("ci_less_upper", less.conf_int_upper),
                ("estimate", two.estimate),
            ] {
                v.entry(k).or_default().push(val);
            }
        }
        // the smallest n in the grid is 10: README tolerances at n = 10
        Check::new(&case)
            .named("p_two", &ok, &v["p_two"], Tol::p_value(10))
            .named("p_less", &all, &v["p_less"], Tol::p_value(10))
            .named("p_greater", &all, &v["p_greater"], Tol::p_value(10))
            .named("ci_two_lower", &all, &v["ci_two_lower"], Tol::stat(10))
            .named("ci_two_upper", &all, &v["ci_two_upper"], Tol::stat(10))
            .named(
                "ci_greater_lower",
                &all,
                &v["ci_greater_lower"],
                Tol::stat(10),
            )
            .named("ci_less_upper", &all, &v["ci_less_upper"], Tol::stat(10))
            .named("estimate", &all, &v["estimate"], Tol::coef(10))
            .finish();
    }

    #[test]
    fn test_validate_binom_upper_tails() {
        // R6: P(X >= x) as a survival function. r13 = (95, 100, 0.5): before, 1 - cdf(94) = 0
        // (greater) and the two-sided p was half of R's; r12 = (10, 10, 0.05): 1 - cdf(9) =
        // 9.7699626167013776e-14 (R 9.765625e-14). Tol::p_value's abs 1e-12 hides these, so
        // this test compares relative accuracy only.
        use p2a_validation::{Check, Tol};
        let (case, _ds) = crate::test_support::load_case("stat_rank_binom");
        let g13 = binom_test(95, 100, 0.5, Alternative::Greater, 0.95)
            .unwrap()
            .p_value;
        let t13 = binom_test(95, 100, 0.5, Alternative::TwoSided, 0.95)
            .unwrap()
            .p_value;
        let g12 = binom_test(10, 10, 0.05, Alternative::Greater, 0.95)
            .unwrap()
            .p_value;
        // tighter than the README row on purpose: relative accuracy of tiny tails
        let tail = Tol::new(1e-10, 0.0);
        Check::new(&case)
            .named_subset(
                "p_greater",
                &["r12", "r13"],
                &["r12", "r13"],
                &[g12, g13],
                tail,
            )
            .named_subset("p_two", &["r13"], &["r13"], &[t13], tail)
            .finish();
    }

    #[test]
    fn test_binom_test_rejects_invalid_level_and_p() {
        // R: "'conf.level' must be a single number between 0 and 1", "'p' must be a single
        // number between 0 and 1". Before: conf_level NaN panicked in statrs' Beta quantile;
        // p NaN returned Ok with a NaN p-value.
        for level in [f64::NAN, 1.0, -0.1] {
            match binom_test(3, 10, 0.5, Alternative::TwoSided, level) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("conf_level"), "{message}")
                }
                other => panic!("{level}: expected InvalidSpecification, got {other:?}"),
            }
        }
        for p in [f64::NAN, 0.0, 1.0] {
            match binom_test(3, 10, p, Alternative::TwoSided, 0.95) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(
                        message.contains("p must be strictly between 0 and 1"),
                        "{message}"
                    )
                }
                other => panic!("{p}: expected InvalidSpecification, got {other:?}"),
            }
        }
    }
}
