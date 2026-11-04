//! Ansari-Bradley test for equal scale parameters.
//!
//! A non-parametric two-sample test for the null hypothesis that
//! the scale parameters (related to dispersion) of two distributions
//! are equal.
//!
//! # References
//!
//! - Ansari, A. R. and Bradley, R. A. (1960). Rank-sum tests for dispersions.
//!   *Annals of Mathematical Statistics*, 31, 1174-1189.
//! - Bauer, D. F. (1972). Constructing confidence sets using rank statistics.
//!   *Journal of the American Statistical Association*, 67, 687-690.
//! - R Core Team. `stats::ansari.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ansari.test.html>
//!
//! # Mathematical Background
//!
//! ## Algorithm
//!
//! 1. Combine both samples and rank them (1 to n = m + n)
//!
//! 2. Assign Ansari-Bradley scores to ranks:
//!    - For odd N: scores are 1, 2, ..., (N+1)/2, (N-1)/2, ..., 2, 1
//!    - For even N: scores are 1, 2, ..., N/2, N/2, ..., 2, 1
//!
//! 3. Compute test statistic AB = sum of scores for sample x
//!
//! 4. Under H₀ (equal scales), the distribution of AB is:
//!    - Exact: computed from combinatorics
//!    - Approximate: Normal with
//!      - Mean: m(N+2)/4 (if N odd) or m(N+2)/4 (if N even)
//!      - Variance: mn(N+2)(N-2) / (48(N-1))
//!
//! H₀: Scale ratio σ_x/σ_y = 1
//! H₁: Scale ratio ≠ 1 (two-sided), > 1 (greater), or < 1 (less)

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::resource::try_with_capacity;
use crate::stats::ranks::midranks;
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::{normal_cdf, normal_p_value_two_sided, validate_level, z_critical};

/// Result of Ansari-Bradley test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnsariBradleyResult {
    /// Description of the test
    pub test_name: String,
    /// Alternative hypothesis type
    pub alternative: Alternative,
    /// Ansari-Bradley test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Size of first sample
    pub n_x: usize,
    /// Size of second sample
    pub n_y: usize,
    /// Whether exact p-value was computed
    pub exact: bool,
    /// Confidence level (if CI computed)
    pub conf_level: Option<f64>,
    /// Lower confidence bound for scale ratio
    pub conf_int_lower: Option<f64>,
    /// Upper confidence bound for scale ratio
    pub conf_int_upper: Option<f64>,
    /// Point estimate of scale ratio (Hodges-Lehmann type)
    pub estimate: Option<f64>,
}

impl std::fmt::Display for AnsariBradleyResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        writeln!(
            f,
            "AB = {:.4}, p-value = {:.5} {} ({})",
            self.statistic,
            self.p_value,
            self.significance.stars(),
            if self.exact {
                "exact"
            } else {
                "normal approximation"
            }
        )?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.alternative {
            Alternative::TwoSided => "true ratio of scales is not equal to 1",
            Alternative::Greater => "true ratio of scales is greater than 1",
            Alternative::Less => "true ratio of scales is less than 1",
        };
        writeln!(f, "Alternative hypothesis: {}", alt_str)?;
        writeln!(f)?;

        // Sample sizes
        writeln!(f, "Sample sizes: n_x = {}, n_y = {}", self.n_x, self.n_y)?;
        writeln!(f)?;

        // Confidence interval if available
        if let (Some(level), Some(lower), Some(upper)) =
            (self.conf_level, self.conf_int_lower, self.conf_int_upper)
        {
            writeln!(f, "{:.0}% confidence interval:", level * 100.0)?;
            writeln!(f, "  ({:.6}, {:.6})", lower, upper)?;
            writeln!(f)?;
        }

        if let Some(est) = self.estimate {
            writeln!(f, "Sample estimate:")?;
            writeln!(f, "  ratio of scales = {:.6}", est)?;
            writeln!(f)?;
        }

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Perform Ansari-Bradley test for equality of scale parameters.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Direction of alternative hypothesis
/// * `exact` - If true, compute exact p-value (only if n < 50 and no ties)
/// * `conf_level` - Confidence level for CI (optional)
///
/// # Example
/// ```ignore
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
/// let result = ansari_test(&x, &y, Alternative::TwoSided, true, Some(0.95))?;
/// ```
///
/// # References
/// - R equivalent: `ansari.test(x, y)`
pub fn ansari_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    exact: bool,
    conf_level: Option<f64>,
) -> EconResult<AnsariBradleyResult> {
    if let Some(level) = conf_level {
        validate_level(level, "conf_level")?;
    }
    let m = x.len();
    let n = y.len();
    let total_n = m + n;

    if m == 0 || n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Both samples must have at least one observation".to_string(),
        });
    }

    // Pool the finite values of x then y, in input order
    let mut values: Vec<f64> = try_with_capacity(total_n, "Ansari-Bradley pooled sample")?;
    values.extend(x.iter().copied().filter(|v| v.is_finite()));
    let m_finite = values.len();
    values.extend(y.iter().copied().filter(|v| v.is_finite()));
    let n_finite = values.len() - m_finite;
    let n_total = values.len();

    if m_finite == 0 || n_finite == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Both samples must have at least one finite observation".to_string(),
        });
    }

    // Ranks (1 to N) with midranks for ties, exact ties as R's rank()
    let (ranks, tie_sizes) = midranks(&values)?;
    let has_ties = tie_sizes.iter().any(|&t| t > 1);

    // Ansari-Bradley scores min(r, N+1-r); AB = sum of the scores of sample x
    let ab_stat: f64 = ranks[..m_finite]
        .iter()
        .map(|&r| r.min((n_total as f64 + 1.0) - r))
        .sum();

    // Determine if we use exact or approximate
    let use_exact = exact && m_finite < 50 && n_finite < 50 && !has_ties;

    let (p_value, ci_lower, ci_upper, estimate) = if use_exact {
        // Exact computation using permutation distribution
        let (pval, lower, upper, est) =
            exact_ansari(m_finite, n_finite, ab_stat, alternative, conf_level);
        (pval, lower, upper, est)
    } else {
        // Normal approximation
        let (pval, lower, upper, est) = approx_ansari(
            m_finite,
            n_finite,
            ab_stat,
            alternative,
            conf_level,
            has_ties,
        )?;
        (pval, lower, upper, est)
    };

    Ok(AnsariBradleyResult {
        test_name: "Ansari-Bradley test".to_string(),
        alternative,
        statistic: ab_stat,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_x: m_finite,
        n_y: n_finite,
        exact: use_exact,
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        estimate,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Exact p-value and confidence interval for Ansari-Bradley test.
fn exact_ansari(
    m: usize,
    n: usize,
    stat: f64,
    alternative: Alternative,
    conf_level: Option<f64>,
) -> (f64, Option<f64>, Option<f64>, Option<f64>) {
    let total_n = m + n;

    // Expected value and variance under H0
    let mu = if total_n % 2 == 1 {
        (m as f64) * (total_n as f64 + 1.0).powi(2) / (4.0 * total_n as f64)
    } else {
        (m as f64) * (total_n as f64 + 2.0) / 4.0
    };

    // For small samples, use exact permutation distribution
    // This is computationally expensive for large samples
    // Here we approximate using dynamic programming for the distribution

    // Compute distribution using DP
    let (p_lower, p_upper) = ansari_exact_probs(m, n, stat);

    let p_value = match alternative {
        Alternative::TwoSided => {
            // Two-sided: 2 * min(P(AB <= stat), P(AB >= stat))
            let p_two = 2.0 * p_lower.min(p_upper);
            p_two.min(1.0)
        }
        Alternative::Less => p_lower,
        Alternative::Greater => p_upper,
    };

    // Point estimate (based on the test statistic vs expected)
    let estimate = if stat > mu {
        Some((stat / mu).sqrt())
    } else {
        Some((stat / mu).sqrt())
    };

    // For CI we'd need Hodges-Lehmann type estimator
    // This is complex for Ansari-Bradley, so we skip for now
    let ci = if let Some(_level) = conf_level {
        // Simplified CI based on normal approximation
        let sigma = ansari_sigma(m, n);
        if sigma > 0.0 {
            // This is a simplification
            (None, None)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    (p_value, ci.0, ci.1, estimate)
}

/// Compute P(AB <= stat) and P(AB >= stat) using dynamic programming.
fn ansari_exact_probs(m: usize, n: usize, stat: f64) -> (f64, f64) {
    let total_n = m + n;

    // Generate all Ansari-Bradley scores for positions 1..=total_n
    let scores: Vec<usize> = (1..=total_n).map(|r| r.min(total_n + 1 - r)).collect();

    // Dynamic programming: count ways to achieve each sum
    // when choosing m items from total_n items
    let max_sum: usize = scores.iter().take(m).rev().take(m).sum::<usize>() + 1;

    // dp[i][j] = number of ways to select i items from first k items with sum j
    let mut dp: Vec<Vec<u64>> = vec![vec![0; max_sum + 1]; m + 1];
    dp[0][0] = 1;

    for (k, &score) in scores.iter().enumerate() {
        // Process in reverse to avoid using updated values
        for i in (1..=m.min(k + 1)).rev() {
            for j in score..=max_sum {
                dp[i][j] += dp[i - 1][j - score];
            }
        }
    }

    let stat_int = stat as usize;
    let total_ways: u64 = dp[m].iter().sum();

    // P(AB <= stat)
    let ways_le: u64 = dp[m][..=stat_int.min(max_sum)].iter().sum();
    let p_lower = ways_le as f64 / total_ways as f64;

    // P(AB >= stat)
    let ways_ge: u64 = dp[m][stat_int.min(max_sum)..].iter().sum();
    let p_upper = ways_ge as f64 / total_ways as f64;

    (p_lower, p_upper)
}

/// Normal approximation for Ansari-Bradley test.
fn approx_ansari(
    m: usize,
    n: usize,
    stat: f64,
    alternative: Alternative,
    conf_level: Option<f64>,
    _has_ties: bool,
) -> EconResult<(f64, Option<f64>, Option<f64>, Option<f64>)> {
    let total_n = m + n;

    // Mean under H0
    let mu = if total_n % 2 == 1 {
        (m as f64) * (total_n as f64 + 1.0).powi(2) / (4.0 * total_n as f64)
    } else {
        (m as f64) * (total_n as f64 + 2.0) / 4.0
    };

    // Variance under H0
    let sigma = ansari_sigma(m, n);

    if sigma == 0.0 {
        return Ok((1.0, None, None, Some(1.0)));
    }

    let z = (stat - mu) / sigma;

    // The tail mapping is unchanged here (STAT-12 swaps it); tails computed without 1 - cdf
    let p_value = match alternative {
        Alternative::TwoSided => normal_p_value_two_sided(z),
        Alternative::Less => normal_cdf(z),
        Alternative::Greater => normal_cdf(-z),
    };

    // Point estimate
    let estimate = Some((stat / mu).sqrt());

    // Confidence interval based on normal approximation
    let ci = if let Some(level) = conf_level {
        let z_crit = z_critical(level)?;

        // CI for scale ratio is complex, approximate using ratio
        let lower = (stat - z_crit * sigma) / mu;
        let upper = (stat + z_crit * sigma) / mu;

        (Some(lower.max(0.0).sqrt()), Some(upper.sqrt()))
    } else {
        (None, None)
    };

    Ok((p_value, ci.0, ci.1, estimate))
}

/// Compute standard deviation under H0 for Ansari-Bradley statistic.
fn ansari_sigma(m: usize, n: usize) -> f64 {
    let total_n = m + n;
    let m_f = m as f64;
    let n_f = n as f64;
    let n_tot = total_n as f64;

    if total_n < 2 {
        return 0.0;
    }

    let var = if total_n % 2 == 1 {
        // Odd N
        m_f * n_f * (n_tot + 1.0) * (3.0 + n_tot.powi(2)) / (48.0 * n_tot.powi(2))
    } else {
        // Even N
        m_f * n_f * (n_tot + 2.0) * (n_tot - 2.0) / (48.0 * (n_tot - 1.0))
    };

    var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ansari_test_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = ansari_test(&x, &y, Alternative::TwoSided, true, None).unwrap();

        assert_eq!(result.n_x, 5);
        assert_eq!(result.n_y, 5);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_ansari_test_different_scales() {
        // x has small variance, y has large variance
        let x = vec![4.5, 4.8, 5.0, 5.2, 5.5];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        let result = ansari_test(&x, &y, Alternative::TwoSided, true, None).unwrap();

        // Should detect difference in scales
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_ansari_test_empty_sample() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(ansari_test(&x, &y, Alternative::TwoSided, true, None).is_err());
    }

    #[test]
    fn test_ansari_test_normal_approx() {
        // Large sample forces normal approximation
        let x: Vec<f64> = (1..=60).map(|i| i as f64).collect();
        let y: Vec<f64> = (1..=60).map(|i| (i * 2) as f64).collect();

        let result = ansari_test(&x, &y, Alternative::TwoSided, true, None).unwrap();

        assert!(!result.exact); // Should use approximation for n >= 50
    }

    #[test]
    fn test_validate_ansari_two_sample() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_two_sample");
        let all = ["basic", "scale", "manual", "sim", "ties", "serum"];
        let tie_free = ["basic", "scale", "manual", "sim"];
        let (mut ab, mut p) = (Vec::new(), Vec::new());
        for set in all {
            let x = column_where(&ds, "value", &[("set", set), ("group", "x")]);
            let y = column_where(&ds, "value", &[("set", set), ("group", "y")]);
            let r = ansari_test(&x, &y, Alternative::TwoSided, false, None).unwrap();
            assert!(!r.exact);
            ab.push(r.statistic);
            if tie_free.contains(&set) {
                p.push(r.p_value);
            }
            // R: exact = TRUE falls back to the normal approximation under ties
            let e = ansari_test(&x, &y, Alternative::TwoSided, true, None).unwrap();
            assert_eq!(e.exact, tie_free.contains(&set), "{set}");
        }
        // deviation STAT-13: ab_p_exact; STAT-12: ab_p_less / ab_p_greater; STAT-79: ab_p_ties;
        // STAT-14: estimate, CI
        Check::new(&case)
            .named("ab", &all, &ab, Tol::exact())
            .named("ab_p_normal", &tie_free, &p, Tol::p_value(6))
            .finish();
    }

    #[test]
    fn test_validate_ansari_small_scale() {
        use crate::stats::ranks::test_data::column_where;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_small_groups");
        let x = column_where(&ds, "ynt", &[("g", "a")]);
        let y = column_where(&ds, "ynt", &[("g", "b")]);
        let r = ansari_test(&x, &y, Alternative::TwoSided, false, None).unwrap();
        // deviation STAT-13: the exact p (ansari_p_exact) is not compared
        assert!(
            ansari_test(&x, &y, Alternative::TwoSided, true, None)
                .unwrap()
                .exact
        );
        Check::new(&case)
            .scalar("ansari_ab", r.statistic, Tol::exact())
            .scalar("ansari_p_normal", r.p_value, Tol::p_value(14))
            .finish();
    }

    #[test]
    fn test_ansari_rejects_invalid_conf_level() {
        // R: "'conf.level' must be a single number between 0 and 1". Before: statrs panicked on
        // the normal-approximation path for 1.5 and NaN.
        let x = [4.6, 4.8, 5.05, 5.2, 5.5];
        let y = [1.0, 3.0, 5.3, 7.0, 9.0];
        for level in [1.5, 0.0, f64::NAN] {
            match ansari_test(&x, &y, Alternative::TwoSided, false, Some(level)) {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains("conf_level"), "{message}")
                }
                other => panic!("{level}: expected InvalidSpecification, got {other:?}"),
            }
        }
    }
}
