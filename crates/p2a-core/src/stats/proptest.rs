//! Test of proportions (prop.test).
//!
//! Tests the null hypothesis that proportions in one or more groups are equal
//! to specified values (one-sample) or equal to each other (k-sample).
//!
//! # References
//!
//! - Newcombe, R. G. (1998). Two-sided confidence intervals for the single
//!   proportion: comparison of seven methods. *Statistics in Medicine*, 17(8), 857-872.
//! - Wilson, E. B. (1927). Probable inference, the law of succession, and
//!   statistical inference. *Journal of the American Statistical Association*, 22(158), 209-212.
//! - R Core Team. `stats::prop.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prop.test.html>
//!
//! # Mathematical Background
//!
//! ## One-Sample Test
//!
//! Tests H₀: p = p₀ using Pearson's chi-squared statistic:
//!
//! ```text
//! χ² = (|x - n·p₀| - c)² / (n·p₀·(1-p₀))
//! ```
//!
//! where c = 0.5 for Yates' continuity correction, 0 otherwise.
//!
//! ## Two-Sample Test
//!
//! Tests H₀: p₁ = p₂ using:
//!
//! ```text
//! p̂ = (x₁ + x₂) / (n₁ + n₂)  (pooled proportion)
//! χ² = (|p̂₁ - p̂₂| - c)² / (p̂·(1-p̂)·(1/n₁ + 1/n₂))
//! ```
//!
//! where c = (1/n₁ + 1/n₂)/2 for Yates' correction, 0 otherwise.
//!
//! ## Confidence Interval
//!
//! For one sample, uses Wilson's score interval:
//!
//! ```text
//! CI = (p̂ + z²/(2n) ± z·√(p̂(1-p̂)/n + z²/(4n²))) / (1 + z²/n)
//! ```
//!
//! For two samples, the CI is for the difference p₁ - p₂.

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::{chi_squared_p_value, normal_quantile, validate_level, z_critical};

/// Result of a proportion test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropTestResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Test Type
    // ═══════════════════════════════════════════════════════════════════════
    /// Description of the test performed
    pub test_name: String,
    /// Alternative hypothesis type
    pub alternative: Alternative,
    /// Whether continuity correction was applied
    pub correct: bool,

    // ═══════════════════════════════════════════════════════════════════════
    // Test Statistics
    // ═══════════════════════════════════════════════════════════════════════
    /// Chi-squared statistic
    pub chi_squared: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,

    // ═══════════════════════════════════════════════════════════════════════
    // Confidence Interval
    // ═══════════════════════════════════════════════════════════════════════
    /// Confidence level (e.g., 0.95)
    pub conf_level: f64,
    /// Lower bound of confidence interval
    pub conf_int_lower: f64,
    /// Upper bound of confidence interval
    pub conf_int_upper: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Estimates
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample proportions
    pub estimates: Vec<f64>,
    /// Null hypothesis proportions (for one-sample test)
    pub null_proportions: Option<Vec<f64>>,
    /// Number of successes
    pub successes: Vec<u64>,
    /// Number of trials
    pub trials: Vec<u64>,
}

impl std::fmt::Display for PropTestResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Test statistic
        writeln!(
            f,
            "X-squared = {:.4}, df = {}, p-value = {:.6} {}",
            self.chi_squared,
            self.df,
            self.p_value,
            self.significance.stars()
        )?;
        writeln!(f)?;

        // Alternative hypothesis
        let alt_str = match self.estimates.len() {
            1 => match self.alternative {
                Alternative::TwoSided => format!(
                    "true p is not equal to {}",
                    self.null_proportions.as_ref().map_or(0.5, |v| v[0])
                ),
                Alternative::Greater => format!(
                    "true p is greater than {}",
                    self.null_proportions.as_ref().map_or(0.5, |v| v[0])
                ),
                Alternative::Less => format!(
                    "true p is less than {}",
                    self.null_proportions.as_ref().map_or(0.5, |v| v[0])
                ),
            },
            _ => "two.sided".to_string(),
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

        // Estimates
        writeln!(f, "Sample estimates:")?;
        for (i, &p) in self.estimates.iter().enumerate() {
            if self.estimates.len() == 1 {
                writeln!(f, "     p = {:.6}", p)?;
            } else {
                writeln!(f, "  prop {} = {:.6}", i + 1, p)?;
            }
        }
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Test one proportion against a specified value.
///
/// # Arguments
/// * `x` - Number of successes
/// * `n` - Number of trials
/// * `p` - Null hypothesis proportion (default: 0.5)
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval (e.g., 0.95)
/// * `correct` - Whether to apply Yates' continuity correction
///
/// # Example
/// ```ignore
/// // Test if proportion differs from 0.5
/// let result = prop_test_one(15, 100, 0.5, Alternative::TwoSided, 0.95, true)?;
///
/// // Test if proportion exceeds 0.1
/// let result = prop_test_one(15, 100, 0.1, Alternative::Greater, 0.95, true)?;
/// ```
///
/// # References
/// - R equivalent: `prop.test(15, 100, p = 0.5)`
pub fn prop_test_one(
    x: u64,
    n: u64,
    p: f64,
    alternative: Alternative,
    conf_level: f64,
    correct: bool,
) -> EconResult<PropTestResult> {
    validate_level(conf_level, "conf_level")?;
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
            message: "Null proportion must be between 0 and 1 (exclusive)".to_string(),
        });
    }
    let n_f = n as f64;
    let x_f = x as f64;
    let p_hat = x_f / n_f;

    // Compute chi-squared statistic with optional Yates correction
    let diff = (p_hat - p).abs();
    let correction = if correct {
        // Continuity correction: min(0.5/n, |p_hat - p|)
        (0.5 / n_f).min(diff)
    } else {
        0.0
    };

    let numerator = ((diff - correction).max(0.0)).powi(2);
    let denominator = p * (1.0 - p) / n_f;
    let chi_sq = numerator / denominator;

    // Compute p-value
    let p_value = compute_prop_p_value(chi_sq, 1, alternative, p_hat, p);

    // Compute confidence interval using Wilson score method
    let (ci_lower, ci_upper) = wilson_ci(x, n, conf_level, alternative)?;

    let correction_str = if correct {
        "with continuity correction"
    } else {
        "without continuity correction"
    };

    Ok(PropTestResult {
        test_name: format!("1-sample proportions test {}", correction_str),
        alternative,
        correct,
        chi_squared: chi_sq,
        df: 1,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        estimates: vec![p_hat],
        null_proportions: Some(vec![p]),
        successes: vec![x],
        trials: vec![n],
    })
}

/// Test equality of two proportions.
///
/// # Arguments
/// * `x1`, `x2` - Number of successes in each group
/// * `n1`, `n2` - Number of trials in each group
/// * `alternative` - Direction of alternative hypothesis
/// * `conf_level` - Confidence level for interval
/// * `correct` - Whether to apply Yates' continuity correction
///
/// # Example
/// ```ignore
/// let result = prop_test_two(30, 100, 40, 100, Alternative::TwoSided, 0.95, true)?;
/// ```
///
/// # References
/// - R equivalent: `prop.test(c(30, 40), c(100, 100))`
pub fn prop_test_two(
    x1: u64,
    n1: u64,
    x2: u64,
    n2: u64,
    alternative: Alternative,
    conf_level: f64,
    correct: bool,
) -> EconResult<PropTestResult> {
    validate_level(conf_level, "conf_level")?;
    if n1 == 0 || n2 == 0 {
        return Err(EconError::InvalidSpecification {
            message: "Number of trials must be positive".to_string(),
        });
    }
    if x1 > n1 || x2 > n2 {
        return Err(EconError::InvalidSpecification {
            message: "Number of successes cannot exceed number of trials".to_string(),
        });
    }

    let n1_f = n1 as f64;
    let n2_f = n2 as f64;
    let p1 = x1 as f64 / n1_f;
    let p2 = x2 as f64 / n2_f;
    let p_pooled = (x1 + x2) as f64 / (n1 + n2) as f64;

    // Compute chi-squared statistic
    let diff = (p1 - p2).abs();
    let correction = if correct {
        // Yates correction for two-sample
        (0.5 * (1.0 / n1_f + 1.0 / n2_f)).min(diff)
    } else {
        0.0
    };

    let variance = p_pooled * (1.0 - p_pooled) * (1.0 / n1_f + 1.0 / n2_f);
    let chi_sq = if variance > 0.0 {
        ((diff - correction).max(0.0)).powi(2) / variance
    } else {
        0.0
    };

    // P-value
    let p_value = compute_prop_p_value(chi_sq, 1, alternative, p1, p2);

    // Confidence interval for difference p1 - p2
    let (ci_lower, ci_upper) = prop_diff_ci(x1, n1, x2, n2, conf_level, correct)?;

    let correction_str = if correct {
        "with continuity correction"
    } else {
        "without continuity correction"
    };

    Ok(PropTestResult {
        test_name: format!(
            "2-sample test for equality of proportions {}",
            correction_str
        ),
        alternative,
        correct,
        chi_squared: chi_sq,
        df: 1,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: ci_lower,
        conf_int_upper: ci_upper,
        estimates: vec![p1, p2],
        null_proportions: None,
        successes: vec![x1, x2],
        trials: vec![n1, n2],
    })
}

/// Test equality of multiple proportions (k-sample chi-squared test).
///
/// # Arguments
/// * `successes` - Number of successes in each group
/// * `trials` - Number of trials in each group
/// * `conf_level` - Confidence level (CI is not computed for k > 2)
///
/// # Example
/// ```ignore
/// let result = prop_test_k(&[30, 40, 35], &[100, 100, 100], 0.95)?;
/// ```
///
/// # References
/// - R equivalent: `prop.test(c(30, 40, 35), c(100, 100, 100))`
pub fn prop_test_k(
    successes: &[u64],
    trials: &[u64],
    conf_level: f64,
) -> EconResult<PropTestResult> {
    validate_level(conf_level, "conf_level")?;
    let k = successes.len();
    if k < 2 {
        return Err(EconError::InvalidSpecification {
            message: "At least two groups required for k-sample test".to_string(),
        });
    }
    if successes.len() != trials.len() {
        return Err(EconError::InvalidSpecification {
            message: "Successes and trials must have same length".to_string(),
        });
    }

    for i in 0..k {
        if trials[i] == 0 {
            return Err(EconError::InvalidSpecification {
                message: format!("Number of trials in group {} must be positive", i + 1),
            });
        }
        if successes[i] > trials[i] {
            return Err(EconError::InvalidSpecification {
                message: format!("Successes cannot exceed trials in group {}", i + 1),
            });
        }
    }

    // Pooled proportion
    let total_successes: u64 = successes.iter().sum();
    let total_trials: u64 = trials.iter().sum();
    let p_pooled = total_successes as f64 / total_trials as f64;

    // Compute chi-squared statistic
    let mut chi_sq = 0.0;
    let mut estimates = Vec::with_capacity(k);

    for i in 0..k {
        let n_i = trials[i] as f64;
        let p_i = successes[i] as f64 / n_i;
        estimates.push(p_i);

        let expected_success = n_i * p_pooled;
        let expected_failure = n_i * (1.0 - p_pooled);

        if expected_success > 0.0 {
            chi_sq += (successes[i] as f64 - expected_success).powi(2) / expected_success;
        }
        if expected_failure > 0.0 {
            chi_sq +=
                ((trials[i] - successes[i]) as f64 - expected_failure).powi(2) / expected_failure;
        }
    }

    // Degrees of freedom = k - 1
    let df = k - 1;

    // P-value from chi-squared distribution
    let p_value = chi_squared_p_value(chi_sq, df as f64);

    Ok(PropTestResult {
        test_name: format!("{}-sample test for equality of proportions", k),
        alternative: Alternative::TwoSided,
        correct: false, // No correction for k > 2
        chi_squared: chi_sq,
        df,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        conf_level,
        conf_int_lower: f64::NAN, // No CI for k > 2
        conf_int_upper: f64::NAN,
        estimates,
        null_proportions: None,
        successes: successes.to_vec(),
        trials: trials.to_vec(),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute p-value for proportion test.
fn compute_prop_p_value(
    chi_sq: f64,
    df: usize,
    alternative: Alternative,
    p_hat: f64,
    p_null: f64,
) -> f64 {
    if chi_sq.is_nan() || chi_sq < 0.0 {
        return f64::NAN;
    }

    let p_two = chi_squared_p_value(chi_sq, df as f64);
    match alternative {
        Alternative::TwoSided => p_two,
        Alternative::Greater => {
            // One-sided: divide by 2 if in expected direction
            if p_hat >= p_null {
                p_two / 2.0
            } else {
                1.0 - p_two / 2.0
            }
        }
        Alternative::Less => {
            if p_hat <= p_null {
                p_two / 2.0
            } else {
                1.0 - p_two / 2.0
            }
        }
    }
}

/// Wilson score confidence interval for a single proportion.
fn wilson_ci(x: u64, n: u64, conf_level: f64, alternative: Alternative) -> EconResult<(f64, f64)> {
    let n_f = n as f64;
    let p_hat = x as f64 / n_f;

    let z = match alternative {
        Alternative::TwoSided => z_critical(conf_level)?,
        Alternative::Greater | Alternative::Less => normal_quantile(conf_level)?,
    };
    let z_sq = z * z;
    let denom = 1.0 + z_sq / n_f;
    let center = (p_hat + z_sq / (2.0 * n_f)) / denom;
    let margin = z * (p_hat * (1.0 - p_hat) / n_f + z_sq / (4.0 * n_f * n_f)).sqrt() / denom;

    Ok(match alternative {
        Alternative::TwoSided => ((center - margin).max(0.0), (center + margin).min(1.0)),
        Alternative::Greater => ((center - margin).max(0.0), 1.0),
        Alternative::Less => (0.0, (center + margin).min(1.0)),
    })
}

/// Confidence interval for difference of two proportions (Newcombe-Wilson method).
fn prop_diff_ci(
    x1: u64,
    n1: u64,
    x2: u64,
    n2: u64,
    conf_level: f64,
    _correct: bool,
) -> EconResult<(f64, f64)> {
    let n1_f = n1 as f64;
    let n2_f = n2 as f64;
    let p1 = x1 as f64 / n1_f;
    let p2 = x2 as f64 / n2_f;
    let diff = p1 - p2;

    let z = z_critical(conf_level)?;

    // Newcombe-Wilson method using score intervals for each proportion
    let (l1, u1) = wilson_ci(x1, n1, conf_level, Alternative::TwoSided)?;
    let (l2, u2) = wilson_ci(x2, n2, conf_level, Alternative::TwoSided)?;

    // Newcombe's method 10 (hybrid score)
    let lower = diff - z * ((l1 * (1.0 - l1) / n1_f) + (u2 * (1.0 - u2) / n2_f)).sqrt();
    let upper = diff + z * ((u1 * (1.0 - u1) / n1_f) + (l2 * (1.0 - l2) / n2_f)).sqrt();

    Ok((lower.max(-1.0), upper.min(1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prop_test_one_basic() {
        // 15 successes out of 100, test against p=0.5
        let result = prop_test_one(15, 100, 0.5, Alternative::TwoSided, 0.95, true).unwrap();

        assert_eq!(result.df, 1);
        assert!((result.estimates[0] - 0.15).abs() < 0.0001);
        assert!(result.p_value < 0.001); // Very significant
    }

    #[test]
    fn test_prop_test_two_basic() {
        // 30/100 vs 40/100
        let result = prop_test_two(30, 100, 40, 100, Alternative::TwoSided, 0.95, true).unwrap();

        assert_eq!(result.df, 1);
        assert_eq!(result.estimates.len(), 2);
        assert!((result.estimates[0] - 0.3).abs() < 0.0001);
        assert!((result.estimates[1] - 0.4).abs() < 0.0001);
    }

    #[test]
    fn test_prop_test_k_basic() {
        let result = prop_test_k(&[30, 40, 35], &[100, 100, 100], 0.95).unwrap();

        assert_eq!(result.df, 2); // k-1 = 3-1 = 2
        assert_eq!(result.estimates.len(), 3);
    }

    #[test]
    fn test_prop_test_invalid_inputs() {
        // x > n
        assert!(prop_test_one(101, 100, 0.5, Alternative::TwoSided, 0.95, true).is_err());

        // n = 0
        assert!(prop_test_one(0, 0, 0.5, Alternative::TwoSided, 0.95, true).is_err());

        // p out of range
        assert!(prop_test_one(15, 100, 0.0, Alternative::TwoSided, 0.95, true).is_err());
        assert!(prop_test_one(15, 100, 1.0, Alternative::TwoSided, 0.95, true).is_err());
    }

    // ========================================================================
    // Validation Tests Against R
    // ========================================================================

    /// X-3: prop_test_two (no check) and NaN levels (passed `<= 0 || >= 1`) reached statrs inverse_cdf.
    #[test]
    fn test_prop_test_rejects_invalid_level() {
        for level in [1.5, 95.0, 0.0, f64::NAN] {
            for r in [
                prop_test_one(15, 100, 0.1, Alternative::TwoSided, level, true),
                prop_test_two(30, 100, 40, 100, Alternative::TwoSided, level, true),
                prop_test_k(&[30, 40, 35], &[100, 100, 100], level),
            ] {
                assert!(r.unwrap_err().to_string().contains("conf_level"));
            }
        }
    }

    /// STAT-67 (prop.test): statistics and p-values for every alternative, with and without correction; the Wilson CI
    /// without correction. The corrected and the two-sample CIs are deviations (STAT-63, STAT-59).
    #[test]
    fn test_validate_prop_test_t3() {
        use p2a_validation::{Check, Tol};
        let (case, _ds) = crate::test_support::load_case("stat_prop_t3");
        let u = |k: &str| case.scalar(k) as u64;
        let (x, n, p0) = (u("one_x"), u("one_n"), case.scalar("one_p"));
        let (x1, n1, x2, n2) = (u("two_x1"), u("two_n1"), u("two_x2"), u("two_n2"));
        let mut c = Check::new(&case);
        for correct in [true, false] {
            for (alt_tag, alt) in [
                ("two", Alternative::TwoSided),
                ("less", Alternative::Less),
                ("greater", Alternative::Greater),
            ] {
                let tag = format!("{}_{alt_tag}", if correct { "cc" } else { "nocc" });
                let one = prop_test_one(x, n, p0, alt, 0.95, correct).unwrap();
                let two = prop_test_two(x1, n1, x2, n2, alt, 0.95, correct).unwrap();
                c.scalar(&format!("one_{tag}_stat"), one.chi_squared, Tol::stat(100))
                    .scalar(&format!("one_{tag}_p"), one.p_value, Tol::p_value(100))
                    .scalar(&format!("two_{tag}_stat"), two.chi_squared, Tol::stat(100))
                    .scalar(&format!("two_{tag}_p"), two.p_value, Tol::p_value(100));
                if !correct {
                    c.vector(
                        &format!("one_{tag}_ci"),
                        &[one.conf_int_lower, one.conf_int_upper],
                        Tol::stat(100),
                    );
                }
                // deviation STAT-63: one_cc_*_ci (continuity-corrected Wilson) and two_*_ci (Wald +- Yates);
                // deviation STAT-59: two-sample one-sided CIs.
            }
        }
        let one = prop_test_one(x, n, p0, Alternative::TwoSided, 0.95, true).unwrap();
        let two = prop_test_two(x1, n1, x2, n2, Alternative::TwoSided, 0.95, true).unwrap();
        let kx: Vec<u64> = case.vector("k_x").iter().map(|&v| v as u64).collect();
        let kn: Vec<u64> = case.vector("k_n").iter().map(|&v| v as u64).collect();
        let k = prop_test_k(&kx, &kn, 0.95).unwrap();
        c.scalar("one_est", one.estimates[0], Tol::coef(100))
            .vector("two_est", &two.estimates, Tol::coef(100))
            .scalar("k_stat", k.chi_squared, Tol::stat(300))
            .scalar("k_df", k.df as f64, Tol::exact())
            .scalar("k_p", k.p_value, Tol::p_value(300))
            .finish();
    }
}
