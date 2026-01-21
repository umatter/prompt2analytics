//! Power analysis for statistical tests.
//!
//! Implements power.t.test, power.prop.test, and power.anova.test from R stats.
//! These functions compute statistical power or determine parameters to achieve
//! a target power level.

use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, StudentsT, Normal, FisherSnedecor};
use crate::errors::{EconError, EconResult};

/// Type of t-test for power analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TTestType {
    /// Two independent samples
    #[default]
    TwoSample,
    /// One sample vs hypothesized mean
    OneSample,
    /// Paired samples
    Paired,
}

/// Alternative hypothesis specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PowerAlternative {
    /// Two-sided test (default)
    #[default]
    TwoSided,
    /// One-sided test
    OneSided,
}

/// Result of power.t.test calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerTTestResult {
    /// Sample size (per group for two-sample)
    pub n: f64,
    /// True difference in means
    pub delta: f64,
    /// Standard deviation
    pub sd: f64,
    /// Significance level (Type I error rate)
    pub sig_level: f64,
    /// Power (1 - Type II error rate)
    pub power: f64,
    /// Alternative hypothesis
    pub alternative: PowerAlternative,
    /// Type of t-test
    pub test_type: TTestType,
    /// Method description
    pub method: String,
    /// Additional notes
    pub note: Option<String>,
}

/// Compute power for a t-test or solve for a missing parameter.
///
/// Given any four of {n, delta, sd, sig_level, power}, computes the fifth.
///
/// # Arguments
///
/// * `n` - Sample size per group (None to solve for it)
/// * `delta` - True difference in means (None to solve for it)
/// * `sd` - Standard deviation (None to solve for it)
/// * `sig_level` - Significance level (None to solve for it)
/// * `power` - Desired power (None to solve for it)
/// * `test_type` - Type of t-test
/// * `alternative` - One-sided or two-sided
///
/// # Returns
///
/// A `PowerTTestResult` with all parameters filled in.
pub fn power_t_test(
    n: Option<f64>,
    delta: Option<f64>,
    sd: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    test_type: TTestType,
    alternative: PowerAlternative,
) -> EconResult<PowerTTestResult> {
    // Count how many parameters are None
    let none_count = [n.is_none(), delta.is_none(), sd.is_none(), sig_level.is_none(), power.is_none()]
        .iter()
        .filter(|&&x| x)
        .count();

    if none_count != 1 {
        return Err(EconError::InvalidSpecification {
            message: "Exactly one of n, delta, sd, sig_level, power must be None".to_string(),
        });
    }

    // Set defaults
    let sd = sd.unwrap_or(1.0);
    let sig_level = sig_level.unwrap_or(0.05);

    // Validate parameters
    if let Some(n_val) = n {
        if n_val < 2.0 {
            return Err(EconError::InvalidSpecification {
                message: "n must be at least 2".to_string(),
            });
        }
    }
    if sig_level <= 0.0 || sig_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "sig_level must be between 0 and 1".to_string(),
        });
    }
    if let Some(p) = power {
        if p <= 0.0 || p >= 1.0 {
            return Err(EconError::InvalidSpecification {
                message: "power must be between 0 and 1".to_string(),
            });
        }
    }

    // Solve for the missing parameter
    let result = if n.is_none() {
        let delta = delta.unwrap();
        let power = power.unwrap();
        let n = solve_n_t_test(delta, sd, sig_level, power, test_type, alternative)?;
        (n, delta, sd, sig_level, power)
    } else if delta.is_none() {
        let n = n.unwrap();
        let power = power.unwrap();
        let delta = solve_delta_t_test(n, sd, sig_level, power, test_type, alternative)?;
        (n, delta, sd, sig_level, power)
    } else if power.is_none() {
        let n = n.unwrap();
        let delta = delta.unwrap();
        let power = compute_power_t_test(n, delta, sd, sig_level, test_type, alternative)?;
        (n, delta, sd, sig_level, power)
    } else {
        // Solving for sd or sig_level - less common
        return Err(EconError::InvalidSpecification {
            message: "Solving for sd or sig_level not yet implemented. Provide them as known values.".to_string(),
        });
    };

    let method = match test_type {
        TTestType::TwoSample => "Two-sample t test power calculation",
        TTestType::OneSample => "One-sample t test power calculation",
        TTestType::Paired => "Paired t test power calculation",
    };

    let note = match test_type {
        TTestType::TwoSample => Some("n is number in *each* group".to_string()),
        _ => None,
    };

    Ok(PowerTTestResult {
        n: result.0,
        delta: result.1,
        sd: result.2,
        sig_level: result.3,
        power: result.4,
        alternative,
        test_type,
        method: method.to_string(),
        note,
    })
}

/// Compute power given all other parameters.
fn compute_power_t_test(
    n: f64,
    delta: f64,
    sd: f64,
    sig_level: f64,
    test_type: TTestType,
    alternative: PowerAlternative,
) -> EconResult<f64> {
    // Degrees of freedom
    let df = match test_type {
        TTestType::TwoSample => 2.0 * (n - 1.0),
        TTestType::OneSample | TTestType::Paired => n - 1.0,
    };

    // Standard error
    let se = match test_type {
        TTestType::TwoSample => sd * (2.0 / n).sqrt(),
        TTestType::OneSample | TTestType::Paired => sd / n.sqrt(),
    };

    // Non-centrality parameter
    let ncp = delta.abs() / se;

    // Critical value(s)
    let t_dist = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| EconError::Computation(format!("t distribution error: {}", e)))?;

    let power = match alternative {
        PowerAlternative::TwoSided => {
            let t_crit = t_dist.inverse_cdf(1.0 - sig_level / 2.0);
            // Power = P(|T| > t_crit) under alternative
            // Using non-central t distribution approximation
            let nct_upper = 1.0 - nct_cdf(t_crit, df, ncp);
            let nct_lower = nct_cdf(-t_crit, df, ncp);
            nct_upper + nct_lower
        }
        PowerAlternative::OneSided => {
            let t_crit = t_dist.inverse_cdf(1.0 - sig_level);
            if delta >= 0.0 {
                1.0 - nct_cdf(t_crit, df, ncp)
            } else {
                nct_cdf(-t_crit, df, -ncp)
            }
        }
    };

    Ok(power.max(0.0).min(1.0))
}

/// Solve for n given other parameters using bisection.
fn solve_n_t_test(
    delta: f64,
    sd: f64,
    sig_level: f64,
    target_power: f64,
    test_type: TTestType,
    alternative: PowerAlternative,
) -> EconResult<f64> {
    // Bisection search for n
    let mut low = 2.0;
    let mut high = 1e6;

    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let power = compute_power_t_test(mid, delta, sd, sig_level, test_type, alternative)?;

        if (power - target_power).abs() < 1e-7 {
            return Ok(mid.ceil());
        }

        if power < target_power {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(((low + high) / 2.0).ceil())
}

/// Solve for delta given other parameters.
fn solve_delta_t_test(
    n: f64,
    sd: f64,
    sig_level: f64,
    target_power: f64,
    test_type: TTestType,
    alternative: PowerAlternative,
) -> EconResult<f64> {
    // Bisection search for delta
    let mut low = 0.0;
    let mut high = 10.0 * sd;

    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let power = compute_power_t_test(n, mid, sd, sig_level, test_type, alternative)?;

        if (power - target_power).abs() < 1e-7 {
            return Ok(mid);
        }

        if power < target_power {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok((low + high) / 2.0)
}

/// Non-central t CDF approximation.
/// Uses the approximation from Lenth (1989).
fn nct_cdf(t: f64, df: f64, ncp: f64) -> f64 {
    // For ncp = 0, this is just the central t
    if ncp.abs() < 1e-10 {
        if let Ok(t_dist) = StudentsT::new(0.0, 1.0, df) {
            return t_dist.cdf(t);
        }
        return 0.5;
    }

    // Use normal approximation for large df
    if df > 100.0 {
        if let Ok(normal) = Normal::new(ncp, 1.0) {
            return normal.cdf(t);
        }
        return 0.5;
    }

    // Approximation using series expansion
    // P(T < t | ncp) ≈ Φ(z) where z is adjusted for non-centrality
    let z = t * (1.0 - 1.0 / (4.0 * df)).sqrt() - ncp;
    if let Ok(normal) = Normal::new(0.0, 1.0) {
        return normal.cdf(z);
    }
    0.5
}

// ============================================================================
// power.prop.test
// ============================================================================

/// Result of power.prop.test calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPropTestResult {
    /// Sample size per group
    pub n: f64,
    /// Proportion in group 1
    pub p1: f64,
    /// Proportion in group 2
    pub p2: f64,
    /// Significance level
    pub sig_level: f64,
    /// Power
    pub power: f64,
    /// Alternative hypothesis
    pub alternative: PowerAlternative,
    /// Method description
    pub method: String,
    /// Additional notes
    pub note: String,
}

/// Compute power for a two-sample proportion test.
///
/// # Arguments
///
/// * `n` - Sample size per group (None to solve for it)
/// * `p1` - Proportion in group 1
/// * `p2` - Proportion in group 2 (None to solve for it given h effect size)
/// * `sig_level` - Significance level (None to solve for it)
/// * `power` - Desired power (None to solve for it)
/// * `alternative` - One-sided or two-sided
pub fn power_prop_test(
    n: Option<f64>,
    p1: f64,
    p2: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    alternative: PowerAlternative,
) -> EconResult<PowerPropTestResult> {
    let sig_level = sig_level.unwrap_or(0.05);

    // Validate
    if p1 <= 0.0 || p1 >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "p1 must be between 0 and 1".to_string(),
        });
    }

    let p2 = p2.ok_or_else(|| EconError::InvalidSpecification {
        message: "p2 must be provided".to_string(),
    })?;

    if p2 <= 0.0 || p2 >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "p2 must be between 0 and 1".to_string(),
        });
    }

    let result = if n.is_none() {
        let power = power.ok_or_else(|| EconError::InvalidSpecification {
            message: "Must provide power when solving for n".to_string(),
        })?;
        let n = solve_n_prop_test(p1, p2, sig_level, power, alternative)?;
        (n, sig_level, power)
    } else if power.is_none() {
        let n = n.unwrap();
        let power = compute_power_prop_test(n, p1, p2, sig_level, alternative)?;
        (n, sig_level, power)
    } else {
        return Err(EconError::InvalidSpecification {
            message: "Exactly one of n, power must be None".to_string(),
        });
    };

    Ok(PowerPropTestResult {
        n: result.0,
        p1,
        p2,
        sig_level: result.1,
        power: result.2,
        alternative,
        method: "Two-sample comparison of proportions power calculation".to_string(),
        note: "n is number in *each* group".to_string(),
    })
}

/// Compute power for proportion test.
fn compute_power_prop_test(
    n: f64,
    p1: f64,
    p2: f64,
    sig_level: f64,
    alternative: PowerAlternative,
) -> EconResult<f64> {
    // Effect size (Cohen's h)
    let h = 2.0 * (p1.sqrt().asin() - p2.sqrt().asin());

    // Standard error under null
    let p_pooled = (p1 + p2) / 2.0;
    let se = (2.0 * p_pooled * (1.0 - p_pooled) / n).sqrt();

    // Non-centrality parameter
    let ncp = (p1 - p2).abs() / se;

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| EconError::Computation(format!("Normal distribution error: {}", e)))?;

    let power = match alternative {
        PowerAlternative::TwoSided => {
            let z_crit = normal.inverse_cdf(1.0 - sig_level / 2.0);
            let power_upper = 1.0 - Normal::new(ncp, 1.0)
                .map(|d| d.cdf(z_crit))
                .unwrap_or(0.5);
            let power_lower = Normal::new(ncp, 1.0)
                .map(|d| d.cdf(-z_crit))
                .unwrap_or(0.5);
            power_upper + power_lower
        }
        PowerAlternative::OneSided => {
            let z_crit = normal.inverse_cdf(1.0 - sig_level);
            1.0 - Normal::new(ncp, 1.0)
                .map(|d| d.cdf(z_crit))
                .unwrap_or(0.5)
        }
    };

    Ok(power.max(0.0).min(1.0))
}

/// Solve for n in proportion test.
fn solve_n_prop_test(
    p1: f64,
    p2: f64,
    sig_level: f64,
    target_power: f64,
    alternative: PowerAlternative,
) -> EconResult<f64> {
    let mut low = 2.0;
    let mut high = 1e6;

    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let power = compute_power_prop_test(mid, p1, p2, sig_level, alternative)?;

        if (power - target_power).abs() < 1e-7 {
            return Ok(mid.ceil());
        }

        if power < target_power {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(((low + high) / 2.0).ceil())
}

// ============================================================================
// power.anova.test
// ============================================================================

/// Result of power.anova.test calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnovaTestResult {
    /// Number of groups
    pub groups: usize,
    /// Sample size per group
    pub n: f64,
    /// Between-group variance
    pub between_var: f64,
    /// Within-group variance
    pub within_var: f64,
    /// Significance level
    pub sig_level: f64,
    /// Power
    pub power: f64,
    /// Method description
    pub method: String,
    /// Additional notes
    pub note: String,
}

/// Compute power for a balanced one-way ANOVA.
///
/// # Arguments
///
/// * `groups` - Number of groups
/// * `n` - Sample size per group (None to solve for it)
/// * `between_var` - Between-group variance
/// * `within_var` - Within-group variance
/// * `sig_level` - Significance level
/// * `power` - Desired power (None to solve for it)
pub fn power_anova_test(
    groups: usize,
    n: Option<f64>,
    between_var: f64,
    within_var: f64,
    sig_level: Option<f64>,
    power: Option<f64>,
) -> EconResult<PowerAnovaTestResult> {
    if groups < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Need at least 2 groups".to_string(),
        });
    }

    let sig_level = sig_level.unwrap_or(0.05);

    let result = if n.is_none() {
        let power = power.ok_or_else(|| EconError::InvalidSpecification {
            message: "Must provide power when solving for n".to_string(),
        })?;
        let n = solve_n_anova(groups, between_var, within_var, sig_level, power)?;
        (n, power)
    } else if power.is_none() {
        let n = n.unwrap();
        let power = compute_power_anova(groups, n, between_var, within_var, sig_level)?;
        (n, power)
    } else {
        return Err(EconError::InvalidSpecification {
            message: "Exactly one of n, power must be None".to_string(),
        });
    };

    Ok(PowerAnovaTestResult {
        groups,
        n: result.0,
        between_var,
        within_var,
        sig_level,
        power: result.1,
        method: "Balanced one-way analysis of variance power calculation".to_string(),
        note: "n is number in each group".to_string(),
    })
}

/// Compute power for ANOVA.
fn compute_power_anova(
    groups: usize,
    n: f64,
    between_var: f64,
    within_var: f64,
    sig_level: f64,
) -> EconResult<f64> {
    let k = groups as f64;
    let df1 = k - 1.0;
    let df2 = k * (n - 1.0);

    // Non-centrality parameter: lambda = n * sum((mu_i - mu_bar)^2) / sigma^2
    // = n * k * between_var / within_var
    let ncp = n * k * between_var / within_var;

    // Critical F value
    let f_dist = FisherSnedecor::new(df1, df2)
        .map_err(|e| EconError::Computation(format!("F distribution error: {}", e)))?;

    let f_crit = f_dist.inverse_cdf(1.0 - sig_level);

    // Power = P(F > f_crit | ncp)
    // Using non-central F approximation
    let power = 1.0 - ncf_cdf(f_crit, df1, df2, ncp);

    Ok(power.max(0.0).min(1.0))
}

/// Solve for n in ANOVA.
fn solve_n_anova(
    groups: usize,
    between_var: f64,
    within_var: f64,
    sig_level: f64,
    target_power: f64,
) -> EconResult<f64> {
    let mut low = 2.0;
    let mut high = 1e6;

    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        let power = compute_power_anova(groups, mid, between_var, within_var, sig_level)?;

        if (power - target_power).abs() < 1e-7 {
            return Ok(mid.ceil());
        }

        if power < target_power {
            low = mid;
        } else {
            high = mid;
        }
    }

    Ok(((low + high) / 2.0).ceil())
}

/// Non-central F CDF approximation.
fn ncf_cdf(f: f64, df1: f64, df2: f64, ncp: f64) -> f64 {
    if ncp < 1e-10 {
        if let Ok(f_dist) = FisherSnedecor::new(df1, df2) {
            return f_dist.cdf(f);
        }
        return 0.5;
    }

    // Patnaik approximation for non-central F
    let h = 2.0 * (df1 + ncp).powi(2) / (df1.powi(2) + 2.0 * (df1 + ncp));
    let k = (df1 + 2.0 * ncp) / (df1 + ncp);

    if let Ok(f_dist) = FisherSnedecor::new(h, df2) {
        f_dist.cdf(f / k)
    } else {
        0.5
    }
}

// ============================================================================
// MCP-friendly wrappers
// ============================================================================

/// Run power.t.test with string parameters (MCP wrapper).
pub fn run_power_t_test(
    n: Option<f64>,
    delta: Option<f64>,
    sd: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    test_type: &str,
    alternative: &str,
) -> EconResult<PowerTTestResult> {
    let test_type = match test_type.to_lowercase().as_str() {
        "two.sample" | "two_sample" | "twosample" => TTestType::TwoSample,
        "one.sample" | "one_sample" | "onesample" => TTestType::OneSample,
        "paired" => TTestType::Paired,
        _ => return Err(EconError::InvalidSpecification {
            message: format!("Unknown test type: {}", test_type)
        }),
    };

    let alternative = match alternative.to_lowercase().as_str() {
        "two.sided" | "two_sided" | "twosided" => PowerAlternative::TwoSided,
        "one.sided" | "one_sided" | "onesided" => PowerAlternative::OneSided,
        _ => return Err(EconError::InvalidSpecification {
            message: format!("Unknown alternative: {}", alternative)
        }),
    };

    power_t_test(n, delta, sd, sig_level, power, test_type, alternative)
}

/// Run power.prop.test with string parameters (MCP wrapper).
pub fn run_power_prop_test(
    n: Option<f64>,
    p1: f64,
    p2: f64,
    sig_level: Option<f64>,
    power: Option<f64>,
    alternative: &str,
) -> EconResult<PowerPropTestResult> {
    let alternative = match alternative.to_lowercase().as_str() {
        "two.sided" | "two_sided" | "twosided" => PowerAlternative::TwoSided,
        "one.sided" | "one_sided" | "onesided" => PowerAlternative::OneSided,
        _ => return Err(EconError::InvalidSpecification {
            message: format!("Unknown alternative: {}", alternative)
        }),
    };

    power_prop_test(n, p1, Some(p2), sig_level, power, alternative)
}

/// Run power.anova.test (MCP wrapper).
pub fn run_power_anova_test(
    groups: usize,
    n: Option<f64>,
    between_var: f64,
    within_var: f64,
    sig_level: Option<f64>,
    power: Option<f64>,
) -> EconResult<PowerAnovaTestResult> {
    power_anova_test(groups, n, between_var, within_var, sig_level, power)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_t_test_compute_power() {
        // With n=20, delta=0.5, sd=1, sig_level=0.05, power should be moderate
        let result = power_t_test(
            Some(20.0),
            Some(0.5),
            Some(1.0),
            Some(0.05),
            None,
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        ).unwrap();

        assert!(result.power > 0.3 && result.power < 0.7);
    }

    #[test]
    fn test_power_t_test_solve_n() {
        // Solve for n to achieve 80% power
        let result = power_t_test(
            None,
            Some(0.5),
            Some(1.0),
            Some(0.05),
            Some(0.8),
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        ).unwrap();

        // n should be around 64 per group for this effect size
        assert!(result.n > 50.0 && result.n < 80.0);
    }

    #[test]
    fn test_power_t_test_solve_delta() {
        // Solve for detectable effect size with n=50
        let result = power_t_test(
            Some(50.0),
            None,
            Some(1.0),
            Some(0.05),
            Some(0.8),
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        ).unwrap();

        // delta should be moderate
        assert!(result.delta > 0.3 && result.delta < 0.8);
    }

    #[test]
    fn test_power_prop_test() {
        // Power for comparing 0.3 vs 0.5 proportions
        let result = power_prop_test(
            Some(100.0),
            0.3,
            Some(0.5),
            Some(0.05),
            None,
            PowerAlternative::TwoSided,
        ).unwrap();

        assert!(result.power > 0.7);
    }

    #[test]
    fn test_power_prop_test_solve_n() {
        let result = power_prop_test(
            None,
            0.3,
            Some(0.5),
            Some(0.05),
            Some(0.8),
            PowerAlternative::TwoSided,
        ).unwrap();

        // Should need around 100 per group
        assert!(result.n > 50.0 && result.n < 150.0);
    }

    #[test]
    fn test_power_anova() {
        // Power for 3-group ANOVA
        let result = power_anova_test(
            3,
            Some(20.0),
            0.25,  // between variance
            1.0,   // within variance
            Some(0.05),
            None,
        ).unwrap();

        assert!(result.power > 0.0 && result.power < 1.0);
    }

    #[test]
    fn test_power_anova_solve_n() {
        let result = power_anova_test(
            3,
            None,
            0.25,
            1.0,
            Some(0.05),
            Some(0.8),
        ).unwrap();

        // Should need moderate n per group
        assert!(result.n > 10.0);
    }
}
