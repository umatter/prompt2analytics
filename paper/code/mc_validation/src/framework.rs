//! Monte Carlo validation framework
//!
//! Provides the core runner, result types, and tolerance calculations for
//! MC-based statistical property validation.

use serde::{Deserialize, Serialize};

/// Configuration for a Monte Carlo validation run.
#[derive(Clone, Debug)]
pub struct McConfig {
    /// Number of simulation replications.
    pub n_sims: usize,
    /// Master seed (individual sims use seed + i).
    pub seed: u64,
    /// Nominal significance level for hypothesis tests and CI coverage.
    pub alpha: f64,
    /// Confidence level for the binomial tolerance interval on MC rates.
    pub mc_confidence: f64,
}

impl Default for McConfig {
    fn default() -> Self {
        Self {
            n_sims: 1000,
            seed: 42,
            alpha: 0.05,
            mc_confidence: 0.95,
        }
    }
}

/// Outcome of a single simulation replication (for estimators).
#[derive(Clone, Debug)]
pub struct EstimatorDraw {
    /// Point estimate (e.g., coefficient).
    pub estimate: f64,
    /// Reported standard error.
    pub std_error: f64,
    /// Lower bound of confidence interval.
    pub ci_lower: f64,
    /// Upper bound of confidence interval.
    pub ci_upper: f64,
}

/// Outcome of a single simulation replication (for hypothesis tests).
#[derive(Clone, Debug)]
pub struct TestDraw {
    /// p-value from the test.
    pub p_value: f64,
    /// Test statistic.
    pub statistic: f64,
}

/// Result of a Monte Carlo validation for one property of one method.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McResult {
    pub method: String,
    pub property: String,
    pub dgp: String,
    pub n: usize,
    pub n_sims: usize,
    pub n_successful: usize,
    pub observed: f64,
    pub expected: f64,
    pub within_tolerance: bool,
    pub tolerance_lower: f64,
    pub tolerance_upper: f64,
    /// Extra info (e.g., mean bias, RMSE).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<McDetails>,
}

/// Additional diagnostics for estimator-type MC results.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct McDetails {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_bias: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rmse: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub se_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_estimate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empirical_sd: Option<f64>,
}

/// Compute binomial confidence interval for a proportion.
///
/// Uses the normal approximation: p ± z * sqrt(p*(1-p)/n).
/// For MC validation this is appropriate since n_sims >= 100.
pub fn binomial_ci(p: f64, n: usize, confidence: f64) -> (f64, f64) {
    let z = normal_quantile((1.0 + confidence) / 2.0);
    let se = (p * (1.0 - p) / n as f64).sqrt();
    ((p - z * se).max(0.0), (p + z * se).min(1.0))
}

/// Check whether an observed rate falls within the expected binomial CI.
pub fn rate_within_tolerance(observed: f64, expected: f64, n_sims: usize, confidence: f64) -> bool {
    let (lo, hi) = binomial_ci(expected, n_sims, confidence);
    observed >= lo && observed <= hi
}

/// Evaluate CI coverage from a set of estimator draws.
pub fn evaluate_coverage(
    draws: &[EstimatorDraw],
    true_value: f64,
    config: &McConfig,
) -> McResult {
    let n_successful = draws.len();
    let n_covered = draws
        .iter()
        .filter(|d| d.ci_lower <= true_value && true_value <= d.ci_upper)
        .count();
    let coverage = n_covered as f64 / n_successful as f64;
    let expected = 1.0 - config.alpha;
    let (lo, hi) = binomial_ci(expected, n_successful, config.mc_confidence);

    // Compute additional diagnostics
    let estimates: Vec<f64> = draws.iter().map(|d| d.estimate).collect();
    let mean_est = mean(&estimates);
    let bias = mean_est - true_value;
    let rmse = (draws.iter().map(|d| (d.estimate - true_value).powi(2)).sum::<f64>()
        / n_successful as f64)
        .sqrt();
    let emp_sd = std_dev(&estimates);
    let mean_se = mean(&draws.iter().map(|d| d.std_error).collect::<Vec<_>>());
    let se_ratio = if emp_sd > 1e-15 { mean_se / emp_sd } else { f64::NAN };

    McResult {
        method: String::new(), // filled by caller
        property: "ci_coverage".to_string(),
        dgp: String::new(),
        n: 0,
        n_sims: config.n_sims,
        n_successful,
        observed: coverage,
        expected,
        within_tolerance: coverage >= lo && coverage <= hi,
        tolerance_lower: lo,
        tolerance_upper: hi,
        details: Some(McDetails {
            mean_bias: Some(bias),
            rmse: Some(rmse),
            se_ratio: Some(se_ratio),
            median_estimate: Some(median(&estimates)),
            empirical_sd: Some(emp_sd),
        }),
    }
}

/// Evaluate Type I error rate from hypothesis test draws under H0.
pub fn evaluate_size(draws: &[TestDraw], config: &McConfig) -> McResult {
    let n_successful = draws.len();
    let n_rejected = draws.iter().filter(|d| d.p_value < config.alpha).count();
    let rejection_rate = n_rejected as f64 / n_successful as f64;
    let (lo, hi) = binomial_ci(config.alpha, n_successful, config.mc_confidence);

    McResult {
        method: String::new(),
        property: "type_i_error".to_string(),
        dgp: String::new(),
        n: 0,
        n_sims: config.n_sims,
        n_successful,
        observed: rejection_rate,
        expected: config.alpha,
        within_tolerance: rejection_rate >= lo && rejection_rate <= hi,
        tolerance_lower: lo,
        tolerance_upper: hi,
        details: None,
    }
}

/// Evaluate SE accuracy: mean(reported SE) / empirical SD of estimates.
pub fn evaluate_se_accuracy(draws: &[EstimatorDraw], config: &McConfig) -> McResult {
    let estimates: Vec<f64> = draws.iter().map(|d| d.estimate).collect();
    let reported_ses: Vec<f64> = draws.iter().map(|d| d.std_error).collect();
    let emp_sd = std_dev(&estimates);
    let mean_se = mean(&reported_ses);
    let ratio = if emp_sd > 1e-15 { mean_se / emp_sd } else { f64::NAN };

    // SE ratio should be close to 1.0. Use tolerance [0.9, 1.1] as a
    // reasonable band (well-calibrated SEs).
    let within = ratio >= 0.9 && ratio <= 1.1;

    McResult {
        method: String::new(),
        property: "se_accuracy".to_string(),
        dgp: String::new(),
        n: 0,
        n_sims: config.n_sims,
        n_successful: draws.len(),
        observed: ratio,
        expected: 1.0,
        within_tolerance: within,
        tolerance_lower: 0.9,
        tolerance_upper: 1.1,
        details: Some(McDetails {
            mean_bias: None,
            rmse: None,
            se_ratio: Some(ratio),
            median_estimate: None,
            empirical_sd: Some(emp_sd),
        }),
    }
}

// ---- Helpers ----

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return f64::NAN; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    if xs.len() < 2 { return f64::NAN; }
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() { return f64::NAN; }
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Normal quantile via rational approximation (Abramowitz & Stegun 26.2.23).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -val } else { val }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binomial_ci() {
        let (lo, hi) = binomial_ci(0.05, 1000, 0.95);
        assert!(lo < 0.05);
        assert!(hi > 0.05);
        // For n=1000, p=0.05: SE ≈ 0.00689, z=1.96, so CI ≈ [0.0365, 0.0635]
        assert!((lo - 0.0365).abs() < 0.001);
        assert!((hi - 0.0635).abs() < 0.001);
    }

    #[test]
    fn test_normal_quantile() {
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.5) - 0.0).abs() < 0.001);
    }
}
