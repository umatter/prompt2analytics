//! Correlation test for paired samples.
//!
//! Implements cor.test from R stats: tests for association between paired samples
//! using Pearson's product moment correlation coefficient, Kendall's tau, or
//! Spearman's rho.

use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::Alternative;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

/// Method for computing correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CorrelationMethod {
    /// Pearson's product-moment correlation (default)
    #[default]
    Pearson,
    /// Kendall's tau rank correlation
    Kendall,
    /// Spearman's rho rank correlation
    Spearman,
}

/// Result of a correlation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorTestResult {
    /// The correlation coefficient estimate
    pub estimate: f64,
    /// Name of the estimate (cor, tau, or rho)
    pub estimate_name: String,
    /// Test statistic (t for Pearson, S for Spearman, z for Kendall)
    pub statistic: f64,
    /// Name of the statistic
    pub statistic_name: String,
    /// P-value of the test
    pub p_value: f64,
    /// Degrees of freedom (for Pearson, None for rank methods)
    pub df: Option<f64>,
    /// Lower bound of confidence interval (Pearson only)
    pub conf_low: Option<f64>,
    /// Upper bound of confidence interval (Pearson only)
    pub conf_high: Option<f64>,
    /// Confidence level used
    pub conf_level: f64,
    /// Sample size
    pub n: usize,
    /// The null hypothesis value (always 0)
    pub null_value: f64,
    /// Alternative hypothesis
    pub alternative: Alternative,
    /// Method used
    pub method: CorrelationMethod,
    /// Method description string
    pub method_name: String,
}

/// Perform a correlation test between two samples.
///
/// Tests for association between paired samples using Pearson, Kendall, or Spearman
/// correlation.
///
/// # Arguments
///
/// * `x` - First sample
/// * `y` - Second sample (must have same length as x)
/// * `method` - Correlation method to use
/// * `alternative` - Alternative hypothesis
/// * `conf_level` - Confidence level for CI (Pearson only)
///
/// # Returns
///
/// A `CorTestResult` containing the correlation coefficient, test statistic,
/// p-value, and confidence interval (for Pearson).
///
/// # Example
///
/// ```
/// use p2a_core::stats::cortest::{cor_test, CorrelationMethod};
/// use p2a_core::stats::Alternative;
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = vec![1.2, 1.9, 3.1, 3.9, 5.2];
///
/// let result = cor_test(&x, &y, CorrelationMethod::Pearson, Alternative::TwoSided, 0.95).unwrap();
/// assert!(result.estimate > 0.99);
/// assert!(result.p_value < 0.01);
/// ```
pub fn cor_test(
    x: &[f64],
    y: &[f64],
    method: CorrelationMethod,
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    // Validate inputs
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: "x and y must have the same length".to_string(),
        });
    }

    // Remove pairs with NaN
    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .map(|(a, b)| (*a, *b))
        .collect();

    let n = pairs.len();
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "correlation test".to_string(),
        });
    }

    if conf_level <= 0.0 || conf_level >= 1.0 {
        return Err(EconError::InvalidSpecification {
            message: "conf_level must be between 0 and 1".to_string(),
        });
    }

    let x_clean: Vec<f64> = pairs.iter().map(|(a, _)| *a).collect();
    let y_clean: Vec<f64> = pairs.iter().map(|(_, b)| *b).collect();

    match method {
        CorrelationMethod::Pearson => pearson_test(&x_clean, &y_clean, alternative, conf_level),
        CorrelationMethod::Spearman => spearman_test(&x_clean, &y_clean, alternative, conf_level),
        CorrelationMethod::Kendall => kendall_test(&x_clean, &y_clean, alternative, conf_level),
    }
}

/// Pearson's product-moment correlation test.
fn pearson_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let n = x.len();
    let r = pearson_correlation(x, y);

    // Test statistic: t = r * sqrt((n-2) / (1-r²))
    let df = (n - 2) as f64;
    let t_stat = if (1.0 - r * r).abs() < 1e-15 {
        // Perfect correlation
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    };

    // P-value from t-distribution
    let p_value = if t_stat.is_infinite() {
        // For perfect correlation
        match alternative {
            Alternative::TwoSided => 0.0,
            Alternative::Less => {
                if t_stat > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Alternative::Greater => {
                if t_stat > 0.0 {
                    0.0
                } else {
                    1.0
                }
            }
        }
    } else {
        let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| {
            EconError::Computation(format!("Failed to create t distribution: {}", e))
        })?;

        match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_stat.abs())),
            Alternative::Less => t_dist.cdf(t_stat),
            Alternative::Greater => 1.0 - t_dist.cdf(t_stat),
        }
    };

    // Confidence interval using Fisher's z-transformation
    // z = 0.5 * ln((1+r)/(1-r)) = atanh(r)
    let (conf_low, conf_high) = if n >= 4 {
        let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
        let se_z = 1.0 / ((n - 3) as f64).sqrt();

        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| EconError::Computation(format!("Failed to create normal: {}", e)))?;

        let alpha = 1.0 - conf_level;
        let (z_low, z_high) = match alternative {
            Alternative::TwoSided => {
                let z_crit = normal.inverse_cdf(1.0 - alpha / 2.0);
                (z - z_crit * se_z, z + z_crit * se_z)
            }
            Alternative::Less => {
                let z_crit = normal.inverse_cdf(1.0 - alpha);
                (-1.0, z + z_crit * se_z)
            }
            Alternative::Greater => {
                let z_crit = normal.inverse_cdf(1.0 - alpha);
                (z - z_crit * se_z, 1.0)
            }
        };

        // Transform back: r = (exp(2z) - 1) / (exp(2z) + 1) = tanh(z)
        let r_low = z_low.tanh().max(-1.0);
        let r_high = z_high.tanh().min(1.0);
        (Some(r_low), Some(r_high))
    } else {
        (None, None)
    };

    Ok(CorTestResult {
        estimate: r,
        estimate_name: "cor".to_string(),
        statistic: t_stat,
        statistic_name: "t".to_string(),
        p_value,
        df: Some(df),
        conf_low,
        conf_high,
        conf_level,
        n,
        null_value: 0.0,
        alternative,
        method: CorrelationMethod::Pearson,
        method_name: "Pearson's product-moment correlation".to_string(),
    })
}

/// Spearman's rank correlation test.
fn spearman_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let n = x.len();

    // Rank the data
    let x_ranks = rank_data(x);
    let y_ranks = rank_data(y);

    // Spearman's rho is Pearson correlation of ranks
    let rho = pearson_correlation(&x_ranks, &y_ranks);

    // For larger samples, use t-approximation
    // S = sum of squared rank differences (for exact test)
    // But asymptotically, t = rho * sqrt((n-2)/(1-rho²))
    let df = (n - 2) as f64;
    let t_stat = if (1.0 - rho * rho).abs() < 1e-15 {
        if rho > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        rho * (df / (1.0 - rho * rho)).sqrt()
    };

    // P-value using t-distribution approximation
    let p_value = if t_stat.is_infinite() {
        0.0
    } else {
        let t_dist = StudentsT::new(0.0, 1.0, df)
            .map_err(|e| EconError::Computation(format!("t distribution error: {}", e)))?;

        match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - t_dist.cdf(t_stat.abs())),
            Alternative::Less => t_dist.cdf(t_stat),
            Alternative::Greater => 1.0 - t_dist.cdf(t_stat),
        }
    };

    // Note: R uses S statistic, but we use t for simplicity
    // The S statistic = sum((rank_x - rank_y)²) relates to rho by:
    // rho = 1 - 6*S / (n*(n²-1))
    let s_stat = {
        let sum_d2: f64 = x_ranks
            .iter()
            .zip(y_ranks.iter())
            .map(|(rx, ry)| (rx - ry).powi(2))
            .sum();
        sum_d2
    };

    Ok(CorTestResult {
        estimate: rho,
        estimate_name: "rho".to_string(),
        statistic: s_stat,
        statistic_name: "S".to_string(),
        p_value,
        df: None, // Rank-based test
        conf_low: None,
        conf_high: None,
        conf_level,
        n,
        null_value: 0.0,
        alternative,
        method: CorrelationMethod::Spearman,
        method_name: "Spearman's rank correlation rho".to_string(),
    })
}

/// Kendall's tau correlation test.
fn kendall_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let n = x.len();

    // S = concordant - discordant pairs (Knight's O(n log n) algorithm)
    let s = kendall_s(x, y)?;
    let n_pairs = tie_pairs(n)?;
    let tau = s as f64 / n_pairs as f64;

    // For the test, we use the normal approximation
    // Var(S) = n(n-1)(2n+5)/18 for no ties
    let var_s = kendall_var_s_no_ties(n);

    // Continuity correction
    let z_stat = if s > 0 {
        (s as f64 - 1.0) / var_s.sqrt()
    } else if s < 0 {
        (s as f64 + 1.0) / var_s.sqrt()
    } else {
        0.0
    };

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| EconError::Computation(format!("Normal distribution error: {}", e)))?;

    let p_value = match alternative {
        Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z_stat.abs())),
        Alternative::Less => normal.cdf(z_stat),
        Alternative::Greater => 1.0 - normal.cdf(z_stat),
    };

    Ok(CorTestResult {
        estimate: tau,
        estimate_name: "tau".to_string(),
        statistic: z_stat,
        statistic_name: "z".to_string(),
        p_value,
        df: None,
        conf_low: None,
        conf_high: None,
        conf_level,
        n,
        null_value: 0.0,
        alternative,
        method: CorrelationMethod::Kendall,
        method_name: "Kendall's rank correlation tau".to_string(),
    })
}

/// `t(t-1)/2`, the number of pairs among `t` values, with checked `u64` arithmetic.
fn tie_pairs(t: usize) -> EconResult<u64> {
    let t = t as u64;
    t.checked_mul(t.saturating_sub(1))
        .map(|v| v / 2)
        .ok_or_else(|| {
            EconError::ResourceLimit(
                "Kendall's tau: the number of observation pairs is too large".to_string(),
            )
        })
}

/// Kendall's S = (concordant pairs) - (discordant pairs), by Knight's (1966) algorithm.
///
/// Sort by `(x, y)`, count the pairs tied in x (`n1`) and in both (`n3`), merge-sort the y
/// values in that order counting strict inversions (the discordant pairs), and count the pairs
/// tied in y (`n2`): `S = n0 - n1 - n2 + n3 - 2 * inversions`. O(n log n). Ties are exact
/// equality, as in R (equal infinities tie; -0.0 equals 0.0). The callers have removed NaN pairs.
fn kendall_s(x: &[f64], y: &[f64]) -> EconResult<i64> {
    let n = x.len();
    let n0 = tie_pairs(n)?;
    // `+ 0.0` turns -0.0 into +0.0 so that total_cmp treats them as equal
    let key = |v: f64| v + 0.0;

    let mut order: Vec<usize> = resource::try_with_capacity(n, "Kendall sort order")?;
    order.extend(0..n);
    order.sort_unstable_by(|&a, &b| {
        key(x[a])
            .total_cmp(&key(x[b]))
            .then_with(|| key(y[a]).total_cmp(&key(y[b])))
    });

    // Pairs tied in x (n1) and tied in both x and y (n3)
    let mut n1 = 0u64;
    let mut n3 = 0u64;
    let mut i = 0;
    while i < n {
        let xi = key(x[order[i]]);
        let mut j = i + 1;
        while j < n && key(x[order[j]]) == xi {
            j += 1;
        }
        n1 += tie_pairs(j - i)?;
        let mut k = i;
        while k < j {
            let yk = key(y[order[k]]);
            let mut l = k + 1;
            while l < j && key(y[order[l]]) == yk {
                l += 1;
            }
            n3 += tie_pairs(l - k)?;
            k = l;
        }
        i = j;
    }

    // Bottom-up merge sort of y in (x, y) order, counting strict inversions
    let mut a: Vec<f64> = resource::try_with_capacity(n, "Kendall merge buffer")?;
    a.extend(order.iter().map(|&i| key(y[i])));
    let mut buf: Vec<f64> = resource::try_vec(n, 0.0, "Kendall merge buffer")?;
    let mut inversions = 0u64;
    let mut width = 1usize;
    while width < n {
        resource::checkpoint()?;
        let mut lo = 0;
        while lo < n {
            let mid = lo.saturating_add(width).min(n);
            let hi = lo.saturating_add(width.saturating_mul(2)).min(n);
            let (mut l, mut r, mut out) = (lo, mid, lo);
            while l < mid && r < hi {
                if a[r] < a[l] {
                    buf[out] = a[r];
                    r += 1;
                    inversions += (mid - l) as u64;
                } else {
                    buf[out] = a[l];
                    l += 1;
                }
                out += 1;
            }
            buf[out..out + (mid - l)].copy_from_slice(&a[l..mid]);
            out += mid - l;
            buf[out..out + (hi - r)].copy_from_slice(&a[r..hi]);
            lo = hi;
        }
        std::mem::swap(&mut a, &mut buf);
        width = width.saturating_mul(2);
    }

    // Pairs tied in y (n2), from the sorted y values
    let mut n2 = 0u64;
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && a[j] == a[i] {
            j += 1;
        }
        n2 += tie_pairs(j - i)?;
        i = j;
    }

    let s = i128::from(n0) - i128::from(n1) - i128::from(n2) + i128::from(n3)
        - 2 * i128::from(inversions);
    i64::try_from(s).map_err(|_| {
        EconError::ResourceLimit("Kendall's tau: the statistic does not fit in 64 bits".to_string())
    })
}

/// Tie-free variance of Kendall's S, `n(n-1)(2n+5)/18`, with the product exact in `u128`.
fn kendall_var_s_no_ties(n: usize) -> f64 {
    let n = n as u128;
    (n * n.saturating_sub(1) * (2 * n + 5)) as f64 / 18.0
}

/// Compute Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let x_mean: f64 = x.iter().sum::<f64>() / n;
    let y_mean: f64 = y.iter().sum::<f64>() / n;

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        let dy = yi - y_mean;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    if sum_x2 == 0.0 || sum_y2 == 0.0 {
        return f64::NAN;
    }

    sum_xy / (sum_x2 * sum_y2).sqrt()
}

/// Assign ranks to data, handling ties with average ranks.
fn rank_data(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        // Find all elements with the same value (ties)
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-15 {
            j += 1;
        }
        // Assign average rank to all tied elements
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }

    ranks
}

/// Run correlation test from Dataset columns (MCP-friendly wrapper).
pub fn run_cor_test(
    x: &[f64],
    y: &[f64],
    method: &str,
    alternative: &str,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let method = match method.to_lowercase().as_str() {
        "pearson" => CorrelationMethod::Pearson,
        "spearman" => CorrelationMethod::Spearman,
        "kendall" => CorrelationMethod::Kendall,
        _ => {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Unknown method: {}. Use 'pearson', 'spearman', or 'kendall'",
                    method
                ),
            });
        }
    };

    let alternative = match alternative.to_lowercase().as_str() {
        "two.sided" | "two_sided" | "twosided" => Alternative::TwoSided,
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        _ => {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Unknown alternative: {}. Use 'two.sided', 'less', or 'greater'",
                    alternative
                ),
            });
        }
    };

    cor_test(x, y, method, alternative, conf_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: today's O(n^2) all-pairs Kendall test, copied verbatim.
    fn kendall_test_pairwise_reference(
        x: &[f64],
        y: &[f64],
        alternative: Alternative,
        conf_level: f64,
    ) -> EconResult<CorTestResult> {
        let n = x.len();

        // Count concordant and discordant pairs
        let mut concordant = 0i64;
        let mut discordant = 0i64;
        let mut ties_x = 0i64;
        let mut ties_y = 0i64;

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[j] - x[i];
                let dy = y[j] - y[i];

                if dx.abs() < 1e-15 && dy.abs() < 1e-15 {
                    // Tie in both
                    ties_x += 1;
                    ties_y += 1;
                } else if dx.abs() < 1e-15 {
                    ties_x += 1;
                } else if dy.abs() < 1e-15 {
                    ties_y += 1;
                } else if dx * dy > 0.0 {
                    concordant += 1;
                } else {
                    discordant += 1;
                }
            }
        }

        let n_pairs = (n * (n - 1) / 2) as i64;
        let tau = (concordant - discordant) as f64 / n_pairs as f64;

        // For the test, we use the normal approximation
        // Var(S) = n(n-1)(2n+5)/18 for no ties
        let s = concordant - discordant;
        let var_s = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;

        // Continuity correction
        let z_stat = if s > 0 {
            (s as f64 - 1.0) / var_s.sqrt()
        } else if s < 0 {
            (s as f64 + 1.0) / var_s.sqrt()
        } else {
            0.0
        };

        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| EconError::Computation(format!("Normal distribution error: {}", e)))?;

        let p_value = match alternative {
            Alternative::TwoSided => 2.0 * (1.0 - normal.cdf(z_stat.abs())),
            Alternative::Less => normal.cdf(z_stat),
            Alternative::Greater => 1.0 - normal.cdf(z_stat),
        };

        Ok(CorTestResult {
            estimate: tau,
            estimate_name: "tau".to_string(),
            statistic: z_stat,
            statistic_name: "z".to_string(),
            p_value,
            df: None,
            conf_low: None,
            conf_high: None,
            conf_level,
            n,
            null_value: 0.0,
            alternative,
            method: CorrelationMethod::Kendall,
            method_name: "Kendall's rank correlation tau".to_string(),
        })
    }

    // R 4.3.3: set.seed(20261021); x <- round(rnorm(40), 4); y <- round(0.6 * x + rnorm(40, sd = 0.8), 4)
    //   cor.test(x, y, method = "kendall", exact = FALSE, continuity = TRUE, alternative = alt)
    const KX: [f64; 40] = [
        0.1603, 0.5681, 0.5066, 1.0145, 0.6190, -0.6551, -0.7773, 0.4089, 1.7813, 0.8909, -0.3686,
        -0.3427, 0.2088, -1.1010, -1.2737, 0.5803, -1.0766, 2.2568, -0.6514, 0.1311, -1.1748,
        -0.5524, 1.6524, 0.8165, 1.4044, -0.5720, -0.5585, 0.5073, 0.8963, 1.1749, 0.2317, -0.3034,
        0.6418, -0.3841, 0.1446, 0.8489, 0.8137, -0.4105, 0.8105, -1.3586,
    ];
    const KY: [f64; 40] = [
        0.2247, -0.3042, 0.5043, 1.4535, -0.1108, -0.0848, 0.5737, -0.8745, 1.5106, -0.3778,
        -0.1653, 0.0670, 1.1093, -2.0171, 0.3239, 1.2860, -1.4404, 1.2230, 0.4424, 0.1354, -1.5561,
        -0.7617, 1.4101, -0.5054, 1.5702, -0.0184, -0.0037, -0.4206, 0.3890, -0.9830, 0.5311,
        0.9519, 1.6406, -1.0129, -0.1247, -0.0295, 0.1230, -0.3533, -0.9709, 0.0493,
    ];

    #[test]
    fn test_validate_kendall_tie_free_against_r() {
        for (alt, p) in [
            (Alternative::TwoSided, 0.021718583114218416),
            (Alternative::Less, 0.9891407084428908),
            (Alternative::Greater, 0.010859291557109208),
        ] {
            let r = cor_test(&KX, &KY, CorrelationMethod::Kendall, alt, 0.95).unwrap();
            assert!((r.estimate - 0.2538461538461539).abs() < 1e-12);
            assert!((r.statistic - 2.2952538084597247).abs() < 1e-12);
            assert!((r.p_value - p).abs() < 1e-12, "{alt:?}: {}", r.p_value);
        }
        assert_eq!(kendall_s(&KX, &KY).unwrap(), 198); // R: S = 2*T - n(n-1)/2 = 2*489 - 780
    }

    fn lcg(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s >> 33
        }
    }

    #[test]
    fn test_kendall_knight_matches_pairwise_reference_with_ties() {
        let mut next = lcg(42);
        let x: Vec<f64> = (0..500).map(|_| (next() % 21) as f64).collect();
        let y: Vec<f64> = (0..500)
            .map(|i| (next() % 13) as f64 + 0.5 * x[i])
            .collect();
        assert_eq!(kendall_s(&x, &y).unwrap(), 55869); // also computed by the planning port of the old loop
        for alt in [
            Alternative::TwoSided,
            Alternative::Less,
            Alternative::Greater,
        ] {
            let new = cor_test(&x, &y, CorrelationMethod::Kendall, alt, 0.95).unwrap();
            let old = kendall_test_pairwise_reference(&x, &y, alt, 0.95).unwrap();
            assert_eq!(
                (new.estimate, new.statistic, new.p_value),
                (old.estimate, old.statistic, old.p_value)
            );
        }
    }

    #[test]
    fn test_kendall_s_scales_and_counts_ties() {
        // 200,000 points, x tied in pairs, y strictly increasing: S = n(n-1)/2 - n/2
        let n = 200_000usize;
        let x: Vec<f64> = (0..n).map(|i| (i / 2) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert_eq!(kendall_s(&x, &y).unwrap(), 19_999_900_000 - 100_000);
    }

    #[test]
    fn test_kendall_variance_does_not_overflow() {
        // n(n-1)(2n+5) = 18522013229989500000 > u64::MAX; exact integer, then f64, then / 18
        assert_eq!(kendall_var_s_no_ties(2_100_000), 1.0290007349994166e18);
        assert_eq!(kendall_var_s_no_ties(40), 7366.666666666667);
    }

    #[test]
    fn test_pearson_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert!((result.estimate - 1.0).abs() < 1e-10);
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_pearson_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert!((result.estimate + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_uncorrelated() {
        // Data that should have low correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 10.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        // Just check it runs and p-value is reasonable
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_spearman() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Spearman,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert!((result.estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_kendall() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Kendall,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert!((result.estimate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_interval() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.2, 2.1, 2.9, 4.1, 4.9, 6.2, 6.8, 8.1, 9.0, 10.1];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();

        assert!(result.conf_low.is_some());
        assert!(result.conf_high.is_some());

        let low = result.conf_low.unwrap();
        let high = result.conf_high.unwrap();

        assert!(low < result.estimate);
        assert!(high > result.estimate);
        assert!(low >= -1.0);
        assert!(high <= 1.0);
    }

    #[test]
    fn test_one_sided() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result_greater = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::Greater,
            0.95,
        )
        .unwrap();
        let result_less =
            cor_test(&x, &y, CorrelationMethod::Pearson, Alternative::Less, 0.95).unwrap();

        // For perfect positive correlation, greater should have small p-value
        assert!(
            result_greater.p_value < 0.01,
            "p_value for greater: {}",
            result_greater.p_value
        );
        // Less should have large p-value (close to 1)
        assert!(
            result_less.p_value > 0.9,
            "p_value for less: {}",
            result_less.p_value
        );
    }

    #[test]
    fn test_input_validation() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_rank_data() {
        let data = vec![3.0, 1.0, 2.0];
        let ranks = rank_data(&data);
        assert_eq!(ranks, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank_data_ties() {
        let data = vec![1.0, 2.0, 2.0, 3.0];
        let ranks = rank_data(&data);
        // Tied values get average rank: 2.5
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    #[test]
    fn test_validate_pearson_correlation() {
        // R: x <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        // R: y <- c(1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0)
        // R: cor.test(x, y, method = "pearson")
        // R: cor = 0.999071, t = 65.555425, p < 0.0001
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();

        let expected_cor = 0.999071;
        assert!(
            (result.estimate - expected_cor).abs() < 0.001,
            "Pearson cor mismatch: Rust={:.6}, R={:.6}",
            result.estimate,
            expected_cor
        );

        // t-statistic should be large
        assert!(
            result.statistic > 50.0,
            "t-statistic should be large: {:.2}",
            result.statistic
        );

        // p-value should be very small
        assert!(
            result.p_value < 0.001,
            "p-value should be small: {:.6}",
            result.p_value
        );

        // df should be n - 2 = 8
        assert_eq!(result.df, Some(8.0), "df should be 8");
    }

    #[test]
    fn test_validate_pearson_confidence_interval() {
        // R: cor.test(x, y) conf.int: [0.995917, 0.999789]
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();

        let expected_low = 0.995917;
        let expected_high = 0.999789;

        assert!(result.conf_low.is_some(), "conf_low should be Some");
        assert!(result.conf_high.is_some(), "conf_high should be Some");

        let conf_low = result.conf_low.unwrap();
        let conf_high = result.conf_high.unwrap();

        // Confidence interval should contain the estimate
        assert!(
            conf_low < result.estimate && result.estimate < conf_high,
            "CI should contain estimate"
        );

        // CI bounds should be close to R's values
        assert!(
            (conf_low - expected_low).abs() < 0.01,
            "conf_low mismatch: Rust={:.6}, R={:.6}",
            conf_low,
            expected_low
        );
        assert!(
            (conf_high - expected_high).abs() < 0.01,
            "conf_high mismatch: Rust={:.6}, R={:.6}",
            conf_high,
            expected_high
        );
    }

    #[test]
    fn test_validate_spearman_rho() {
        // R: cor.test(x, y, method = "spearman")
        // R: rho = 1.0 (perfect monotonic relationship)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Spearman,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();

        let expected_rho = 1.0;
        assert!(
            (result.estimate - expected_rho).abs() < 0.01,
            "Spearman rho mismatch: Rust={:.6}, R={:.6}",
            result.estimate,
            expected_rho
        );

        // S statistic for perfect correlation should be 0
        assert!(
            result.statistic.abs() < 1e-10,
            "S statistic should be 0 for perfect monotonic: {:.4}",
            result.statistic
        );
    }

    #[test]
    fn test_validate_kendall_tau() {
        // R: cor.test(x, y, method = "kendall")
        // R: tau = 1.0 (all pairs concordant)
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0];

        let result = cor_test(
            &x,
            &y,
            CorrelationMethod::Kendall,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();

        let expected_tau = 1.0;
        assert!(
            (result.estimate - expected_tau).abs() < 0.01,
            "Kendall tau mismatch: Rust={:.6}, R={:.6}",
            result.estimate,
            expected_tau
        );

        // p-value should be very small for perfect correlation
        assert!(
            result.p_value < 0.001,
            "Kendall p-value should be small: {:.6}",
            result.p_value
        );
    }

    #[test]
    fn test_validate_correlation_names() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let pearson = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert_eq!(pearson.estimate_name, "cor");
        assert_eq!(pearson.statistic_name, "t");

        let spearman = cor_test(
            &x,
            &y,
            CorrelationMethod::Spearman,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert_eq!(spearman.estimate_name, "rho");
        assert_eq!(spearman.statistic_name, "S");

        let kendall = cor_test(
            &x,
            &y,
            CorrelationMethod::Kendall,
            Alternative::TwoSided,
            0.95,
        )
        .unwrap();
        assert_eq!(kendall.estimate_name, "tau");
        assert_eq!(kendall.statistic_name, "z");
    }

    #[test]
    fn test_validate_run_cor_test_wrapper() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = run_cor_test(&x, &y, "pearson", "two.sided", 0.95).unwrap();
        assert!(
            (result.estimate - 1.0).abs() < 1e-10,
            "Perfect correlation expected"
        );

        let result = run_cor_test(&x, &y, "spearman", "two_sided", 0.95).unwrap();
        assert!(
            (result.estimate - 1.0).abs() < 1e-10,
            "Perfect rank correlation expected"
        );

        let result = run_cor_test(&x, &y, "kendall", "twosided", 0.95).unwrap();
        assert!(
            (result.estimate - 1.0).abs() < 1e-10,
            "Perfect Kendall tau expected"
        );
    }
}
