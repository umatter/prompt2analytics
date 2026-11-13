//! Correlation test for paired samples.
//!
//! Implements cor.test from R stats: tests for association between paired samples
//! using Pearson's product moment correlation coefficient, Kendall's tau, or
//! Spearman's rho.
//!
//! The p-values follow `cor.test`'s defaults. Pearson: t on n - 2 degrees of freedom, with the
//! Fisher-z interval (one-sided intervals end at -1 or 1). Spearman: without ties and for
//! n < 1290 the exact distribution of S (all permutations for n <= 9, the AS 89 Edgeworth series
//! above), otherwise the t approximation. Kendall: tau-b; without ties and for n < 50 the exact
//! distribution of T, otherwise z with the tie-corrected variance (no continuity correction).

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy, refuse_non_finite};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::Alternative;
use crate::stats::ttest::one_sided_p;
use crate::traits::estimator::{
    normal_cdf, normal_p_value_two_sided, normal_quantile, t_test_p_value, validate_level,
    z_critical,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::OnceLock;

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

crate::option_enum!(
    CorrelationMethod,
    crate::options::Choices {
        option: "method",
        values: &[
            (&["pearson"], CorrelationMethod::Pearson),
            (&["kendall"], CorrelationMethod::Kendall),
            (&["spearman"], CorrelationMethod::Spearman),
        ],
    }
);

/// Below this a sum of squares has lost digits to subnormal numbers.
pub(crate) const MIN_ACCURATE_SQUARES: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// Refuse a constant column, whose correlation is undefined. `what` names it in the message.
fn refuse_constant(values: &[f64], what: &str) -> EconResult<()> {
    match values.first() {
        Some(&first) if values.iter().all(|&v| v == first) => {
            Err(EconError::InvalidSpecification {
                message: format!(
                    "{what} is constant (all {} values are {first}); the correlation is undefined \
                     (R's cor.test returns NA: 'the standard deviation is zero')",
                    values.len()
                ),
            })
        }
        _ => Ok(()),
    }
}

/// Result of a correlation test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorTestResult {
    /// The correlation coefficient estimate
    pub estimate: f64,
    /// Name of the estimate (cor, tau, or rho)
    pub estimate_name: String,
    /// Test statistic: t for Pearson; S = (n³ - n)(1 - rho)/6 for Spearman; for Kendall T (the
    /// number of concordant pairs, exact path) or z (normal path), as `statistic_name` says
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
    /// Rows dropped because of a missing value in either column (`run_cor_test_dataset`;
    /// nothing for slices).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
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
/// p-value, and confidence interval (for Pearson). `conf_level` must lie in (0, 1). A constant `x`
/// or `y` is refused (R's `cor.test` returns NA), as are infinite values.
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

    // Remove pairs with NaN; the input is used as it is when it has none
    let has_nan = x.iter().chain(y).any(|v| v.is_nan());
    let (x_clean, y_clean): (Cow<[f64]>, Cow<[f64]>) = if has_nan {
        let (xs, ys): (Vec<f64>, Vec<f64>) = x
            .iter()
            .zip(y)
            .filter(|(a, b)| !a.is_nan() && !b.is_nan())
            .map(|(a, b)| (*a, *b))
            .unzip();
        (Cow::Owned(xs), Cow::Owned(ys))
    } else {
        (Cow::Borrowed(x), Cow::Borrowed(y))
    };

    let n = x_clean.len();
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "correlation test".to_string(),
        });
    }

    validate_level(conf_level, "conf_level")?;

    // Missing values are gone; an infinite value would make r NaN.
    refuse_non_finite(&x_clean, "x")?;
    refuse_non_finite(&y_clean, "y")?;
    refuse_constant(&x_clean, "x")?;
    refuse_constant(&y_clean, "y")?;

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
    // The columns are not constant here (`cor_test` refuses that), so only their range is left.
    if r.is_nan() {
        return Err(EconError::InvalidSpecification {
            message: "the squared deviations of x or y do not fit a double (overflow or \
                      underflow); rescale the column"
                .to_string(),
        });
    }

    // Test statistic: t = r * sqrt((n-2) / (1-r²)), infinite for |r| = 1
    let df = (n - 2) as f64;
    let t_stat = t_from_r(r, df);
    let p_value = t_tail_p(t_stat, df, alternative);

    // Confidence interval using Fisher's z-transformation
    // z = 0.5 * ln((1+r)/(1-r)) = atanh(r)
    let (conf_low, conf_high) = if n >= 4 {
        let z = 0.5 * ((1.0 + r) / (1.0 - r)).ln();
        let se_z = 1.0 / ((n - 3) as f64).sqrt();

        // The open end of a one-sided interval is -1 or 1 itself, not tanh(-1) or tanh(1)
        let (r_low, r_high) = match alternative {
            Alternative::TwoSided => {
                let z_crit = z_critical(conf_level)?;
                ((z - z_crit * se_z).tanh(), (z + z_crit * se_z).tanh())
            }
            Alternative::Less => {
                let z_crit = normal_quantile(conf_level)?;
                (-1.0, (z + z_crit * se_z).tanh())
            }
            Alternative::Greater => {
                let z_crit = normal_quantile(conf_level)?;
                ((z - z_crit * se_z).tanh(), 1.0)
            }
        };
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
        missing: MissingSummary::none(n),
    })
}

/// Spearman's rank correlation test.
///
/// `statistic` is R's S = (n³ - n)(1 - rho)/6. The p-value follows `cor.test`'s default: without
/// ties and for n < 1290 the exact null distribution of S (all n! permutations for n <= 9, the
/// AS 89 Edgeworth series for larger n); with ties or for n >= 1290 the t approximation
/// t = rho sqrt((n - 2)/(1 - rho²)) on n - 2 degrees of freedom.
fn spearman_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let n = x.len();

    // Rank the data
    let x_ranks = crate::stats::ranks::midrank_values(x)?;
    let y_ranks = crate::stats::ranks::midrank_values(y)?;

    // Spearman's rho is Pearson correlation of ranks
    let rho = pearson_correlation(&x_ranks, &y_ranks);
    let nf = n as f64;
    let n3n = (nf * nf - 1.0) * nf;
    let s_stat = n3n * (1.0 - rho) / 6.0;

    let p_value = if n < SPEARMAN_EXACT_MAX_N && !has_ties(&x_ranks) && !has_ties(&y_ranks) {
        // R: pspearman(q, n, lower.tail) = prho(n, round(q) + 2 lower.tail, lower.tail)
        let q = s_stat.round();
        let lower = || spearman_tail(n, q + 2.0, true);
        let upper = || spearman_tail(n, q, false);
        match alternative {
            Alternative::TwoSided => {
                let p = if s_stat > n3n / 6.0 {
                    upper()?
                } else {
                    lower()?
                };
                (2.0 * p).min(1.0)
            }
            Alternative::Greater => lower()?,
            Alternative::Less => upper()?,
        }
    } else {
        let df = nf - 2.0;
        t_tail_p(t_from_r(rho, df), df, alternative)
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
        missing: MissingSummary::none(n),
    })
}

/// R's `cor.test` uses the exact Spearman distribution below this sample size (without ties).
const SPEARMAN_EXACT_MAX_N: usize = 1290;

/// Largest n whose Spearman distribution is counted over all n! permutations (R's `n_small`).
const SPEARMAN_ENUMERATION_MAX_N: usize = 9;

/// Whether midranks contain a tie: Σ(2r)² equals 4 n(n+1)(2n+1)/6 exactly without ties and is
/// smaller with them (the values are integers, exact in f64 for the n this is asked for).
fn has_ties(ranks: &[f64]) -> bool {
    let n = ranks.len() as f64;
    let sum: f64 = ranks.iter().map(|r| (2.0 * r) * (2.0 * r)).sum();
    sum < 2.0 * n * (n + 1.0) * (2.0 * n + 1.0) / 3.0
}

/// t statistic of a correlation r on `df` degrees of freedom; ±∞ for |r| = 1.
fn t_from_r(r: f64, df: f64) -> f64 {
    if (1.0 - r * r).abs() < 1e-15 {
        if r > 0.0 {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        }
    } else {
        r * (df / (1.0 - r * r)).sqrt()
    }
}

/// p-value of a t statistic for `alternative`. An infinite t (|r| = 1) gets the direction-aware
/// 0 or 1, as R's `pt`.
fn t_tail_p(t: f64, df: f64, alternative: Alternative) -> f64 {
    if t.is_infinite() {
        let positive = t > 0.0;
        match alternative {
            Alternative::TwoSided => 0.0,
            Alternative::Less => f64::from(u8::from(positive)),
            Alternative::Greater => f64::from(u8::from(!positive)),
        }
    } else {
        one_sided_p(t_test_p_value(t, df), t, alternative)
    }
}

/// AS 89 (R's `prho`): P(S >= is) (upper) or P(S < is) (lower) for S = Σ(i - π(i))² under a
/// uniformly random permutation π of n >= 2 elements. Exact for n <= 9, Edgeworth series above.
fn spearman_tail(n: usize, is: f64, lower: bool) -> EconResult<f64> {
    let upper_one = if lower { 0.0 } else { 1.0 };
    if is <= 0.0 {
        return Ok(upper_one);
    }
    let nf = n as f64;
    let n3 = nf * (nf * nf - 1.0) / 3.0; // the largest S
    if is > n3 {
        return Ok(1.0 - upper_one);
    }
    if n <= SPEARMAN_ENUMERATION_MAX_N {
        let counts = spearman_upper_counts(n);
        // S takes integer values 0..=n3; `is` is an integer here (round(q) or round(q) + 2)
        let idx = is.ceil() as usize;
        let at_least = counts.get(idx).copied().unwrap_or(0);
        let total = counts[0];
        let k = if lower { total - at_least } else { at_least };
        return Ok(f64::from(k) / f64::from(total));
    }
    const C: [f64; 12] = [
        0.2274, 0.2531, 0.1745, 0.0758, 0.1033, 0.3932, 0.0879, 0.0151, 0.0072, 0.0831, 0.0131,
        4.6e-4,
    ];
    let b = 1.0 / nf;
    let x = (6.0 * (is - 1.0) * b / (nf * nf - 1.0) - 1.0) * (1.0 / b - 1.0).sqrt();
    let y = x * x;
    let u = x
        * b
        * (C[0]
            + b * (C[1] + C[2] * b)
            + y * (-C[3] + b * (C[4] + C[5] * b)
                - y * b * (C[6] + C[7] * b - y * (C[8] - C[9] * b + y * b * (C[10] - C[11] * y)))));
    let correction = u * (-y / 2.0).exp();
    let p = if lower {
        normal_cdf(x) - correction
    } else {
        correction + normal_cdf(-x)
    };
    Ok(p.clamp(0.0, 1.0))
}

/// For n <= 9: `counts[s]` = number of the n! permutations with S = Σ(i - π(i))² >= s, for
/// s in 0..=(n³ - n)/3 (so `counts[0]` = n!). Counted once per n (Heap's algorithm, S updated in
/// O(1) per swap) and cached.
fn spearman_upper_counts(n: usize) -> &'static [u32] {
    static TABLES: [OnceLock<Vec<u32>>; SPEARMAN_ENUMERATION_MAX_N + 1] =
        [const { OnceLock::new() }; SPEARMAN_ENUMERATION_MAX_N + 1];
    TABLES[n].get_or_init(|| {
        let max_s = n * (n * n - 1) / 3;
        let mut freq = vec![0u32; max_s + 1];
        let mut perm: Vec<usize> = (0..n).collect();
        let mut s = 0usize; // identity: S = 0
        freq[0] += 1;
        let d2 = |i: usize, v: usize| (i.abs_diff(v)) * (i.abs_diff(v));
        // Heap's algorithm, iterative
        let mut c = vec![0usize; n];
        let mut i = 1;
        while i < n {
            if c[i] < i {
                let j = if i % 2 == 0 { 0 } else { c[i] };
                s -= d2(j, perm[j]) + d2(i, perm[i]);
                perm.swap(j, i);
                s += d2(j, perm[j]) + d2(i, perm[i]);
                freq[s] += 1;
                c[i] += 1;
                i = 1;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        // cumulative from the top: counts[s] = #{S >= s}
        let mut acc = 0u32;
        for f in freq.iter_mut().rev() {
            acc += *f;
            *f = acc;
        }
        freq
    })
}

/// Kendall's tau correlation test.
///
/// `estimate` is tau-b = S / sqrt((n0 - n1)(n0 - n2)) (tau-a without ties). As R's `cor.test`
/// by default: without ties and for n < 50 the exact null distribution of T, the number of
/// concordant pairs (`statistic` T); otherwise z = S / sqrt(var S) with the tie-corrected variance
/// and no continuity correction (`statistic` z).
fn kendall_test(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let n = x.len();

    // S = concordant - discordant pairs (Knight's O(n log n) algorithm)
    let counts = kendall_s(x, y)?;
    let n0 = tie_pairs(n)?;
    let s = counts.s;
    let tau = s as f64 / ((n0 - counts.n1) as f64 * (n0 - counts.n2) as f64).sqrt();

    let (statistic, statistic_name, p_value) =
        if n < KENDALL_EXACT_MAX_N && counts.n1 == 0 && counts.n2 == 0 {
            // T = (S + n0)/2 concordant pairs
            let t_obs = u64::try_from((i128::from(s) + i128::from(n0)) / 2).unwrap_or(0);
            let cdf = kendall_exact_cdf(n)?;
            let at_most = |k: u64| cdf[usize::try_from(k).unwrap_or(usize::MAX).min(cdf.len() - 1)];
            let less = at_most(t_obs);
            // P(T >= t) = P(T <= n0 - t) by symmetry
            let greater = at_most(n0 - t_obs);
            let p = match alternative {
                Alternative::TwoSided => {
                    let tail = if 2 * t_obs > n0 { greater } else { less };
                    (2.0 * tail).min(1.0)
                }
                Alternative::Less => less,
                Alternative::Greater => greater,
            };
            (t_obs as f64, "T", p)
        } else {
            let var_s = kendall_var_s(n, &counts.x_ties, &counts.y_ties);
            let z = s as f64 / var_s.sqrt();
            (
                z,
                "z",
                one_sided_p(normal_p_value_two_sided(z), z, alternative),
            )
        };

    Ok(CorTestResult {
        estimate: tau,
        estimate_name: "tau".to_string(),
        statistic,
        statistic_name: statistic_name.to_string(),
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
        missing: MissingSummary::none(n),
    })
}

/// R's `cor.test` uses the exact Kendall distribution below this sample size (without ties).
const KENDALL_EXACT_MAX_N: usize = 50;

/// `cdf[k]` = P(T <= k) for T the number of concordant pairs (equivalently, of inversions) of a
/// uniformly random permutation of n elements, k in 0..=n(n-1)/2. Mahonian recursion
/// P_m(k) = (1/m) Σ_{i<m} P_{m-1}(k - i) with sliding sums, O(n · n(n-1)/2); `n` < 50.
fn kendall_exact_cdf(n: usize) -> EconResult<Vec<f64>> {
    let n0 = n * n.saturating_sub(1) / 2;
    let mut p = resource::try_vec(n0 + 1, 0.0f64, "Kendall exact distribution")?;
    let mut next = resource::try_vec(n0 + 1, 0.0f64, "Kendall exact distribution")?;
    p[0] = 1.0;
    let mut top = 0usize; // largest k with P_{m-1}(k) > 0
    for m in 2..=n {
        resource::checkpoint()?;
        let new_top = top + m - 1;
        let inv_m = 1.0 / m as f64;
        let mut window = 0.0;
        for k in 0..=new_top {
            if k <= top {
                window += p[k];
            }
            if k >= m {
                window -= p[k - m];
            }
            next[k] = window * inv_m;
        }
        std::mem::swap(&mut p, &mut next);
        top = new_top;
    }
    let mut acc = 0.0;
    for v in p.iter_mut() {
        acc += *v;
        *v = acc.min(1.0);
    }
    Ok(p)
}

/// Sums over the tie groups (sizes t) of one variable, exact in `u128`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TieSums {
    /// Σ t(t-1)
    t1: u128,
    /// Σ t(t-1)(t-2)
    t2: u128,
    /// Σ t(t-1)(2t+5)
    t5: u128,
}

impl TieSums {
    fn add(&mut self, t: usize) {
        let t = t as u128;
        if t > 1 {
            self.t1 += t * (t - 1);
            self.t2 += t * (t - 1) * (t - 2);
            self.t5 += t * (t - 1) * (2 * t + 5);
        }
    }
}

/// Kendall's S and the tie counts behind tau-b and the variance of S.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct KendallCounts {
    /// concordant - discordant pairs
    s: i64,
    /// pairs tied in x (including those tied in both)
    n1: u64,
    /// pairs tied in y (including those tied in both)
    n2: u64,
    x_ties: TieSums,
    y_ties: TieSums,
}

/// Variance of Kendall's S under independence with ties (R's `cor.test`, continuity = FALSE):
/// (v0 - vt - vu)/18 + v1/(2n(n-1)) + v2/(9n(n-1)(n-2)).
fn kendall_var_s(n: usize, x: &TieSums, y: &TieSums) -> f64 {
    let nn = n as u128;
    let v0 = nn * nn.saturating_sub(1) * (2 * nn + 5);
    // exact: each tie sum is at most v0 (every pair tied)
    let base = (v0 as i128 - x.t5 as i128 - y.t5 as i128) as f64 / 18.0;
    let nf = n as f64;
    let v1 = x.t1 as f64 * y.t1 as f64 / (2.0 * nf * (nf - 1.0));
    let v2 = x.t2 as f64 * y.t2 as f64 / (9.0 * nf * (nf - 1.0) * (nf - 2.0));
    base + v1 + v2
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
/// The tie-group sums of both variables are collected in the same scans.
fn kendall_s(x: &[f64], y: &[f64]) -> EconResult<KendallCounts> {
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
    let mut x_ties = TieSums::default();
    let mut i = 0;
    while i < n {
        let xi = key(x[order[i]]);
        let mut j = i + 1;
        while j < n && key(x[order[j]]) == xi {
            j += 1;
        }
        n1 += tie_pairs(j - i)?;
        x_ties.add(j - i);
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
    let mut y_ties = TieSums::default();
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && a[j] == a[i] {
            j += 1;
        }
        n2 += tie_pairs(j - i)?;
        y_ties.add(j - i);
        i = j;
    }

    let s = i128::from(n0) - i128::from(n1) - i128::from(n2) + i128::from(n3)
        - 2 * i128::from(inversions);
    let s = i64::try_from(s).map_err(|_| {
        EconError::ResourceLimit("Kendall's tau: the statistic does not fit in 64 bits".to_string())
    })?;
    Ok(KendallCounts {
        s,
        n1,
        n2,
        x_ties,
        y_ties,
    })
}

/// Pearson correlation coefficient. NaN for a constant column and for squares a double cannot hold
/// accurately (infinite, or lost to underflow).
pub(crate) fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
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

    let accurate = |sum: f64| (MIN_ACCURATE_SQUARES..=f64::MAX).contains(&sum);
    if !accurate(sum_x2) || !accurate(sum_y2) || !sum_xy.is_finite() {
        return f64::NAN;
    }

    // The product of the sums can overflow where each sum fits; rounding can leave |r| just above 1.
    (sum_xy / (sum_x2.sqrt() * sum_y2.sqrt())).clamp(-1.0, 1.0)
}

/// Run correlation test from Dataset columns (MCP-friendly wrapper).
pub fn run_cor_test(
    x: &[f64],
    y: &[f64],
    method: &str,
    alternative: &str,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let method = method.parse::<CorrelationMethod>()?;
    let alternative = alternative.parse::<Alternative>()?;

    cor_test(x, y, method, alternative, conf_level)
}

/// Run `cor.test` on two dataset columns.
///
/// Rows with a missing value (null or NaN) in either column are dropped, as R's `cor.test`
/// does (`complete.cases(x, y)`), and reported in `missing`. Integer, decimal and boolean
/// columns are accepted; a text column is refused (`EconError::NonNumericColumn`), as are
/// infinite values (`EconError::NonFiniteValues`). `method`, `alternative` and `conf_level`
/// are as in [`run_cor_test`]. The MCP `hypothesis_cor_test` tool calls this.
pub fn run_cor_test_dataset(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    method: &str,
    alternative: &str,
    conf_level: f64,
) -> EconResult<CorTestResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numerics(&[x_col, y_col])
        .na(NaPolicy::Omit)
        .extract()?;
    refuse_constant(cols.numeric(x_col), &format!("column '{x_col}'"))?;
    refuse_constant(cols.numeric(y_col), &format!("column '{y_col}'"))?;
    let mut result = run_cor_test(
        cols.numeric(x_col),
        cols.numeric(y_col),
        method,
        alternative,
        conf_level,
    )?;
    result.missing = cols.missing().clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: the O(n^2) all-pairs count of S = concordant - discordant and of the pairs
    /// tied in x and in y (a pair tied in both counts in both, as R's n1 and n2).
    fn kendall_pairwise_reference(x: &[f64], y: &[f64]) -> (i64, u64, u64) {
        let n = x.len();
        let (mut s, mut ties_x, mut ties_y) = (0i64, 0u64, 0u64);
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = x[j] - x[i];
                let dy = y[j] - y[i];
                if dx == 0.0 {
                    ties_x += 1;
                }
                if dy == 0.0 {
                    ties_y += 1;
                }
                if dx != 0.0 && dy != 0.0 {
                    s += if dx * dy > 0.0 { 1 } else { -1 };
                }
            }
        }
        (s, ties_x, ties_y)
    }

    // R 4.3.3: set.seed(20261021); x <- round(rnorm(40), 4); y <- round(0.6 * x + rnorm(40, sd = 0.8), 4)
    // (set kd40 of case stat_cor_test_t4)
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

    /// Without ties: the exact T for n < 50 (R's default), z without continuity correction for
    /// n >= 50. Fails before: z with a continuity correction always.
    #[test]
    fn test_validate_kendall_tie_free_against_r() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_cor_test_t4");
        let mut check = Check::new(&case);
        for (set, name) in [
            ("perm", "T"),
            ("kd40", "T"),
            ("perf7", "T"),
            ("perf12", "T"),
            ("k60", "z"),
        ] {
            let (x, y) = (t4(&ds, set, "x"), t4(&ds, set, "y"));
            let n = x.len();
            for (alt, a) in ALTS {
                let r = cor_test(&x, &y, CorrelationMethod::Kendall, alt, 0.95).unwrap();
                assert_eq!(r.statistic_name, name, "{set}");
                check.scalar(&format!("kendall_{set}_{a}_p"), r.p_value, Tol::p_value(n));
                if alt == Alternative::TwoSided {
                    let stat_tol = if name == "T" {
                        Tol::exact()
                    } else {
                        Tol::stat(n)
                    };
                    check
                        .scalar(&format!("kendall_{set}_tau"), r.estimate, Tol::coef(n))
                        .scalar(&format!("kendall_{set}_stat"), r.statistic, stat_tol)
                        .scalar(
                            &format!("kendall_{set}_is_T"),
                            f64::from(u8::from(name == "T")),
                            Tol::exact(),
                        );
                }
            }
        }
        check.finish();
        let r = cor_test(
            &KX,
            &KY,
            CorrelationMethod::Kendall,
            Alternative::TwoSided,
            0.95,
        );
        assert_eq!(r.unwrap().statistic, 489.0);
        assert_eq!(kendall_s(&KX, &KY).unwrap().s, 198); // S = 2 T - n(n-1)/2 = 2*489 - 780
    }

    /// With ties: tau-b and z with the tie-corrected variance, no continuity correction.
    /// Fails before: tau-a and the tie-free variance.
    #[test]
    fn test_validate_kendall_ties_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_cor_test_t4");
        let mut check = Check::new(&case);
        for set in ["perft", "tie12", "t60"] {
            let (x, y) = (t4(&ds, set, "x"), t4(&ds, set, "y"));
            let n = x.len();
            for (alt, a) in ALTS {
                let r = cor_test(&x, &y, CorrelationMethod::Kendall, alt, 0.95).unwrap();
                assert_eq!(r.statistic_name, "z", "{set}");
                check.scalar(&format!("kendall_{set}_{a}_p"), r.p_value, Tol::p_value(n));
                if alt == Alternative::TwoSided {
                    check
                        .scalar(&format!("kendall_{set}_tau"), r.estimate, Tol::coef(n))
                        .scalar(&format!("kendall_{set}_stat"), r.statistic, Tol::stat(n));
                }
            }
        }
        check.finish();
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
        let counts = kendall_s(&x, &y).unwrap();
        assert_eq!(counts.s, 55869); // also computed by a port of the old pairwise loop
        let (s, ties_x, ties_y) = kendall_pairwise_reference(&x, &y);
        assert_eq!((counts.s, counts.n1, counts.n2), (s, ties_x, ties_y));
    }

    #[test]
    fn test_kendall_s_scales_and_counts_ties() {
        // 200,000 points, x tied in pairs, y strictly increasing: S = n(n-1)/2 - n/2
        let n = 200_000usize;
        let x: Vec<f64> = (0..n).map(|i| (i / 2) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        assert_eq!(kendall_s(&x, &y).unwrap().s, 19_999_900_000 - 100_000);
    }

    #[test]
    fn test_kendall_variance_does_not_overflow() {
        let none = TieSums::default();
        // n(n-1)(2n+5) = 18522013229989500000 > u64::MAX; exact integer, then f64, then / 18
        assert_eq!(
            kendall_var_s(2_100_000, &none, &none),
            1.0290007349994166e18
        );
        assert_eq!(kendall_var_s(40, &none, &none), 7366.666666666667);
        // n = 2^31 in two tie groups per variable: every term stays finite and the variance positive
        let n = 1usize << 31;
        let mut half = TieSums::default();
        half.add(n / 2);
        half.add(n / 2);
        let v = kendall_var_s(n, &half, &half);
        assert!(v.is_finite() && v > 0.0, "{v}");
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
        // exact path (n = 5): P(S = 0) = 1/5!
        assert!(
            (result.p_value - 2.0 / 120.0).abs() < 1e-15,
            "{}",
            result.p_value
        );
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

        // r = 0.99907 > tanh(1): the one-sided interval contains it and its open end is -1 or 1
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        let y = [1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0];
        let less = cor_test(&x, &y, CorrelationMethod::Pearson, Alternative::Less, 0.95).unwrap();
        let greater = cor_test(
            &x,
            &y,
            CorrelationMethod::Pearson,
            Alternative::Greater,
            0.95,
        )
        .unwrap();
        assert_eq!(less.conf_low, Some(-1.0));
        assert!(less.conf_high.unwrap() > less.estimate);
        assert_eq!(greater.conf_high, Some(1.0));
        assert!(greater.conf_low.unwrap() < greater.estimate);
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
        let ranks = crate::stats::ranks::midranks(&data).unwrap().0;
        assert_eq!(ranks, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_rank_data_ties() {
        let data = vec![1.0, 2.0, 2.0, 3.0];
        let ranks = crate::stats::ranks::midranks(&data).unwrap().0;
        // Tied values get average rank: 2.5
        assert_eq!(ranks, vec![1.0, 2.5, 2.5, 4.0]);
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    /// Column `col` of set `set` of case `stat_cor_test_t4`.
    fn t4(ds: &Dataset, set: &str, col: &str) -> Vec<f64> {
        crate::stats::ranks::test_data::column_where(ds, col, &[("set", set)])
    }

    const ALTS: [(Alternative, &str); 3] = [
        (Alternative::TwoSided, "two"),
        (Alternative::Less, "less"),
        (Alternative::Greater, "greater"),
    ];

    /// r, t, df, two-sided p and the 95% and 90% intervals (R 4.3.3, case `stat_cor_test_t4`).
    #[test]
    fn test_validate_pearson_correlation() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_cor_test_t4");
        let mut check = Check::new(&case);
        for set in ["pear", "pear2"] {
            let (x, y) = (t4(&ds, set, "x"), t4(&ds, set, "y"));
            let n = x.len();
            let run = |level| {
                cor_test(
                    &x,
                    &y,
                    CorrelationMethod::Pearson,
                    Alternative::TwoSided,
                    level,
                )
                .unwrap()
            };
            let r = run(0.95);
            let r90 = run(0.90);
            assert_eq!(r.df, Some((n - 2) as f64));
            check
                .scalar(&format!("pearson_{set}_r"), r.estimate, Tol::coef(n))
                .scalar(&format!("pearson_{set}_t"), r.statistic, Tol::stat(n))
                .scalar(&format!("pearson_{set}_two_p"), r.p_value, Tol::p_value(n))
                .vector(
                    &format!("pearson_{set}_two_ci"),
                    &[r.conf_low.unwrap(), r.conf_high.unwrap()],
                    Tol::stat(n),
                )
                .vector(
                    &format!("pearson_{set}_ci90"),
                    &[r90.conf_low.unwrap(), r90.conf_high.unwrap()],
                    Tol::stat(n),
                );
        }
        check.finish();
    }

    /// One-sided p and intervals: the open end is -1 (`less`) or 1 (`greater`), as R.
    /// Fails before: the open end went through tanh(±1) = ±0.7616, below r for `pear`.
    #[test]
    fn test_validate_pearson_confidence_interval() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_cor_test_t4");
        let mut check = Check::new(&case);
        for set in ["pear", "pear2"] {
            let (x, y) = (t4(&ds, set, "x"), t4(&ds, set, "y"));
            let n = x.len();
            for (alt, a) in &ALTS[1..] {
                let r = cor_test(&x, &y, CorrelationMethod::Pearson, *alt, 0.95).unwrap();
                check
                    .scalar(&format!("pearson_{set}_{a}_p"), r.p_value, Tol::p_value(n))
                    .vector(
                        &format!("pearson_{set}_{a}_ci"),
                        &[r.conf_low.unwrap(), r.conf_high.unwrap()],
                        Tol::stat(n),
                    );
            }
        }
        check.finish();
    }

    /// rho, S and the three p-values on the exact (n <= 9), Edgeworth (n < 1290) and t (ties,
    /// n >= 1290) paths of R's cor.test. Fails before: t approximation always, S = Σd², p = 0
    /// for both one-sided tests at rho = 1.
    #[test]
    fn test_validate_spearman_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_cor_test_t4");
        let mut check = Check::new(&case);
        for set in [
            "perm", "s8", "s40", "s1300", "perf7", "perf12", "perft", "tie12",
        ] {
            let (x, y) = (t4(&ds, set, "x"), t4(&ds, set, "y"));
            let n = x.len();
            for (alt, a) in ALTS {
                let r = cor_test(&x, &y, CorrelationMethod::Spearman, alt, 0.95).unwrap();
                check.scalar(&format!("spearman_{set}_{a}_p"), r.p_value, Tol::p_value(n));
                if alt == Alternative::TwoSided {
                    check.scalar(&format!("spearman_{set}_rho"), r.estimate, Tol::coef(n));
                    if set.starts_with("perf") {
                        // R's S is a rounding residual of rho = 1 (1.2e-14 for perf7)
                        assert!(r.statistic.abs() < 1e-12, "{set}: S = {}", r.statistic);
                    } else {
                        check.scalar(&format!("spearman_{set}_S"), r.statistic, Tol::stat(n));
                    }
                }
            }
        }
        check.finish();
    }

    /// n = 9 is the largest n counted over all 9! permutations (R's n_small; AS 89 used 6):
    /// 368 permutations have S <= 12, the observed S of rho = 0.9.
    #[test]
    fn test_spearman_exact_enumeration_at_n9() {
        let x: Vec<f64> = (1..=9).map(f64::from).collect();
        let y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 9.0, 7.0, 8.0];
        let run = |alt| cor_test(&x, &y, CorrelationMethod::Spearman, alt, 0.95).unwrap();
        let total = 362_880.0;
        let two = run(Alternative::TwoSided);
        assert!((two.estimate - 0.9).abs() < 1e-15);
        assert!(
            (two.p_value - 2.0 * 368.0 / total).abs() < 1e-15,
            "{}",
            two.p_value
        );
        let greater = run(Alternative::Greater).p_value;
        assert!((greater - 368.0 / total).abs() < 1e-15, "{greater}");
        let less = run(Alternative::Less).p_value;
        assert!((less - 362_642.0 / total).abs() < 1e-15, "{less}");
    }

    #[test]
    fn test_spearman_rho_monotone_is_one() {
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
        // A perfect increasing relation is no evidence for a negative one
        let less = cor_test(&x, &y, CorrelationMethod::Spearman, Alternative::Less, 0.95).unwrap();
        assert_eq!(less.p_value, 1.0);
    }

    #[test]
    fn test_kendall_tau_concordant_pairs_is_one() {
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

    /// r does not depend on the scale; the product of the two sums leaves the range before either sum does.
    #[test]
    fn test_pearson_is_scale_free() {
        let x = [1.0, -2.3, 1.7, 3.1, -2.2, 1.4, 2.8];
        let y = [0.4, 2.3, -1.7, 1.1, 2.2, -1.4, 0.8];
        let run = |x: &[f64], y: &[f64]| {
            cor_test(
                x,
                y,
                CorrelationMethod::Pearson,
                Alternative::TwoSided,
                0.95,
            )
            .unwrap()
        };
        let base = run(&x, &y);
        for scale in [1e80, 1e100, 1e-100] {
            let scaled = run(&x.map(|v| v * scale), &y.map(|v| v * scale));
            assert!((scaled.estimate - base.estimate).abs() < 1e-12);
            assert!((scaled.p_value - base.p_value).abs() < 1e-10);
        }
        // Perfectly correlated data stay perfectly correlated, with r inside [-1, 1] and a defined CI.
        let big = x.map(|v| v * 1e100);
        let perfect = run(&big, &big);
        assert!((perfect.estimate - 1.0).abs() < 1e-12 && perfect.estimate <= 1.0);
        assert!(perfect.p_value < 1e-10);
        assert!(perfect.conf_low.unwrap() > 0.99);
    }

    /// Squares that overflow or underflow are refused, not an r of 0 or NaN in an `Ok`.
    #[test]
    fn test_pearson_refuses_squares_out_of_range() {
        let x = [1.0, -2.3, 1.7, 3.1, -2.2, 1.4, 2.8];
        let y = [0.4, 2.3, -1.7, 1.1, 2.2, -1.4, 0.8];
        for scale in [1e200, 1e-200] {
            for (a, b) in [(x.map(|v| v * scale), y), (x, y.map(|v| v * scale))] {
                let err = cor_test(
                    &a,
                    &b,
                    CorrelationMethod::Pearson,
                    Alternative::TwoSided,
                    0.95,
                )
                .unwrap_err();
                assert!(
                    matches!(err, EconError::InvalidSpecification { ref message } if message.contains("rescale")),
                    "{err}"
                );
            }
            // Ranks do not see the scale.
            let big = x.map(|v| v * scale);
            let spearman = cor_test(
                &big,
                &y,
                CorrelationMethod::Spearman,
                Alternative::TwoSided,
                0.95,
            );
            assert!(spearman.unwrap().estimate.is_finite());
        }
    }

    #[test]
    fn test_cor_test_names() {
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
        assert_eq!(kendall.statistic_name, "T"); // exact path: n < 50, no ties
    }

    #[test]
    fn test_run_cor_test_wrapper() {
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

    // ========================================================================
    // Dataset columns with missing values (case stat_two_sample_na,
    // validation/r/cases/stats_onesample.R)
    // ========================================================================

    /// cor.test(x, y) keeps complete pairs only (complete.cases(x, y)).
    #[test]
    fn test_validate_cor_test_two_sample_na() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let r = run_cor_test_dataset(&ds, "x", "y", "pearson", "two.sided", 0.95).unwrap();
        Check::new(&case)
            .scalar("cor_r", r.estimate, Tol::coef(6))
            .scalar("cor_t", r.statistic, Tol::stat(6))
            .scalar("cor_df", r.df.unwrap(), Tol::exact())
            .scalar("cor_p", r.p_value, Tol::p_value(6))
            .vector(
                "cor_ci",
                &[r.conf_low.unwrap(), r.conf_high.unwrap()],
                Tol::stat(6),
            )
            .scalar("n_pairs", r.n as f64, Tol::exact())
            .finish();
        assert_eq!(
            r.missing.by_column,
            vec![("x".to_string(), 15), ("y".to_string(), 15)]
        );
        assert_eq!(r.missing.n_dropped, 16);
    }

    #[test]
    fn test_validate_na_cor_test_columns() {
        use crate::data::Dataset;
        use polars::prelude::*;
        let (_case, ds) = crate::test_support::load_case("stat_two_sample_na");
        let mut df = ds.df().clone();
        for c in ["xi", "yi"] {
            let col = df
                .column(c)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .with_name(format!("{c}_f").into());
            df.with_column(col).unwrap();
        }
        let ds2 = Dataset::new(df);
        let a = run_cor_test_dataset(&ds2, "xi", "yi", "pearson", "two.sided", 0.95).unwrap();
        let b = run_cor_test_dataset(&ds2, "xi_f", "yi_f", "pearson", "two.sided", 0.95).unwrap();
        assert_eq!(a.n, 18);
        assert_eq!(a.estimate, b.estimate);
        assert!(matches!(
            run_cor_test_dataset(&ds, "x", "g", "pearson", "two.sided", 0.95),
            Err(EconError::NonNumericColumn { column }) if column == "g"
        ));
        assert!(matches!(
            run_cor_test_dataset(&ds, "x", "nope", "pearson", "two.sided", 0.95),
            Err(EconError::ColumnNotFound { .. })
        ));
        assert!(matches!(
            run_cor_test_dataset(&ds, "x", "y", "pearsn", "two.sided", 0.95),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    /// X-3: a constant column is refused by every method; r would be undefined, as R's cor.test finds.
    #[test]
    fn test_cor_test_constant_column() {
        let x = [1.0, 2.0, 3.5, 4.0, 5.2, 6.1];
        let c = [3.0; 6];
        for m in [
            CorrelationMethod::Pearson,
            CorrelationMethod::Spearman,
            CorrelationMethod::Kendall,
        ] {
            for alt in [
                Alternative::TwoSided,
                Alternative::Less,
                Alternative::Greater,
            ] {
                let e = cor_test(&x, &c, m, alt, 0.95).unwrap_err();
                assert!(
                    matches!(&e, EconError::InvalidSpecification { message } if message.starts_with("y is constant")),
                    "{m:?}: {e}"
                );
                let e = cor_test(&c, &x, m, alt, 0.95).unwrap_err();
                assert!(
                    matches!(&e, EconError::InvalidSpecification { message } if message.starts_with("x is constant")),
                    "{m:?}: {e}"
                );
            }
        }
        // constant after the NaN pair is dropped
        let y = [3.0, 3.0, f64::NAN, 3.0, 3.0, 3.0];
        assert!(
            cor_test(
                &x,
                &y,
                CorrelationMethod::Pearson,
                Alternative::TwoSided,
                0.95
            )
            .is_err()
        );
        // ±Inf is refused (r would be NaN)
        let xi = [1.0, 2.0, f64::INFINITY, 4.0, 5.0];
        let yi = [1.1, 2.3, 2.9, 4.2, 5.1];
        assert!(
            matches!(cor_test(&xi, &yi, CorrelationMethod::Pearson, Alternative::TwoSided, 0.95),
            Err(EconError::NonFiniteValues { ref column, count: 1 }) if column == "x")
        );
        // the dataset wrapper names the column
        let ds = crate::data::Dataset::new(
            polars::prelude::df! { "a" => x.to_vec(), "k" => c.to_vec() }.unwrap(),
        );
        let e = run_cor_test_dataset(&ds, "a", "k", "pearson", "two.sided", 0.95).unwrap_err();
        assert!(e.to_string().contains("'k' is constant"), "{e}");
    }

    #[test]
    fn test_cor_test_rejects_invalid_level() {
        let x = [1.0, 2.0, 3.5, 4.0, 5.2, 6.1];
        let y = [1.3, 2.1, 2.9, 4.4, 5.0, 6.6];
        for level in [f64::NAN, 95.0, 0.0, 1.0] {
            let e = cor_test(
                &x,
                &y,
                CorrelationMethod::Pearson,
                Alternative::TwoSided,
                level,
            )
            .unwrap_err();
            assert!(
                e.to_string()
                    .contains("conf_level must be strictly between 0 and 1"),
                "{e}"
            );
        }
    }

    #[test]
    fn test_correlation_method_parses_aliases_and_rejects_unknown() {
        assert_eq!(CorrelationMethod::choices().validate(), Ok(()));
        assert_eq!(
            "Spearman".parse::<CorrelationMethod>().unwrap(),
            CorrelationMethod::Spearman
        );
        let e = run_cor_test(
            &[1.0, 2.0, 3.0, 4.5],
            &[1.2, 1.9, 3.3, 4.1],
            "pearsons",
            "two.sided",
            0.95,
        )
        .unwrap_err();
        assert!(e.to_string().contains("unknown method 'pearsons'"), "{e}");
        assert!(
            run_cor_test(
                &[1.0, 2.0, 3.0, 4.5],
                &[1.2, 1.9, 3.3, 4.1],
                "kendall",
                "Two-Sided",
                0.95
            )
            .is_ok()
        );
    }
}
