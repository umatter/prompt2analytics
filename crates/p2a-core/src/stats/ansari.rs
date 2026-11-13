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
//!    - Exact: subset sums of the scores, on R's support (m+1)²%/%4 to that plus mn%/%2
//!    - Approximate: Normal with
//!      - Mean: m(N+2)/4 (N even) or m(N+1)²/(4N) (N odd)
//!      - Variance: mn(N+2)(N-2) / (48(N-1)) (N even) or mn(N+1)(3+N²) / (48N²) (N odd),
//!        with R's tie-corrected form when the pooled sample has ties
//!
//! H₀: Scale ratio σ_x/σ_y = 1
//! H₁: Scale ratio ≠ 1 (two-sided), > 1 (greater), or < 1 (less). A large AB means x is
//! concentrated in the middle, i.e. a smaller scale, so "less" is the upper tail of AB.
//!
//! The interval and estimate of the ratio of scales are Bauer's (1972), as `ansari.test`
//! computes them.

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::resource::{checked_len, checkpoint, try_vec, try_with_capacity};
use crate::stats::ranks::midranks;
use crate::stats::ttest::Alternative;
use crate::traits::SignificanceLevel;
use crate::traits::estimator::{
    normal_cdf, normal_p_value_two_sided, normal_quantile, validate_level,
};

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
    /// Point estimate of the ratio of scales (Bauer's estimator, as R)
    pub estimate: Option<f64>,
    /// R's warnings, joined with "; " (ties on a requested or default exact path, intervals
    /// that cannot be computed)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
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

        if let Some(ref warn) = self.warning {
            writeln!(f, "Warning: {warn}")?;
        }

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}

/// Perform Ansari-Bradley test for equality of scale parameters.
///
/// `exact = true` follows R's default (`exact = NULL`): the exact distribution when both
/// samples have fewer than 50 values and there are no ties. `exact = false` uses the normal
/// approximation. [`ansari_test_with`] takes R's `exact` argument as it is.
///
/// # Arguments
/// * `x` - First sample
/// * `y` - Second sample
/// * `alternative` - Direction of alternative hypothesis
/// * `exact` - `true`: R's default rule; `false`: normal approximation
/// * `conf_level` - Confidence level for Bauer's interval and estimate (optional)
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
    let exact = if exact { None } else { Some(false) };
    ansari_test_with(x, y, alternative, exact, conf_level)
}

/// R's warning when an exact p-value was asked for (or chosen) but the data have ties.
const EXACT_TIES: &str = "cannot compute exact p-value with ties";
/// R's companion warning for the interval.
const EXACT_CI_TIES: &str = "cannot compute exact confidence intervals with ties";
const DIFFER_IN_LOCATION: &str =
    "samples differ in location: cannot compute confidence set, returning NA";

/// Perform the Ansari-Bradley test with R's semantics of `exact`.
///
/// `exact = None` is R's `exact = NULL`: the exact distribution when both samples have
/// fewer than 50 finite values. Ties force the normal approximation (with R's tie-corrected
/// variance), and a requested or chosen exact test then carries R's warning. `conf_level`
/// asks for Bauer's (1972) confidence interval and estimate of the ratio of scales: exact on
/// the exact path, from the normal approximation otherwise.
///
/// # References
/// - R equivalent: `ansari.test(x, y, alternative, exact, conf.int = TRUE, conf.level)`
pub fn ansari_test_with(
    x: &[f64],
    y: &[f64],
    alternative: Alternative,
    exact: Option<bool>,
    conf_level: Option<f64>,
) -> EconResult<AnsariBradleyResult> {
    if let Some(level) = conf_level {
        validate_level(level, "conf_level")?;
    }
    let mut xs: Vec<f64> = try_with_capacity(x.len(), "Ansari-Bradley sample")?;
    xs.extend(x.iter().copied().filter(|v| v.is_finite()));
    let mut ys: Vec<f64> = try_with_capacity(y.len(), "Ansari-Bradley sample")?;
    ys.extend(y.iter().copied().filter(|v| v.is_finite()));
    let (m, n) = (xs.len(), ys.len());
    if m == 0 || n == 0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Both samples must have at least one finite observation".to_string(),
        });
    }
    let (ab_stat, has_ties) = ab_statistic(&xs, &ys, 1.0)?;
    let want_exact = exact.unwrap_or(m < 50 && n < 50);
    let use_exact = want_exact && !has_ties;
    let mut warnings: Vec<&'static str> = Vec::new();
    if conf_level.is_some() {
        // the intervals work on the sorted samples; the test itself does not need them
        xs.sort_by(f64::total_cmp);
        ys.sort_by(f64::total_cmp);
    }

    let mut interval = (None, None, None);
    let p_value;
    if use_exact {
        let dist = AnsariDist::new(m, n)?;
        let half = ((m + 1) * (m + 1) / 4) as f64 + ((m * n) / 2) as f64 / 2.0;
        p_value = match alternative {
            Alternative::TwoSided => {
                let p = if ab_stat > half {
                    dist.upper(ab_stat - 1.0)
                } else {
                    dist.lower(ab_stat)
                };
                (2.0 * p).min(1.0)
            }
            Alternative::Less => dist.upper(ab_stat - 1.0),
            Alternative::Greater => dist.lower(ab_stat),
        };
        if let Some(level) = conf_level {
            interval = bauer_exact_interval(&xs, &ys, &dist, level, alternative, &mut warnings)?;
        }
    } else {
        let z = ab_normalize(m, n, ab_stat, &xs, &ys, 1.0, has_ties)?;
        p_value = match alternative {
            Alternative::TwoSided => normal_p_value_two_sided(z),
            Alternative::Less => normal_cdf(-z),
            Alternative::Greater => normal_cdf(z),
        };
        if let Some(level) = conf_level {
            if !want_exact {
                interval = bauer_asymptotic_interval(&xs, &ys, level, alternative, &mut warnings)?;
            }
        }
        if want_exact && has_ties {
            warnings.push(EXACT_TIES);
            if conf_level.is_some() {
                warnings.push(EXACT_CI_TIES);
            }
        }
    }
    let warning = if warnings.is_empty() {
        None
    } else {
        Some(warnings.join("; "))
    };
    Ok(AnsariBradleyResult {
        test_name: "Ansari-Bradley test".to_string(),
        alternative,
        statistic: ab_stat,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        n_x: m,
        n_y: n,
        exact: use_exact,
        conf_level,
        conf_int_lower: interval.0,
        conf_int_upper: interval.1,
        estimate: interval.2,
        warning,
    })
}

/// AB of `x / sig` against `y` (scores `min(r, N + 1 - r)` of the midranks) and whether the
/// pooled sample has ties.
fn ab_statistic(x: &[f64], y: &[f64], sig: f64) -> EconResult<(f64, bool)> {
    let (ranks, sizes) = ab_ranks(x, y, sig)?;
    let nt = (x.len() + y.len()) as f64;
    let s = ranks[..x.len()].iter().map(|&r| r.min(nt + 1.0 - r)).sum();
    Ok((s, sizes.iter().any(|&t| t > 1)))
}

fn ab_ranks(x: &[f64], y: &[f64], sig: f64) -> EconResult<(Vec<f64>, Vec<usize>)> {
    let mut pooled: Vec<f64> = try_with_capacity(x.len() + y.len(), "Ansari-Bradley pooled")?;
    pooled.extend(x.iter().map(|&v| v / sig));
    pooled.extend_from_slice(y);
    midranks(&pooled)
}

/// R's `normalize`: the standardised AB of `x / sig` against `y`, with the tie-corrected
/// SIGMA from the scores of the pooled sample when it has ties.
fn ab_normalize(
    m: usize,
    n: usize,
    s: f64,
    x: &[f64],
    y: &[f64],
    sig: f64,
    ties: bool,
) -> EconResult<f64> {
    let (mf, nf) = (m as f64, n as f64);
    let nt = mf + nf;
    let even = (m + n).is_multiple_of(2);
    let z = if even {
        s - mf * (nt + 2.0) / 4.0
    } else {
        s - mf * (nt + 1.0).powi(2) / (4.0 * nt)
    };
    let sigma = if !ties {
        if even {
            (mf * nf * (nt + 2.0) * (nt - 2.0) / (48.0 * (nt - 1.0))).sqrt()
        } else {
            (mf * nf * (nt + 1.0) * (3.0 + nt * nt) / (48.0 * nt * nt)).sqrt()
        }
    } else {
        let (ranks, _) = ab_ranks(x, y, sig)?;
        // sum over the pooled scores of score^2 (rle of the sorted scores, lengths * values^2)
        let sq: f64 = ranks
            .iter()
            .map(|&r| {
                let a = r.min(nt + 1.0 - r);
                a * a
            })
            .sum();
        if even {
            (mf * nf * (16.0 * sq - nt * (nt + 2.0).powi(2)) / (16.0 * nt * (nt - 1.0))).sqrt()
        } else {
            (mf * nf * (16.0 * nt * sq - (nt + 1.0).powi(4)) / (16.0 * nt * nt * (nt - 1.0))).sqrt()
        }
    };
    Ok(z / sigma)
}

/// Exact null distribution of AB (sizes `m`, `n`): the counts of the subset sums of `m` of
/// the scores `min(i, N + 1 - i)`, i = 1..N, on R's support `l = (m + 1)^2 %/% 4` to
/// `u = l + m n %/% 2`, normalised to probabilities.
struct AnsariDist {
    l: usize,
    probs: Vec<f64>,
}

impl AnsariDist {
    fn new(m: usize, n: usize) -> EconResult<Self> {
        let nt = m + n;
        let l = (m + 1) * (m + 1) / 4;
        let u = l + checked_len(&[m, n], "Ansari-Bradley exact distribution")? / 2;
        let width = u + 1;
        let cells = checked_len(&[m + 1, width], "Ansari-Bradley exact distribution")?;
        // dp[j * width + s]: weight of the subsets of j scores with sum s
        let mut dp: Vec<f64> = try_vec(cells, 0.0f64, "Ansari-Bradley exact distribution")?;
        dp[0] = 1.0;
        for i in 1..=nt {
            checkpoint()?;
            let score = i.min(nt + 1 - i);
            for j in (1..=m.min(i)).rev() {
                let (lo, hi) = dp.split_at_mut(j * width);
                let prev = &lo[(j - 1) * width..];
                let cur = &mut hi[..width];
                for s in (score..width).rev() {
                    cur[s] += prev[s - score];
                }
            }
            // every cell counts subsets of the first i scores, so it is at most 2^i: rescaling
            // can only be needed once i passes 800 (2^800 < 1e250)
            if i >= 800 {
                let big = dp.iter().copied().fold(0.0f64, f64::max);
                if big > 1e250 {
                    dp.iter_mut().for_each(|v| *v *= 1e-250);
                }
            }
        }
        let row = &dp[m * width..(m + 1) * width];
        let total: f64 = row.iter().sum();
        let mut probs: Vec<f64> =
            try_with_capacity(width - l, "Ansari-Bradley exact distribution")?;
        probs.extend(row[l..].iter().map(|c| c / total));
        Ok(AnsariDist { l, probs })
    }

    fn max(&self) -> usize {
        self.l + self.probs.len() - 1
    }

    /// `pansari(q)`: P(AB <= floor(q + 1e-7)).
    fn lower(&self, q: f64) -> f64 {
        let q = (q + 1e-7).floor();
        if q < self.l as f64 {
            0.0
        } else if q >= self.max() as f64 {
            1.0
        } else {
            let k = q as usize - self.l;
            self.probs[..=k].iter().sum::<f64>().min(1.0)
        }
    }

    /// P(AB > floor(q + 1e-7)), summed directly over the upper tail.
    fn upper(&self, q: f64) -> f64 {
        let q = (q + 1e-7).floor();
        if q < self.l as f64 {
            1.0
        } else if q >= self.max() as f64 {
            0.0
        } else {
            let k = q as usize - self.l;
            self.probs[k + 1..].iter().sum::<f64>().min(1.0)
        }
    }

    /// `qansari(p)`: the smallest q with P(AB <= q) >= p.
    fn quantile(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return self.l as f64;
        }
        if p >= 1.0 {
            return self.max() as f64;
        }
        let mut cum = 0.0;
        for (k, pr) in self.probs.iter().enumerate() {
            cum += pr;
            if cum >= p {
                return (self.l + k) as f64;
            }
        }
        self.max() as f64
    }
}

type Bounds = (Option<f64>, Option<f64>, Option<f64>);

/// Bauer's exact confidence interval and estimate, as `ansari.test`: AB evaluated between
/// consecutive non-negative ratios `x_i / y_j` (and at 1.01 times the largest), the bounds
/// read off against `qansari(alpha / 2)` and `qansari(1 - alpha / 2)`.
fn bauer_exact_interval(
    xs: &[f64],
    ys: &[f64],
    dist: &AnsariDist,
    level: f64,
    alternative: Alternative,
    warnings: &mut Vec<&'static str>,
) -> EconResult<Bounds> {
    let cap = checked_len(&[xs.len(), ys.len()], "Ansari-Bradley ratios")?;
    let mut sigma: Vec<f64> = try_with_capacity(cap, "Ansari-Bradley ratios")?;
    for &xi in xs {
        for &yj in ys {
            let r = xi / yj;
            if r >= 0.0 {
                sigma.push(r);
            }
        }
    }
    if sigma.is_empty() {
        warnings.push("cannot compute confidence set, returning NA");
        return Ok((None, None, None));
    }
    sigma.sort_by(f64::total_cmp);
    let k = sigma.len();
    let mut absigma: Vec<f64> = try_with_capacity(k, "Ansari-Bradley ratios")?;
    for i in 0..k {
        if i.is_multiple_of(256) {
            checkpoint()?;
        }
        let at = if i + 1 < k {
            sigma[i] + (sigma[i + 1] - sigma[i]) / 2.0
        } else {
            sigma[k - 1] * 1.01
        };
        absigma.push(ab_statistic(xs, ys, at)?.0);
    }
    let alpha = 1.0 - level;
    let cci = |alpha: f64, warnings: &mut Vec<&'static str>| -> (Option<f64>, Option<f64>) {
        let qa = dist.quantile(alpha / 2.0);
        let qb = dist.quantile(1.0 - alpha / 2.0);
        // u >= 0 entries with the smallest u: the smallest such sigma; l > 0 entries with the
        // smallest l: the largest such sigma
        let pick = |q: f64, strict: bool, take_max: bool| -> Option<f64> {
            let mut best: Option<f64> = None;
            let mut out: Option<f64> = None;
            for (s, a) in sigma.iter().zip(&absigma) {
                let d = a - q;
                if (strict && d > 0.0) || (!strict && d >= 0.0) {
                    match best {
                        Some(b) if d > b => {}
                        Some(b) if d == b => {
                            out = Some(match out {
                                Some(o) if take_max => o.max(*s),
                                Some(o) => o.min(*s),
                                None => *s,
                            });
                        }
                        _ => {
                            best = Some(d);
                            out = Some(*s);
                        }
                    }
                }
            }
            out
        };
        let (Some(mut uci), Some(mut lci)) = (pick(qa, false, false), pick(qb, true, true)) else {
            warnings.push(DIFFER_IN_LOCATION);
            return (None, None);
        };
        if uci > lci {
            match (pick(qb, false, false), pick(qa, true, true)) {
                (Some(u), Some(l)) => {
                    uci = u;
                    lci = l;
                }
                _ => return (None, None),
            }
        }
        (Some(uci), Some(lci))
    };
    let (lower, upper) = match alternative {
        Alternative::TwoSided => cci(alpha, warnings),
        Alternative::Greater => (cci(alpha * 2.0, warnings).0, Some(f64::INFINITY)),
        Alternative::Less => (Some(0.0), cci(alpha * 2.0, warnings).1),
    };
    let q5 = dist.quantile(0.5);
    let mut sgr: Option<f64> = None;
    let mut sle: Option<f64> = None;
    for (s, a) in sigma.iter().zip(&absigma) {
        if a - q5 <= 0.0 {
            sgr = Some(sgr.map_or(*s, |g: f64| g.max(*s)));
        } else {
            sle = Some(sle.map_or(*s, |g: f64| g.min(*s)));
        }
    }
    let estimate = match (sle, sgr) {
        (Some(a), Some(b)) => Some((a + b) / 2.0),
        _ => None,
    };
    Ok((lower, upper, estimate))
}

/// Positive ratios `|x_i| / |y_j|` of same-signed non-zero pairs (equal to R's `x_i / y_j`):
/// the points where AB of `x / sig` against `y` can change. Sorted magnitudes per sign.
struct RatioSet {
    pos: (Vec<f64>, Vec<f64>),
    neg: (Vec<f64>, Vec<f64>),
}

impl RatioSet {
    fn new(xs: &[f64], ys: &[f64]) -> Self {
        let part = |v: &[f64], positive: bool| -> Vec<f64> {
            let mut out: Vec<f64> = v
                .iter()
                .filter(|&&a| if positive { a > 0.0 } else { a < 0.0 })
                .map(|a| a.abs())
                .collect();
            out.sort_by(f64::total_cmp);
            out
        };
        RatioSet {
            pos: (part(xs, true), part(ys, true)),
            neg: (part(xs, false), part(ys, false)),
        }
    }

    fn len(&self) -> usize {
        self.pos.0.len() * self.pos.1.len() + self.neg.0.len() * self.neg.1.len()
    }

    /// Number of ratios `a / b <= t`.
    fn count_le(&self, t: f64) -> usize {
        let group = |a: &[f64], b: &[f64]| -> usize {
            // for b ascending, a / b decreases, so the count of a with a / b <= t grows
            let mut i = 0usize;
            let mut count = 0usize;
            for &bj in b {
                while i < a.len() && a[i] / bj <= t {
                    i += 1;
                }
                count += i;
            }
            count
        };
        group(&self.pos.0, &self.pos.1) + group(&self.neg.0, &self.neg.1)
    }

    /// The smallest ratio above `t` (infinity when there is none).
    fn next_above(&self, t: f64) -> f64 {
        let group = |a: &[f64], b: &[f64]| -> f64 {
            // for b ascending the first a with a / b > t moves right
            let mut i = 0usize;
            let mut best = f64::INFINITY;
            for &bj in b {
                while i < a.len() && a[i] / bj <= t {
                    i += 1;
                }
                if i < a.len() {
                    best = best.min(a[i] / bj);
                }
            }
            best
        };
        group(&self.pos.0, &self.pos.1).min(group(&self.neg.0, &self.neg.1))
    }

    /// The k-th and (k+1)-th smallest ratios (1-based), with one bisection.
    fn kth_and_next(&self, k: usize) -> (f64, f64) {
        let a = self.kth(k);
        let b = if self.count_le(a) > k {
            a
        } else {
            self.next_above(a)
        };
        (a, b)
    }

    fn bounds(&self) -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = 0.0f64;
        for (a, b) in [&self.pos, &self.neg] {
            if !a.is_empty() && !b.is_empty() {
                lo = lo.min(a[0] / b[b.len() - 1]);
                hi = hi.max(a[a.len() - 1] / b[0]);
            }
        }
        (lo, hi)
    }

    /// The k-th smallest ratio (1-based).
    fn kth(&self, k: usize) -> f64 {
        let (lo, hi) = self.bounds();
        let (mut a, mut b) = (lo.to_bits(), hi.to_bits());
        while a < b {
            let mid = a + (b - a) / 2;
            if self.count_le(f64::from_bits(mid)) >= k {
                b = mid;
            } else {
                a = mid + 1;
            }
        }
        f64::from_bits(a)
    }
}

/// The asymptotic Bauer interval and estimate: R's `uniroot` on `ab2(sig, zq)` over
/// `srange`, a step function that changes only at the positive ratios, so each root is the
/// ratio where `ab2` changes sign (found by bisection over the rank of the ratio, each ratio
/// selected without forming the m n ratios). A root inside an interval where `ab2` is zero
/// is that interval's midpoint.
fn bauer_asymptotic_interval(
    xs: &[f64],
    ys: &[f64],
    level: f64,
    alternative: Alternative,
    warnings: &mut Vec<&'static str>,
) -> EconResult<Bounds> {
    let (m, n) = (xs.len(), ys.len());
    let ab2 = |sig: f64, zq: f64| -> EconResult<f64> {
        let (s, ties) = ab_statistic(xs, ys, sig)?;
        Ok(ab_normalize(m, n, s, xs, ys, sig, ties)? - zq)
    };
    // R's srange from the positive and the non-positive parts
    let pos = |v: &[f64]| v.iter().copied().filter(|&a| a > 0.0).collect::<Vec<_>>();
    let (xp, yp) = (pos(xs), pos(ys));
    let xn: Vec<f64> = xs.iter().copied().filter(|&a| a <= 0.0).collect();
    let yn: Vec<f64> = ys.iter().copied().filter(|&a| a < 0.0).collect();
    let minf = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let maxf = |v: &[f64]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut ends: Vec<f64> = Vec::new();
    if !xp.is_empty() && !yp.is_empty() {
        ends.push(minf(&xp) / maxf(&yp));
        ends.push(maxf(&xp) / minf(&yp));
    }
    if !xn.is_empty() && !yn.is_empty() {
        ends.push(minf(&xn) / maxf(&yn));
        ends.push(maxf(&xn) / minf(&yn));
    }
    if ends.iter().any(|v| v.is_infinite()) || ends.is_empty() {
        warnings.push("cannot compute asymptotic confidence set or estimator");
        return Ok((None, None, None));
    }
    let srange = (minf(&ends), maxf(&ends));
    let ratios = RatioSet::new(xs, ys);
    let total = ratios.len();
    // ab2 on the interval above the k-th ratio (k = 0: srange's lower end; k = total: its
    // upper end)
    let value_above = |k: usize, zq: f64| -> EconResult<f64> {
        checkpoint()?;
        if k == 0 {
            return ab2(srange.0, zq);
        }
        if k >= total {
            return ab2(srange.1, zq);
        }
        let (a, b) = ratios.kth_and_next(k);
        ab2(a + (b - a) / 2.0, zq)
    };
    // smallest k in 0..=total with pred(value_above(k)), assuming monotone
    let first = |zq: f64, pred: &dyn Fn(f64) -> bool| -> EconResult<Option<usize>> {
        if !pred(value_above(total, zq)?) {
            return Ok(None);
        }
        let (mut lo, mut hi) = (0usize, total);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(value_above(mid, zq)?) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(Some(lo))
    };
    let at = |k: usize| -> f64 {
        if k == 0 {
            srange.0
        } else if k >= total {
            srange.1
        } else {
            ratios.kth(k)
        }
    };
    let root = |zq: f64| -> EconResult<f64> {
        let k_nonneg = first(zq, &|v| v >= 0.0)?;
        let k_pos = first(zq, &|v| v > 0.0)?;
        Ok(match (k_nonneg, k_pos) {
            (Some(a), Some(b)) if a < b => {
                let (lo, hi) = (at(a), at(b));
                lo + (hi - lo) / 2.0
            }
            (Some(a), _) => at(a),
            (None, _) => srange.1,
        })
    };
    let ccia =
        |alpha: f64, warnings: &mut Vec<&'static str>| -> EconResult<(Option<f64>, Option<f64>)> {
            let zl = normal_quantile(alpha / 2.0)?;
            let zu = -zl;
            let statu = ab2(srange.0, zl)?;
            let statl = ab2(srange.1, zu)?;
            if statu > 0.0 || statl < 0.0 {
                warnings.push(DIFFER_IN_LOCATION);
                return Ok((None, None));
            }
            let (u, l) = (root(zl)?, root(zu)?);
            Ok((Some(u.min(l)), Some(u.max(l))))
        };
    let alpha = 1.0 - level;
    let (lower, upper) = match alternative {
        Alternative::TwoSided => ccia(alpha, warnings)?,
        Alternative::Greater => (ccia(alpha * 2.0, warnings)?.0, Some(f64::INFINITY)),
        Alternative::Less => (Some(0.0), ccia(alpha * 2.0, warnings)?.1),
    };
    let statu = ab2(srange.0, 0.0)?;
    let statl = ab2(srange.1, 0.0)?;
    let estimate = if statu > 0.0 || statl < 0.0 {
        warnings.push("cannot compute estimate, returning NA");
        None
    } else {
        Some(root(0.0)?)
    };
    Ok((lower, upper, estimate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ratio_selection_matches_naive_sort() {
        // duplicated ratios (2 / 1 = 4 / 2) and both signs
        let xs = [-3.0, -1.5, 0.0, 1.0, 2.0, 2.0, 4.0, 7.5];
        let ys = [-2.0, -0.5, 1.0, 2.0, 3.0];
        let set = RatioSet::new(&xs, &ys);
        let mut naive: Vec<f64> = Vec::new();
        for &a in &xs {
            for &b in &ys {
                if (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0) {
                    naive.push(a / b);
                }
            }
        }
        naive.sort_by(f64::total_cmp);
        assert_eq!(set.len(), naive.len());
        for k in 1..naive.len() {
            let (a, b) = set.kth_and_next(k);
            assert_eq!(a, naive[k - 1], "k = {k}");
            assert_eq!(b, naive[k], "k = {k}");
        }
    }

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
        let (mut pe, mut pl, mut pg, mut pt) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for set in all {
            let x = column_where(&ds, "value", &[("set", set), ("group", "x")]);
            let y = column_where(&ds, "value", &[("set", set), ("group", "y")]);
            if tie_free.contains(&set) {
                pe.push(
                    ansari_test(&x, &y, Alternative::TwoSided, true, None)
                        .unwrap()
                        .p_value,
                );
                pl.push(
                    ansari_test(&x, &y, Alternative::Less, false, None)
                        .unwrap()
                        .p_value,
                );
                pg.push(
                    ansari_test(&x, &y, Alternative::Greater, false, None)
                        .unwrap()
                        .p_value,
                );
            } else {
                pt.push(
                    ansari_test(&x, &y, Alternative::TwoSided, false, None)
                        .unwrap()
                        .p_value,
                );
            }
        }
        Check::new(&case)
            .named("ab", &all, &ab, Tol::exact())
            .named("ab_p_normal", &tie_free, &p, Tol::p_value(6))
            .named("ab_p_exact", &tie_free, &pe, Tol::p_value(6))
            .named("ab_p_less", &tie_free, &pl, Tol::p_value(6))
            .named("ab_p_greater", &tie_free, &pg, Tol::p_value(6))
            .named("ab_p_ties", &["ties", "serum"], &pt, Tol::p_value(6))
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
        let e = ansari_test(&x, &y, Alternative::TwoSided, true, None).unwrap();
        assert!(e.exact);
        Check::new(&case)
            .scalar("ansari_ab", r.statistic, Tol::exact())
            .scalar("ansari_p_normal", r.p_value, Tol::p_value(14))
            .scalar("ansari_p_exact", e.p_value, Tol::p_value(14))
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

    fn t4(ds: &crate::data::Dataset, set: &str, group: &str) -> Vec<f64> {
        crate::stats::ranks::test_data::column_where(ds, "value", &[("set", set), ("group", group)])
    }

    /// R's exact distribution on its full support, its tails (less = upper tail of AB) and
    /// the tie-corrected normal approximation. Fails before: the support was truncated, the
    /// tails were swapped and ties were ignored in the variance.
    #[test]
    fn test_validate_ansari_exact_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_scale_t4");
        let run = |set: &str, alt: Alternative, exact: Option<bool>| {
            ansari_test_with(&t4(&ds, set, "x"), &t4(&ds, set, "y"), alt, exact, None).unwrap()
        };
        let (tt, lt, gt) = (
            Alternative::TwoSided,
            Alternative::Less,
            Alternative::Greater,
        );
        let scale = run("scale", tt, Some(true));
        let odd = run("odd", tt, Some(true));
        let ab45 = run("ab45", tt, Some(true));
        let cen = run("cen", tt, Some(true));
        for r in [&scale, &odd, &ab45, &cen] {
            assert!(r.exact && r.warning.is_none());
        }
        let ties_l = run("ties", lt, Some(false));
        let ties_g = run("ties", gt, Some(false));
        assert!(!ties_l.exact);
        let ties_default = run("ties", lt, None);
        assert_eq!(
            ties_default.warning.as_deref(),
            Some("cannot compute exact p-value with ties")
        );
        Check::new(&case)
            .scalar("scale_p_exact_two", scale.p_value, Tol::p_value(20))
            .scalar(
                "scale_p_exact_less",
                run("scale", lt, Some(true)).p_value,
                Tol::p_value(20),
            )
            .scalar(
                "scale_p_exact_greater",
                run("scale", gt, Some(true)).p_value,
                Tol::p_value(20),
            )
            .scalar("odd_p_exact_two", odd.p_value, Tol::p_value(14))
            .scalar(
                "odd_p_exact_less",
                run("odd", lt, Some(true)).p_value,
                Tol::p_value(14),
            )
            .scalar("ab45_AB", ab45.statistic, Tol::exact())
            .scalar("ab45_p_exact", ab45.p_value, Tol::p_value(90))
            .scalar("cen_AB", cen.statistic, Tol::exact())
            .scalar("cen_p_exact", cen.p_value, Tol::p_value(22))
            .scalar("ties_p_less", ties_l.p_value, Tol::p_value(10))
            .scalar("ties_p_greater", ties_g.p_value, Tol::p_value(10))
            .finish();
    }

    /// The exact probabilities are a distribution at R's largest default sizes.
    #[test]
    fn test_ansari_exact_distribution_49_by_49() {
        let d = AnsariDist::new(49, 49).unwrap();
        assert_eq!(d.l, 625);
        assert_eq!(d.max(), 625 + 49 * 49 / 2);
        assert!((d.probs.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(d.probs.iter().all(|p| p.is_finite() && *p >= 0.0));
        let q = 1200.0;
        assert!((d.lower(q) + d.upper(q) - 1.0).abs() < 1e-12);
    }

    /// Bauer's exact and asymptotic intervals and estimates. Fails before: no interval, and
    /// the estimate was sqrt(AB / E[AB]).
    #[test]
    fn test_validate_ansari_intervals_t4() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_rank_scale_t4");
        let run = |set: &str, alt: Alternative, exact: Option<bool>| {
            ansari_test_with(
                &t4(&ds, set, "x"),
                &t4(&ds, set, "y"),
                alt,
                exact,
                Some(0.95),
            )
            .unwrap()
        };
        let bounds =
            |r: &AnsariBradleyResult| [r.conf_int_lower.unwrap(), r.conf_int_upper.unwrap()];
        let scale = run("scale", Alternative::TwoSided, Some(true));
        let cen = run("cen", Alternative::TwoSided, Some(true));
        let cen_g = run("cen", Alternative::Greater, Some(true));
        let big = run("big", Alternative::TwoSided, Some(false));
        assert!(scale.exact && cen.exact && !big.exact);
        assert_eq!(cen_g.conf_int_upper, Some(f64::INFINITY));
        Check::new(&case)
            .vector("scale_ci_exact", &bounds(&scale), Tol::stat(20))
            .scalar("scale_est_exact", scale.estimate.unwrap(), Tol::stat(20))
            .vector("cen_ci_exact", &bounds(&cen), Tol::stat(22))
            .scalar("cen_est_exact", cen.estimate.unwrap(), Tol::stat(22))
            .scalar(
                "cen_ci_greater_lower",
                cen_g.conf_int_lower.unwrap(),
                Tol::stat(22),
            )
            .scalar("big_AB", big.statistic, Tol::exact())
            .scalar("big_p_normal", big.p_value, Tol::p_value(65))
            .vector("big_ci", &bounds(&big), Tol::stat(65))
            .scalar("big_est", big.estimate.unwrap(), Tol::stat(65))
            .finish();
        // R's uniroot root (tol 1e-4) lies within 1e-4 of each jump point
        let r_ci = case.vector("big_ci_r");
        assert!((bounds(&big)[0] - r_ci[0]).abs() < 1e-4);
        assert!((bounds(&big)[1] - r_ci[1]).abs() < 1e-4);
        assert!((big.estimate.unwrap() - case.scalar("big_est_r")).abs() < 1e-4);
    }

    /// Samples that differ in location: R's warning and no interval.
    #[test]
    fn test_ansari_interval_differ_in_location() {
        let x = [10.1, 10.4, 10.9, 11.3, 11.8];
        let y = [-9.0, -8.0, -7.0, -6.0, 0.5];
        let r = ansari_test_with(&x, &y, Alternative::TwoSided, Some(false), Some(0.95)).unwrap();
        assert_eq!((r.conf_int_lower, r.conf_int_upper), (None, None));
        assert!(
            r.warning.as_deref().unwrap().contains(
                "samples differ in location: cannot compute confidence set, returning NA"
            ),
            "{:?}",
            r.warning
        );
    }
}
