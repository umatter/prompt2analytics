//! Fisher's Exact Test for count data.
//!
//! Provides Fisher's exact test for 2×2 contingency tables, which tests for
//! independence between two categorical variables using exact probability
//! calculations based on the hypergeometric distribution. The p-values, the
//! conditional maximum-likelihood estimate of the odds ratio and its exact
//! confidence interval are those of R's `stats::fisher.test` (2 x 2 case).
//!
//! # References
//!
//! - Fisher, R. A. (1935). "The logic of inductive inference".
//!   *Journal of the Royal Statistical Society*, 98(1), 39-82.
//! - Fisher, R. A. (1922). "On the interpretation of χ² from contingency tables,
//!   and the calculation of P". *Journal of the Royal Statistical Society*,
//!   85(1), 87-94.
//! - R Core Team. `stats::fisher.test()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html>
//! - SciPy. `scipy.stats.fisher_exact()` function.
//!   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html>
//!
//! # Mathematical Background
//!
//! ## Hypergeometric Distribution
//!
//! For a 2×2 table:
//! ```text
//!              | Col 1 | Col 2 | Total
//!    ----------+-------+-------+-------
//!    Row 1     |   a   |   b   | a + b
//!    Row 2     |   c   |   d   | c + d
//!    ----------+-------+-------+-------
//!    Total     | a + c | b + d |   n
//! ```
//!
//! Given fixed marginals (m = a + c, n = b + d, k = a + b), the top-left cell X
//! ranges over `lo = max(0, k - n) ..= hi = min(k, m)` and, for a true odds ratio
//! psi, follows the noncentral hypergeometric distribution
//!
//! ```text
//! P(X = s) ∝ C(m, s) × C(n, k - s) × psi^s
//! ```
//!
//! which is the central hypergeometric distribution at psi = 1. The weights are
//! computed in log space, so the test works for any table size.
//!
//! ## P-Value Calculation
//!
//! - **Two-sided**: the probability of the tables whose probability is at most
//!   `1 + 1e-7` times that of the observed table (R's `relErr`)
//! - **Greater**: P(X ≥ a) - tests for positive association
//! - **Less**: P(X ≤ a) - tests for negative association
//!
//! ## Odds Ratio
//!
//! `conditional_mle` is the conditional maximum-likelihood estimate (R's
//! `estimate`): the psi at which the mean of X equals a. `odds_ratio` is the
//! sample odds ratio `(a × d) / (b × c)`. The confidence interval inverts the
//! one-sided tests, as R does: two-sided `(L(alpha/2), U(alpha/2))`, `less`
//! `(0, U(alpha))`, `greater` `(L(alpha), Inf)`. The estimating equations are
//! solved by Brent's method (R's `zeroin`) with the same brackets and the same
//! `1/t` substitution as R, to full precision (R's `uniroot` stops at an absolute
//! tolerance of about 1e-4 on the `1/t` scale).

use serde::{Deserialize, Serialize};
use statrs::function::factorial::ln_binomial;

use crate::data::Dataset;
use crate::data::columns::MissingSummary;
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::contingency::contingency_table_2x2;
use crate::traits::SignificanceLevel;

/// Alternative hypothesis for Fisher's exact test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FisherAlternative {
    /// Two-sided test: odds ratio ≠ 1
    #[default]
    TwoSided,
    /// One-sided test: odds ratio > 1 (positive association)
    Greater,
    /// One-sided test: odds ratio < 1 (negative association)
    Less,
}

impl std::fmt::Display for FisherAlternative {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FisherAlternative::TwoSided => write!(f, "two.sided"),
            FisherAlternative::Greater => write!(f, "greater"),
            FisherAlternative::Less => write!(f, "less"),
        }
    }
}

fn nan_f64() -> f64 {
    f64::NAN
}

/// Result of Fisher's exact test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherExactResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Test Identification
    // ═══════════════════════════════════════════════════════════════════════
    /// Description of the test performed
    pub test_name: String,

    // ═══════════════════════════════════════════════════════════════════════
    // Primary Results
    // ═══════════════════════════════════════════════════════════════════════
    /// P-value from the exact test
    pub p_value: f64,
    /// Significance level
    pub significance: SignificanceLevel,
    /// Alternative hypothesis tested
    pub alternative: FisherAlternative,

    // ═══════════════════════════════════════════════════════════════════════
    // Odds Ratio
    // ═══════════════════════════════════════════════════════════════════════
    /// Sample odds ratio: (a × d) / (b × c)
    pub odds_ratio: f64,
    /// Exact conditional confidence interval for the odds ratio, as R's `fisher.test`
    /// computes it for `alternative` (if a confidence level was given)
    pub odds_ratio_ci: Option<(f64, f64)>,
    /// Confidence level used for CI
    pub conf_level: Option<f64>,
    /// Conditional maximum-likelihood estimate of the odds ratio (R fisher.test's
    /// `estimate`); 0 or +inf at the support edges.
    #[serde(default = "nan_f64")]
    pub conditional_mle: f64,

    // ═══════════════════════════════════════════════════════════════════════
    // Table Data
    // ═══════════════════════════════════════════════════════════════════════
    /// The 2×2 contingency table [a, b, c, d] in row-major order
    pub table: [f64; 4],
    /// Row marginals [a+b, c+d]
    pub row_totals: [f64; 2],
    /// Column marginals [a+c, b+d]
    pub col_totals: [f64; 2],
    /// Grand total
    pub n: f64,
    /// Row level labels in table order (empty for array input).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub row_levels: Vec<String>,
    /// Column level labels in table order (empty for array input).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub col_levels: Vec<String>,
    /// Rows dropped because of a missing value (dataset input).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,

    // ═══════════════════════════════════════════════════════════════════════
    // Probability Under Null
    // ═══════════════════════════════════════════════════════════════════════
    /// Probability of the observed table under null hypothesis
    pub prob_observed: f64,
}

/// A number as the Display shows it: `Inf`, `-Inf` and `NaN` as R prints them.
fn show(v: f64, decimals: usize) -> String {
    if v == f64::INFINITY {
        "Inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-Inf".to_string()
    } else if v.is_nan() {
        "NaN".to_string()
    } else {
        format!("{v:.decimals$}")
    }
}

/// A p-value as the Display shows it: six decimals, or four significant digits below 1e-4.
fn show_p(p: f64) -> String {
    if p.is_finite() && p > 0.0 && p < 1e-4 {
        format!("{p:.4e}")
    } else {
        show(p, 6)
    }
}

impl std::fmt::Display for FisherExactResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.test_name)?;
        writeln!(f, "{}", "=".repeat(self.test_name.len()))?;
        writeln!(f)?;

        // Table display, with the level labels when the table came from data
        let labels = |levels: &[String], default: [&str; 2]| -> [String; 2] {
            match levels {
                [first, second] => [first.clone(), second.clone()],
                _ => default.map(str::to_string),
            }
        };
        let rows = labels(&self.row_levels, ["Row1", "Row2"]);
        let cols = labels(&self.col_levels, ["Col1", "Col2"]);
        let rw = rows
            .iter()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0)
            .max(5);
        let cw = cols
            .iter()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0)
            .max(6);
        writeln!(f, "Observed:")?;
        writeln!(
            f,
            "  {:<rw$}  {:>cw$}  {:>cw$}  {:>cw$}",
            "", cols[0], cols[1], "Total"
        )?;
        for (i, row) in rows.iter().enumerate() {
            writeln!(
                f,
                "  {:<rw$}  {:>cw$.0}  {:>cw$.0}  {:>cw$.0}",
                row,
                self.table[2 * i],
                self.table[2 * i + 1],
                self.row_totals[i]
            )?;
        }
        writeln!(
            f,
            "  {:<rw$}  {:>cw$.0}  {:>cw$.0}  {:>cw$.0}",
            "Total", self.col_totals[0], self.col_totals[1], self.n
        )?;
        writeln!(f)?;

        // P-value
        writeln!(
            f,
            "p-value = {} {}",
            show_p(self.p_value),
            self.significance.stars()
        )?;
        let direction = match self.alternative {
            FisherAlternative::TwoSided => "not equal to 1",
            FisherAlternative::Greater => "greater than 1",
            FisherAlternative::Less => "less than 1",
        };
        writeln!(f, "alternative hypothesis: true odds ratio is {direction}")?;
        writeln!(f)?;

        // Odds ratios
        writeln!(
            f,
            "conditional MLE odds ratio: {}",
            show(self.conditional_mle, 6)
        )?;
        if let (Some((lo, hi)), Some(level)) = (self.odds_ratio_ci, self.conf_level) {
            // the level in percent without binary noise (0.9 is 90, not 90.00000000000001)
            let percent = (level * 1e8).round() / 1e6;
            writeln!(
                f,
                "{percent}% confidence interval (exact conditional): ({}, {})",
                show(lo, 4),
                show(hi, 4)
            )?;
        }
        let sample = if self.odds_ratio.is_nan() {
            "undefined (a d = b c = 0)".to_string()
        } else {
            show(self.odds_ratio, 6)
        };
        writeln!(f, "sample odds ratio: {sample}")?;

        if let Some(note) = self.missing.note() {
            writeln!(f)?;
            writeln!(f, "Missing values: {note}")?;
        }
        Ok(())
    }
}

/// Fisher's exact test for a 2×2 contingency table.
///
/// Tests the null hypothesis that the odds ratio equals 1 (independence)
/// using exact probability calculations based on the hypergeometric distribution,
/// as R's `fisher.test` does for a 2 x 2 table: the same p-values, conditional MLE
/// (`conditional_mle`) and exact conditional confidence interval for `alternative`.
///
/// # Arguments
///
/// * `table` - A 2×2 contingency table as `[[a, b], [c, d]]` of whole-number counts
/// * `alternative` - Alternative hypothesis: TwoSided, Greater, or Less
/// * `conf_level` - Confidence level for odds ratio CI (e.g., 0.95)
///
/// # Returns
///
/// `FisherExactResult` containing p-value, odds ratios, and optionally the CI.
///
/// # Errors
///
/// - [`EconError::InvalidSpecification`] for a negative cell, a cell that is not a finite
///   whole number up to 2^53, or a confidence level outside (0, 1).
/// - [`EconError::InsufficientData`] for an empty table.
/// - [`EconError::ResourceLimit`] when the support of the table cannot be allocated.
/// - [`EconError::Cancelled`] when the thread's budget is cancelled.
///
/// # Mathematical Details
///
/// The probability of observing exactly `a` in the top-left cell (given fixed marginals)
/// is:
/// ```text
/// P(X = a) = C(a+c, a) × C(b+d, b) / C(n, a+b)
/// ```
///
/// # Example
///
/// ```
/// use p2a_core::stats::fisher::{fisher_exact_test, FisherAlternative};
///
/// // Classic tea-tasting lady example
/// let table = [[3.0, 1.0], [1.0, 3.0]];
/// let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();
/// // R: p-value = 0.4857, odds ratio 6.408309, CI 0.2117329 621.9337 (tighter: 626.2435)
/// assert!((result.p_value - 0.4857142857142858).abs() < 1e-12);
/// assert!((result.conditional_mle - 6.408319658199671).abs() < 1e-8);
/// println!("{}", result);
/// ```
pub fn fisher_exact_test(
    table: &[[f64; 2]; 2],
    alternative: FisherAlternative,
    conf_level: Option<f64>,
) -> EconResult<FisherExactResult> {
    // Validate inputs: finite whole numbers, so that the support is well defined (a
    // fractional count used to be truncated, and Inf saturated to u64::MAX)
    for (i, row) in table.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if v < 0.0 {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Fisher's exact test needs non-negative counts; cell [{i}][{j}] is {v}"
                    ),
                });
            }
            if !v.is_finite() || v.fract() != 0.0 || v > MAX_COUNT {
                return Err(EconError::InvalidSpecification {
                    message: format!(
                        "Fisher's exact test needs whole-number counts; cell [{i}][{j}] is {v}"
                    ),
                });
            }
        }
    }
    if let Some(level) = conf_level
        && !(level > 0.0 && level < 1.0)
    {
        return Err(EconError::InvalidSpecification {
            message: format!("Confidence level must be between 0 and 1, got {level}"),
        });
    }

    let a = table[0][0];
    let b = table[0][1];
    let c = table[1][0];
    let d = table[1][1];

    // Calculate marginals
    let row1 = a + b;
    let row2 = c + d;
    let col1 = a + c;
    let col2 = b + d;
    let n = a + b + c + d;

    if n == 0.0 {
        return Err(EconError::InsufficientData {
            required: 1,
            provided: 0,
            context: "Table total must be positive".to_string(),
        });
    }

    // Calculate sample odds ratio
    let odds_ratio = if b * c > 0.0 {
        (a * d) / (b * c)
    } else if a * d > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };

    // Whole numbers up to 2^53 convert exactly
    let support = FisherSupport::new(a as u64, b as u64, c as u64, d as u64)?;

    // The one-sided p-values are R's pnhyper(x, 1): the tail of the normalised weights. A
    // ratio of two sums with common rounding keeps P(X <= x) = 1 - 6.5e-27 at 1, where
    // statrs' Hypergeometric::cdf, which sums the central probabilities one by one from 0,
    // gives 0.99999999999995 (and loops over every value below the support).
    let p_value = match alternative {
        FisherAlternative::Greater => support.tail(1.0, true)?,
        FisherAlternative::Less => support.tail(1.0, false)?,
        FisherAlternative::TwoSided => support.two_sided_p()?,
    };

    let conditional_mle = support.conditional_mle()?;

    let odds_ratio_ci = match conf_level {
        None => None,
        Some(level) => {
            let alpha = 1.0 - level;
            Some(match alternative {
                FisherAlternative::TwoSided => (
                    support.ncp_lower(alpha / 2.0)?,
                    support.ncp_upper(alpha / 2.0)?,
                ),
                FisherAlternative::Less => (0.0, support.ncp_upper(alpha)?),
                FisherAlternative::Greater => (support.ncp_lower(alpha)?, f64::INFINITY),
            })
        }
    };

    let test_name = format!(
        "Fisher's Exact Test for Count Data (alternative: {})",
        alternative
    );

    Ok(FisherExactResult {
        test_name,
        p_value,
        significance: SignificanceLevel::from_p_value(p_value),
        alternative,
        odds_ratio,
        odds_ratio_ci,
        conf_level,
        conditional_mle,
        table: [a, b, c, d],
        row_totals: [row1, row2],
        col_totals: [col1, col2],
        n,
        row_levels: Vec::new(),
        col_levels: Vec::new(),
        missing: MissingSummary::none(n as usize),
        prob_observed: support.prob_observed(),
    })
}

/// R's `relErr` in `fisher.test`: a table counts as at most as likely as the observed one
/// when its probability is at most `1 + 1e-7` times the observed probability.
const REL_ERR: f64 = 1.0 + 1e-7;

/// The largest count accepted: whole numbers up to 2^53 are exact in an f64.
const MAX_COUNT: f64 = 9_007_199_254_740_992.0;

/// Iterations of Brent's root finder (R's `uniroot` default is 1000).
const ZEROIN_MAXIT: usize = 2000;

/// Passes over the support call [`resource::checkpoint`] this often.
const CHECK_EVERY: usize = 65_536;

/// The support of the top-left cell of a 2 x 2 table given its margins (R `fisher.test`'s
/// `lo:hi`), with the log of each point's unnormalised hypergeometric probability.
struct FisherSupport {
    /// The observed top-left count.
    x: u64,
    /// Smallest and largest top-left count the margins allow.
    lo: u64,
    hi: u64,
    /// `ln C(m, s) + ln C(n, k - s)` for `s` in `lo..=hi` (R's `dhyper(.., log = TRUE)` up to
    /// the constant `-ln C(m + n, k)`, which normalises away).
    logdc: Vec<f64>,
    /// `ln C(m + n, k)`, the normalising constant of the central distribution.
    ln_denom: f64,
}

impl FisherSupport {
    fn new(a: u64, b: u64, c: u64, d: u64) -> EconResult<Self> {
        let (m, n, k) = (a + c, b + d, a + b);
        let lo = k.saturating_sub(n);
        let hi = k.min(m);
        let len = usize::try_from(hi - lo)
            .ok()
            .and_then(|l| l.checked_add(1))
            .ok_or_else(|| {
                EconError::ResourceLimit(
                    "Fisher's exact test: the support of the table is too large".to_string(),
                )
            })?;
        let mut logdc: Vec<f64> = resource::try_with_capacity(len, "Fisher's exact test support")?;
        for (i, s) in (lo..=hi).enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            logdc.push(ln_binomial(m, s) + ln_binomial(n, k - s));
        }
        Ok(FisherSupport {
            x: a,
            lo,
            hi,
            logdc,
            ln_denom: ln_binomial(m + n, k),
        })
    }

    /// Index of the observed count in the support.
    fn obs(&self) -> usize {
        // x is in lo..=hi, whose length fits a usize (checked in `new`)
        (self.x - self.lo) as usize
    }

    /// The support point of index `i`.
    fn point(&self, i: usize) -> f64 {
        (self.lo + i as u64) as f64
    }

    /// `max_i (logdc[i] + ln_psi * s_i)`, the shift that keeps every weight at most 1.
    fn max_log(&self, ln_psi: f64) -> EconResult<f64> {
        let mut max = f64::NEG_INFINITY;
        for (i, &l) in self.logdc.iter().enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            max = max.max(l + ln_psi * self.point(i));
        }
        Ok(max)
    }

    /// R's `mnhyper(psi)`: the mean of the noncentral hypergeometric distribution.
    fn mean(&self, psi: f64) -> EconResult<f64> {
        if psi == 0.0 {
            return Ok(self.lo as f64);
        }
        if psi == f64::INFINITY {
            return Ok(self.hi as f64);
        }
        let ln_psi = psi.ln();
        let max = self.max_log(ln_psi)?;
        let (mut num, mut den) = (0.0, 0.0);
        for (i, &l) in self.logdc.iter().enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            let s = self.point(i);
            let w = (l + ln_psi * s - max).exp();
            num += s * w;
            den += w;
        }
        Ok(num / den)
    }

    /// R's `pnhyper(x, psi, upper.tail)`: `P(X >= x)` when `upper`, else `P(X <= x)`, under
    /// odds ratio `psi`.
    fn tail(&self, psi: f64, upper: bool) -> EconResult<f64> {
        let x = self.x;
        let indicator = |b: bool| if b { 1.0 } else { 0.0 };
        if psi == 0.0 {
            return Ok(indicator(if upper { x <= self.lo } else { x >= self.lo }));
        }
        if psi == f64::INFINITY {
            return Ok(indicator(if upper { x <= self.hi } else { x >= self.hi }));
        }
        let ln_psi = psi.ln();
        let max = self.max_log(ln_psi)?;
        let obs = self.obs();
        let (mut selected, mut total) = (0.0, 0.0);
        for (i, &l) in self.logdc.iter().enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            let w = (l + ln_psi * self.point(i) - max).exp();
            total += w;
            if (upper && i >= obs) || (!upper && i <= obs) {
                selected += w;
            }
        }
        Ok(selected / total)
    }

    /// The two-sided p-value of R's `fisher.test`: the probability of the tables at most
    /// `REL_ERR` times as likely as the observed one, all in log space (STAT-2, STAT-5).
    fn two_sided_p(&self) -> EconResult<f64> {
        let max = self.max_log(0.0)?;
        let threshold = (self.logdc[self.obs()] - max).exp() * REL_ERR;
        let (mut selected, mut total) = (0.0, 0.0);
        for (i, &l) in self.logdc.iter().enumerate() {
            if i % CHECK_EVERY == 0 {
                resource::checkpoint()?;
            }
            let w = (l - max).exp();
            total += w;
            if w <= threshold {
                selected += w;
            }
        }
        Ok((selected / total).min(1.0))
    }

    /// The probability of the observed table under independence,
    /// `exp(ln C(m, x) + ln C(n, k - x) - ln C(m + n, k))` (statrs' `Hypergeometric::ln_pmf`).
    fn prob_observed(&self) -> f64 {
        (self.logdc[self.obs()] - self.ln_denom).exp()
    }

    /// R's `mle(x)`: the conditional maximum-likelihood estimate of the odds ratio.
    fn conditional_mle(&self) -> EconResult<f64> {
        if self.x == self.lo {
            return Ok(0.0);
        }
        if self.x == self.hi {
            return Ok(f64::INFINITY);
        }
        let x = self.x as f64;
        let mu = self.mean(1.0)?;
        if mu > x {
            root_in_unit(&mut |t| Ok(self.mean(t)? - x))
        } else if mu < x {
            inverse_root(&mut |t| Ok(self.mean(1.0 / t)? - x))
        } else {
            Ok(1.0)
        }
    }

    /// R's `ncp.U(x, alpha)`: the upper confidence limit of the odds ratio.
    fn ncp_upper(&self, alpha: f64) -> EconResult<f64> {
        if self.x == self.hi {
            return Ok(f64::INFINITY);
        }
        let p = self.tail(1.0, false)?;
        if p < alpha {
            root_in_unit(&mut |t| Ok(self.tail(t, false)? - alpha))
        } else if p > alpha {
            inverse_root(&mut |t| Ok(self.tail(1.0 / t, false)? - alpha))
        } else {
            Ok(1.0)
        }
    }

    /// R's `ncp.L(x, alpha)`: the lower confidence limit of the odds ratio.
    fn ncp_lower(&self, alpha: f64) -> EconResult<f64> {
        if self.x == self.lo {
            return Ok(0.0);
        }
        let p = self.tail(1.0, true)?;
        if p > alpha {
            root_in_unit(&mut |t| Ok(self.tail(t, true)? - alpha))
        } else if p < alpha {
            inverse_root(&mut |t| Ok(self.tail(1.0 / t, true)? - alpha))
        } else {
            Ok(1.0)
        }
    }
}

/// R's `uniroot(f, c(0, 1))$root`, solved to full precision. `f(0)` and `f(1)` have opposite
/// signs wherever this is called; if they do not, the error says so instead of guessing.
fn root_in_unit(f: &mut impl FnMut(f64) -> EconResult<f64>) -> EconResult<f64> {
    bracketed_root(f, 0.0, 1.0)?.ok_or_else(|| {
        EconError::Computation(
            "Fisher's exact test: the estimating equation has no sign change on (0, 1)".to_string(),
        )
    })
}

/// R's `1 / uniroot(function(t) h(1 / t), c(.Machine$double.eps, 1))$root` for
/// `f(t) = h(1 / t)`. When the root lies below `eps` (an odds ratio above 4.5e15, where R's
/// uniroot stops with an error), the bracket is widened to (`f64::MIN_POSITIVE`, 1); when it
/// lies below that too, the odds ratio is +inf.
fn inverse_root(f: &mut impl FnMut(f64) -> EconResult<f64>) -> EconResult<f64> {
    for lower in [f64::EPSILON, f64::MIN_POSITIVE] {
        if let Some(t) = bracketed_root(f, lower, 1.0)? {
            return Ok(1.0 / t);
        }
    }
    Ok(f64::INFINITY)
}

/// The root of `f` on `(ax, bx)`, or `None` when `f(ax)` and `f(bx)` have one sign (or one
/// of them is NaN), as R's `uniroot` checks before calling `zeroin`.
fn bracketed_root(
    f: &mut impl FnMut(f64) -> EconResult<f64>,
    ax: f64,
    bx: f64,
) -> EconResult<Option<f64>> {
    let fa = f(ax)?;
    let fb = f(bx)?;
    if fa == 0.0 {
        return Ok(Some(ax));
    }
    if fb == 0.0 {
        return Ok(Some(bx));
    }
    if fa.is_nan() || fb.is_nan() || (fa > 0.0) == (fb > 0.0) {
        return Ok(None);
    }
    zeroin(f, ax, bx, fa, fb).map(Some)
}

/// Brent's method exactly as R's `R_zeroin2` (src/library/stats/src/zeroin.c) with `tol = 0`,
/// so the step tolerance is `2 eps |b|` (relative), and at most [`ZEROIN_MAXIT`] iterations
/// (on the last one it returns the current approximation, where R's uniroot warns).
fn zeroin(
    f: &mut impl FnMut(f64) -> EconResult<f64>,
    ax: f64,
    bx: f64,
    fa: f64,
    fb: f64,
) -> EconResult<f64> {
    let (mut a, mut b, mut fa, mut fb) = (ax, bx, fa, fb);
    let (mut c, mut fc) = (a, fa);
    for _ in 0..ZEROIN_MAXIT {
        resource::checkpoint()?;
        let prev_step = b - a;
        if fc.abs() < fb.abs() {
            // swap so that b is the best approximation
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs();
        let mut new_step = (c - b) / 2.0;
        if new_step.abs() <= tol_act || fb == 0.0 {
            return Ok(b);
        }
        // interpolation when the previous step was large enough and in the right direction
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q) = if a == c {
                // linear interpolation
                let t1 = fb / fa;
                (cb * t1, 1.0 - t1)
            } else {
                // inverse quadratic interpolation
                let qa = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                (
                    t2 * (cb * qa * (qa - t1) - (b - a) * (t1 - 1.0)),
                    (qa - 1.0) * (t1 - 1.0) * (t2 - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if p < 0.75 * cb * q - (tol_act * q).abs() / 2.0 && p < (prev_step * q / 2.0).abs() {
                new_step = p / q;
            }
        }
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }
        a = b;
        fa = fb;
        b += new_step;
        fb = f(b)?;
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    Ok(b)
}

/// Fisher's exact test from integer table.
///
/// Convenience wrapper for integer inputs.
///
/// # Example
///
/// ```
/// use p2a_core::stats::fisher::{fisher_exact_test_int, FisherAlternative};
///
/// let table = [[3, 1], [1, 3]];
/// let result = fisher_exact_test_int(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();
/// assert!(result.p_value > 0.2);  // Not significant
/// ```
pub fn fisher_exact_test_int(
    table: &[[u64; 2]; 2],
    alternative: FisherAlternative,
    conf_level: Option<f64>,
) -> EconResult<FisherExactResult> {
    let table_f64 = [
        [table[0][0] as f64, table[0][1] as f64],
        [table[1][0] as f64, table[1][1] as f64],
    ];
    fisher_exact_test(&table_f64, alternative, conf_level)
}

/// Run Fisher's exact test on two categorical columns from a dataset.
///
/// Builds the 2×2 table as [`contingency_table`](crate::stats::contingency_table) does: rows
/// with a missing value (null, or NaN in a float column) in either column are dropped and
/// reported in `missing`, as R's `table()` drops them, and the levels follow R `factor()`
/// order (numbers numerically, text in C-locale order), so the first level is the first row
/// or column. The level counts are checked before the table is built. The result carries
/// the level labels.
///
/// # Arguments
///
/// * `dataset` - Input dataset
/// * `row_col` - Column name for rows (exactly 2 levels after dropping missing rows)
/// * `col_col` - Column name for columns (exactly 2 levels after dropping missing rows)
/// * `alternative` - Alternative hypothesis
/// * `conf_level` - Confidence level for odds ratio CI
///
/// # Errors
///
/// [`contingency_table`](crate::stats::contingency_table)'s errors (`ColumnNotFound`,
/// `InvalidSpecification` for a column that cannot be categorical, `InsufficientData`,
/// `ResourceLimit`, `Cancelled`),
/// `InvalidSpecification` naming the levels when the table is not 2 x 2, and
/// [`fisher_exact_test`]'s errors.
///
/// # Example
///
/// ```ignore
/// let result = run_fisher_test(&dataset, "treatment", "outcome",
///     FisherAlternative::TwoSided, Some(0.95))?;
/// ```
pub fn run_fisher_test(
    dataset: &Dataset,
    row_col: &str,
    col_col: &str,
    alternative: FisherAlternative,
    conf_level: Option<f64>,
) -> EconResult<FisherExactResult> {
    let t = contingency_table_2x2(dataset.df(), row_col, col_col)?;
    let table = t
        .as_2x2()
        .ok_or_else(|| EconError::Computation("the 2 x 2 table has another shape".to_string()))?;
    let mut result = fisher_exact_test(&table, alternative, conf_level)?;
    result.row_levels = t.row_levels;
    result.col_levels = t.col_levels;
    result.missing = t.missing;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{ColumnRequest, NaPolicy};
    use p2a_validation::{Case, Check, Tol};

    /// Tight re-solve of fisher.test's equations (uniroot tol 1e-15; validation/stats/fisher.md,
    /// "Tight limits, not R's printed ones, are the reference").
    const TIGHT: Tol = Tol::new(1e-9, 1e-12);
    /// fisher.test's own limits and estimate (uniroot tol eps^0.25): up to 0.69% off on the
    /// case tables.
    const PRINTED: Tol = Tol::new(1e-2, 0.0);

    type Table = [[f64; 2]; 2];

    fn case_tables(case_id: &str) -> (Case, Vec<(String, Table)>) {
        let (case, ds) = crate::test_support::load_case(case_id);
        let cols = ColumnRequest::new(ds.df())
            .factor("table")
            .numerics(&["a", "b", "c", "d"])
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        let t = (0..cols.n())
            .map(|i| {
                (
                    cols.factor("table").label(i).to_string(),
                    [
                        [cols.numeric("a")[i], cols.numeric("b")[i]],
                        [cols.numeric("c")[i], cols.numeric("d")[i]],
                    ],
                )
            })
            .collect();
        (case, t)
    }

    #[test]
    fn test_validate_fisher_tables() {
        // R: fisher.test(matrix(c(a, b, c, d), 2, byrow = TRUE), alternative = ...) on 11
        // tables (case stat_fisher_tables). Fails before: two-sided p = 0 for N >= 1030
        // (STAT-2), 1.2e-10 for tiny_p (STAT-5, R 4.906449339029371e-38), CI (0, 10 x OR)
        // (STAT-24, tea R 0.21173559544657847, 626.2435305888135).
        let (case, tables) = case_tables("stat_fisher_tables");
        assert_eq!(tables.len(), 11);
        let mut check = Check::new(&case);
        for (id, t) in &tables {
            let k = |s: &str| format!("{id}_{s}");
            let n = case.scalar(&k("n")) as usize;
            let two = fisher_exact_test(t, FisherAlternative::TwoSided, Some(0.95)).unwrap();
            let less = fisher_exact_test(t, FisherAlternative::Less, Some(0.95)).unwrap();
            let greater = fisher_exact_test(t, FisherAlternative::Greater, Some(0.95)).unwrap();
            let (l2, u2) = two.odds_ratio_ci.unwrap();
            let (ll, ul) = less.odds_ratio_ci.unwrap();
            let (lg, ug) = greater.odds_ratio_ci.unwrap();
            check
                .scalar(&k("p_two_sided"), two.p_value, Tol::p_value(n))
                .scalar(&k("p_less"), less.p_value, Tol::p_value(n))
                .scalar(&k("p_greater"), greater.p_value, Tol::p_value(n))
                .scalar(&k("estimate_cond_mle_tight"), two.conditional_mle, TIGHT)
                .scalar(&k("estimate_cond_mle"), two.conditional_mle, PRINTED)
                .scalar(&k("ci_lower_two_sided_tight"), l2, TIGHT)
                .scalar(&k("ci_upper_two_sided_tight"), u2, TIGHT)
                .scalar(&k("ci_lower_two_sided"), l2, PRINTED)
                .scalar(&k("ci_upper_two_sided"), u2, PRINTED)
                .scalar(&k("ci_lower_less"), ll, Tol::exact())
                .scalar(&k("ci_upper_less_tight"), ul, TIGHT)
                .scalar(&k("ci_upper_less"), ul, PRINTED)
                .scalar(&k("ci_lower_greater_tight"), lg, TIGHT)
                .scalar(&k("ci_lower_greater"), lg, PRINTED)
                .scalar(
                    &k("sample_odds_ratio"),
                    two.odds_ratio,
                    Tol::new(1e-15, 0.0),
                )
                .scalar(&k("n"), two.n, Tol::exact());
            // R reports Inf as the upper limit of "greater"; the harness stores it as null
            assert!(case.scalar(&k("ci_upper_greater")).is_nan(), "{id}");
            assert_eq!(ug, f64::INFINITY, "{id}: greater upper limit");
            assert_eq!(
                two.conditional_mle.to_bits(),
                less.conditional_mle.to_bits(),
                "{id}: MLE does not depend on the alternative"
            );
            assert_eq!(
                two.conditional_mle.to_bits(),
                greater.conditional_mle.to_bits(),
                "{id}"
            );
        }
        // The p-values below 1e-12 compare equal under Tol::p_value's absolute floor; pin them
        // relatively so the STAT-5 floor (1.2e-10 against R's 4.9e-38) cannot hide.
        for (id, tag, alt) in [
            ("tiny_p", "two_sided", FisherAlternative::TwoSided),
            ("tiny_p", "greater", FisherAlternative::Greater),
            ("n2000_sig", "two_sided", FisherAlternative::TwoSided),
            ("n2000_sig", "greater", FisherAlternative::Greater),
        ] {
            let t = tables.iter().find(|(i, _)| i == id).unwrap().1;
            check.scalar(
                &format!("{id}_p_{tag}"),
                fisher_exact_test(&t, alt, None).unwrap().p_value,
                Tol::new(1e-9, 0.0),
            );
        }
        check.finish();
    }

    #[test]
    fn test_validate_fisher_tables_row_swap() {
        // Swapping the rows maps psi -> 1/psi: p two-sided unchanged, less <-> greater,
        // (L, U) -> (1/U, 1/L), MLE -> 1/MLE. zero_cell [[0,5],[3,2]] swapped is [[3,2],[0,5]],
        // x = hi: MLE and the upper limit are +inf (R: estimate Inf, CI 0.49337123440314978,
        // Inf), the edge no table of the case reaches directly.
        let inv = |v: f64| {
            if v == 0.0 {
                f64::INFINITY
            } else if v.is_infinite() {
                0.0
            } else {
                1.0 / v
            }
        };
        let (case, tables) = case_tables("stat_fisher_tables");
        let mut check = Check::new(&case);
        for (id, t) in &tables {
            let k = |s: &str| format!("{id}_{s}");
            let n = case.scalar(&k("n")) as usize;
            let s = [t[1], t[0]];
            let two = fisher_exact_test(&s, FisherAlternative::TwoSided, Some(0.95)).unwrap();
            let less = fisher_exact_test(&s, FisherAlternative::Less, Some(0.95)).unwrap();
            let greater = fisher_exact_test(&s, FisherAlternative::Greater, Some(0.95)).unwrap();
            let (l2, u2) = two.odds_ratio_ci.unwrap();
            check
                .scalar(&k("p_two_sided"), two.p_value, Tol::p_value(n))
                .scalar(&k("p_less"), greater.p_value, Tol::p_value(n))
                .scalar(&k("p_greater"), less.p_value, Tol::p_value(n))
                .scalar(
                    &k("estimate_cond_mle_tight"),
                    inv(two.conditional_mle),
                    TIGHT,
                )
                .scalar(&k("ci_lower_two_sided_tight"), inv(u2), TIGHT)
                .scalar(&k("ci_upper_two_sided_tight"), inv(l2), TIGHT)
                .scalar(
                    &k("ci_upper_less_tight"),
                    inv(greater.odds_ratio_ci.unwrap().0),
                    TIGHT,
                )
                .scalar(
                    &k("ci_lower_greater_tight"),
                    inv(less.odds_ratio_ci.unwrap().1),
                    TIGHT,
                )
                .scalar(
                    &k("sample_odds_ratio"),
                    inv(two.odds_ratio),
                    Tol::new(1e-14, 0.0),
                );
        }
        check.finish();
        let z = fisher_exact_test(
            &[[3.0, 2.0], [0.0, 5.0]],
            FisherAlternative::TwoSided,
            Some(0.95),
        )
        .unwrap();
        assert_eq!(
            (z.conditional_mle, z.odds_ratio_ci.unwrap().1),
            (f64::INFINITY, f64::INFINITY)
        );
    }

    #[test]
    fn test_fisher_extreme_table_matches_tight_r() {
        // [[50,1],[1,50]]: R 4.3.3 fisher.test prints CI (127.00893441058923, 4503599627370496)
        // and estimate 1530.9010005032094 because uniroot's tol (eps^0.25) is absolute on the
        // 1/t scale; the same equations re-solved with uniroot(tol = 1e-15) give the values
        // below (validation/stats/fisher.md, "Tight limits"). p-values are R's own:
        // two.sided 1.3022734448002492e-26, greater 6.5113672240012631e-27, less 1.
        let t = [[50.0, 1.0], [1.0, 50.0]];
        let rel = |a: f64, e: f64| ((a - e) / e).abs() < 1e-9;
        let two = fisher_exact_test(&t, FisherAlternative::TwoSided, Some(0.95)).unwrap();
        let less = fisher_exact_test(&t, FisherAlternative::Less, Some(0.95)).unwrap();
        let greater = fisher_exact_test(&t, FisherAlternative::Greater, Some(0.95)).unwrap();
        assert!(rel(two.p_value, 1.3022734448002492e-26), "{}", two.p_value);
        assert!(
            rel(greater.p_value, 6.511367224001263e-27),
            "{}",
            greater.p_value
        );
        assert_eq!(less.p_value, 1.0);
        assert!(
            rel(two.conditional_mle, 1512.507433369302),
            "{}",
            two.conditional_mle
        );
        let (l, u) = two.odds_ratio_ci.unwrap();
        assert!(
            rel(l, 126.58051644404978) && rel(u, 102061.81206230588),
            "({l}, {u})"
        );
        assert!(rel(less.odds_ratio_ci.unwrap().1, 50039.549513258804));
        assert!(rel(greater.odds_ratio_ci.unwrap().0, 162.65609650777515));
    }

    #[test]
    fn test_fisher_rejects_invalid_counts() {
        // NaN, Inf, fractional, > 2^53 and negative cells are refused, naming the
        // cell, before any pass over the support.
        for bad in [f64::NAN, f64::INFINITY, 2.5, 1e16, -1.0, f64::NEG_INFINITY] {
            let e = fisher_exact_test(
                &[[bad, 1.0], [1.0, 1.0]],
                FisherAlternative::TwoSided,
                Some(0.95),
            )
            .unwrap_err();
            assert!(
                matches!(e, EconError::InvalidSpecification { .. })
                    && e.to_string().contains("[0][0]"),
                "{bad}: {e}"
            );
        }
        assert!(
            fisher_exact_test(
                &[[1.0, 1.0], [1.0, 1.0]],
                FisherAlternative::TwoSided,
                Some(f64::NAN)
            )
            .is_err()
        );
    }

    #[test]
    fn test_run_fisher_test_refuses_the_shape_before_building_a_table() {
        // Two 40-level columns are refused for their shape, naming the levels, before a 40 x 40
        // table is allocated; before, the table came first, and a small allocation ceiling
        // turned the user's error into a ResourceLimit for a table that could only be refused.
        let a: Vec<i64> = (0..40).collect();
        let ds = Dataset::new(polars::prelude::df!("a" => a.clone(), "b" => a).unwrap());
        let r = resource::with_alloc_ceiling(Some(1024), || {
            run_fisher_test(&ds, "a", "b", FisherAlternative::TwoSided, Some(0.95))
        });
        assert!(
            matches!(&r, Err(EconError::InvalidSpecification { message })
                if message.contains("2 x 2") && message.contains("'a' has 40 level(s) [0, 1,")),
            "{r:?}"
        );
    }

    #[test]
    fn test_fisher_one_sided_p_does_not_walk_below_the_support() {
        // x = lo = 1e12 - 1 with a two-point support: statrs' cdf(x), which the old code
        // called, adds 1e12 zero terms below the support first and never returns. R:
        // phyper(1e12 - 1, 1e12, 2, 1e12 + 1) = 1e12 / (1e12 + 2). The tolerance is loose because statrs' ln_gamma, which the
        // log binomials use, has an absolute error of about 1e-2 at 1e12.
        let t = [[1e12 - 1.0, 2.0], [1.0, 0.0]];
        let less = fisher_exact_test(&t, FisherAlternative::Less, None).unwrap();
        assert!((less.p_value - 1.0).abs() < 0.05, "{}", less.p_value);
        let greater = fisher_exact_test(&t, FisherAlternative::Greater, None).unwrap();
        assert_eq!(greater.p_value, 1.0);
        assert_eq!(less.conditional_mle, 0.0);
    }

    #[test]
    fn test_fisher_honours_cancellation() {
        use crate::resource::{Budget, CancelToken, with_budget};
        let token = CancelToken::new();
        token.cancel();
        let r = with_budget(Budget::new(None, token), || {
            fisher_exact_test(&[[3.0, 1.0], [1.0, 3.0]], FisherAlternative::TwoSided, None)
        });
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
    }

    #[test]
    fn test_basic_fisher_exact() {
        // Classic lady tasting tea example
        let table = [[3.0, 1.0], [1.0, 3.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        assert!(result.p_value > 0.0);
        assert!(result.p_value <= 1.0);
        assert_eq!(result.n, 8.0);
        assert_eq!(result.odds_ratio, 9.0); // (3*3)/(1*1) = 9
    }

    #[test]
    fn test_fisher_integer_wrapper() {
        let table = [[3, 1], [1, 3]];
        let result =
            fisher_exact_test_int(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        assert!(result.p_value > 0.0);
        assert_eq!(result.odds_ratio, 9.0);
    }

    #[test]
    fn test_one_sided_alternatives() {
        let table = [[10.0, 2.0], [3.0, 15.0]];

        let two_sided = fisher_exact_test(&table, FisherAlternative::TwoSided, None).unwrap();
        let greater = fisher_exact_test(&table, FisherAlternative::Greater, None).unwrap();
        let less = fisher_exact_test(&table, FisherAlternative::Less, None).unwrap();

        // Two-sided should be >= one-sided
        assert!(two_sided.p_value >= greater.p_value - 1e-10);
        // greater tests for odds ratio > 1 (positive association)
        assert!(greater.p_value < 0.05); // Should be significant
        // less tests for odds ratio < 1
        assert!(less.p_value > 0.5); // Should not be significant in this direction
    }

    #[test]
    fn test_zero_cells() {
        // Table with zero cell
        let table = [[0.0, 5.0], [5.0, 10.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, None).unwrap();

        assert!(result.p_value > 0.0);
        assert_eq!(result.odds_ratio, 0.0);
    }

    #[test]
    fn test_extreme_table() {
        // Very unbalanced table
        let table = [[50.0, 1.0], [1.0, 50.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        assert!(result.p_value < 0.001); // Should be highly significant
        assert!(result.odds_ratio > 100.0); // Very large odds ratio
    }

    /// The legacy tables of `stat_fisher_legacy` (fisher.test, limits re-solved tight as
    /// T2-WP05): p for the three alternatives, the tight limits and estimate, the sample odds
    /// ratio and n.
    fn check_legacy(case_id: &str, id: &str) {
        let (case, tables) = case_tables(case_id);
        let table = tables
            .iter()
            .find(|(name, _)| name == id)
            .map(|(_, t)| *t)
            .unwrap();
        let run = |alt| fisher_exact_test(&table, alt, Some(0.95)).unwrap();
        let (two, less, greater) = (
            run(FisherAlternative::TwoSided),
            run(FisherAlternative::Less),
            run(FisherAlternative::Greater),
        );
        let p = Tol::new(1e-9, 0.0);
        let (two_lo, two_hi) = two.odds_ratio_ci.unwrap();
        let (less_lo, less_hi) = less.odds_ratio_ci.unwrap();
        let (greater_lo, greater_hi) = greater.odds_ratio_ci.unwrap();
        assert_eq!(less_lo, 0.0, "{id}");
        assert_eq!(greater_hi, f64::INFINITY, "{id}");
        Check::new(&case)
            .scalar(&format!("{id}_p_two_sided"), two.p_value, p)
            .scalar(&format!("{id}_p_less"), less.p_value, p)
            .scalar(&format!("{id}_p_greater"), greater.p_value, p)
            .scalar(&format!("{id}_ci_lower_two_sided_tight"), two_lo, TIGHT)
            .scalar(&format!("{id}_ci_upper_two_sided_tight"), two_hi, TIGHT)
            .scalar(&format!("{id}_ci_upper_less_tight"), less_hi, TIGHT)
            .scalar(&format!("{id}_ci_lower_greater_tight"), greater_lo, TIGHT)
            .scalar(
                &format!("{id}_estimate_cond_mle_tight"),
                two.conditional_mle,
                TIGHT,
            )
            .scalar(
                &format!("{id}_sample_odds_ratio"),
                two.odds_ratio,
                Tol::new(1e-15, 0.0),
            )
            .scalar(&format!("{id}_n"), two.n, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_against_r_basic() {
        // R: fisher.test(matrix(c(1, 9, 11, 3), 2)); p 0.002759456185220086
        check_legacy("stat_fisher_legacy", "basic");
    }

    #[test]
    fn test_validate_against_r_lady_tea() {
        // R: fisher.test(matrix(c(3, 1, 1, 3), 2)); p 0.4857142857142858
        check_legacy("stat_fisher_legacy", "tea");
    }

    #[test]
    fn test_validate_against_r_one_sided() {
        // R: fisher.test(matrix(c(6, 1, 2, 7), 2), alternative = "greater"); p 0.020279720279720286
        check_legacy("stat_fisher_legacy", "one_sided");
    }

    #[test]
    fn test_validate_fisher_legacy_ci() {
        // R: fisher.test(matrix(c(10, 5, 5, 10), 2))$conf.int, re-solved tight:
        // 0.7054372156512816, 23.921858982581192 (before T2-WP05 p2a gave (0, 40))
        check_legacy("stat_fisher_legacy", "ci");
    }

    #[test]
    fn test_error_handling() {
        // Negative values
        let result = fisher_exact_test(
            &[[-1.0, 5.0], [5.0, 5.0]],
            FisherAlternative::TwoSided,
            None,
        );
        assert!(result.is_err());

        // Empty table
        let result =
            fisher_exact_test(&[[0.0, 0.0], [0.0, 0.0]], FisherAlternative::TwoSided, None);
        assert!(result.is_err());

        // Invalid confidence level
        let result = fisher_exact_test(
            &[[1.0, 1.0], [1.0, 1.0]],
            FisherAlternative::TwoSided,
            Some(1.5),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_display() {
        let table = [[3.0, 1.0], [1.0, 3.0]];
        let result = fisher_exact_test(&table, FisherAlternative::TwoSided, Some(0.95)).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Fisher's Exact Test"));
        assert!(display.contains("p-value"));
        assert!(display.contains("odds ratio"));
    }

    #[test]
    fn test_validate_fisher_rows_2000() {
        // R: fisher.test(table(treat, outcome)) on 2000 rows (case stat_fisher_rows_2000): p
        // 0.08111006057913338, MLE 1.1735170755267106, CI 0.9809033190255882,
        // 1.4041857925265708 (tight). Fails before: p = 0 (STAT-2).
        let (case, ds) = crate::test_support::load_case("stat_fisher_rows_2000");
        let mut check = Check::new(&case);
        for (alt, tag) in [
            (FisherAlternative::TwoSided, "two_sided"),
            (FisherAlternative::Less, "less"),
            (FisherAlternative::Greater, "greater"),
        ] {
            let r = run_fisher_test(&ds, "treat", "outcome", alt, Some(0.95)).unwrap();
            assert_eq!(r.table, [520.0, 480.0, 480.0, 520.0]);
            assert_eq!(
                (r.row_levels.clone(), r.col_levels.clone()),
                (
                    vec!["control".to_string(), "treated".into()],
                    vec!["no".to_string(), "yes".into()]
                )
            );
            assert!(r.missing.is_empty());
            check
                .scalar(&format!("p_{tag}"), r.p_value, Tol::p_value(2000))
                .scalar("estimate_cond_mle_tight", r.conditional_mle, TIGHT)
                .scalar("sample_odds_ratio", r.odds_ratio, Tol::new(1e-15, 0.0));
            let (lo, hi) = r.odds_ratio_ci.unwrap();
            match alt {
                FisherAlternative::TwoSided => {
                    check.scalar("ci_lower_two_sided_tight", lo, TIGHT).scalar(
                        "ci_upper_two_sided_tight",
                        hi,
                        TIGHT,
                    );
                }
                FisherAlternative::Less => {
                    check.scalar("ci_lower_less", lo, Tol::exact()).scalar(
                        "ci_upper_less_tight",
                        hi,
                        TIGHT,
                    );
                }
                FisherAlternative::Greater => {
                    check.scalar("ci_lower_greater_tight", lo, TIGHT);
                    assert_eq!(hi, f64::INFINITY);
                }
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_fisher_na_codes_orientation() {
        // R: fisher.test(table(code, out)), rows "2" then "10" (STAT-50), the NA code row
        // dropped (STAT-17): table [4, 2, 1, 4], p 0.2424242424242425, sample OR 8, MLE
        // 6.439146409959153 (tight). Fails before: unique() counts the null, "requires exactly
        // 2 unique values ... got 3"; on complete rows the Debug order "Int64(10)" < "Int64(2)"
        // swaps the rows (sample OR 0.125).
        let (case, ds) = crate::test_support::load_case("stat_contingency_na_codes");
        let r =
            run_fisher_test(&ds, "code", "out", FisherAlternative::TwoSided, Some(0.95)).unwrap();
        let (lo, hi) = r.odds_ratio_ci.unwrap();
        Check::new(&case)
            .vector("code_out_table", &r.table, Tol::exact())
            .scalar("code_out_fisher_p", r.p_value, Tol::p_value(11))
            .scalar("code_out_sample_or", r.odds_ratio, Tol::exact())
            .scalar("code_out_fisher_estimate_tight", r.conditional_mle, TIGHT)
            .scalar("code_out_fisher_ci_lower_tight", lo, TIGHT)
            .scalar("code_out_fisher_ci_upper_tight", hi, TIGHT)
            .scalar("code_out_fisher_estimate", r.conditional_mle, PRINTED)
            .scalar("code_out_fisher_ci_lower", lo, PRINTED)
            .scalar("code_out_fisher_ci_upper", hi, PRINTED)
            .finish();
        assert_eq!(r.row_levels, ["2", "10"]);
        assert_eq!(
            (r.missing.n_dropped, r.missing.by_column.clone()),
            (1, vec![("code".to_string(), 1)])
        );
        let shown = r.to_string();
        assert!(
            shown.contains("conditional MLE") && shown.contains("Missing values: 1 of 12 rows"),
            "{shown}"
        );
    }

    #[test]
    fn test_validate_fisher_needs_two_levels_per_column() {
        // x has 3 levels (p, q, r): Fisher's 2 x 2 test refuses it and names the levels
        let (_case, ds) = crate::test_support::load_case("stat_chisq_small_expected");
        let e = run_fisher_test(&ds, "x", "y", FisherAlternative::TwoSided, None)
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("'x' has 3 level(s) [p, q, r]") && e.contains("2 x 2"),
            "{e}"
        );
        assert!(matches!(
            run_fisher_test(&ds, "x", "nope", FisherAlternative::TwoSided, None),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_validate_run_fisher_test_inside_tokio_runtime() {
        // STAT-10: run_fisher_test builds its table without polars' lazy engine, so it runs
        // inside an async runtime.
        let (_case, ds) = crate::test_support::load_case("stat_fisher_rows_2000");
        let r = run_fisher_test(
            &ds,
            "treat",
            "outcome",
            FisherAlternative::TwoSided,
            Some(0.95),
        )
        .unwrap();
        assert!((r.p_value - 0.08111006057913338).abs() < 1e-12);
    }
}
