//! Power analysis for statistical tests.
//!
//! Implements power.t.test, power.prop.test, and power.anova.test from R stats.
//! These functions compute statistical power or determine parameters to achieve
//! a target power level.
//!
//! Every input finishes in bounded time, for any sample size or number of
//! groups: the t and F critical values come from quantile routines whose loops
//! have fixed bounds (statrs' own quantiles are used only at small degrees of
//! freedom and away from the far tails, see the "Bounded t and F quantiles"
//! section), and the solvers check the cooperative-cancellation budget
//! ([`crate::resource::checkpoint`]) once per bisection step.
//!
//! A significance level reaches the critical value as 1 - sig_level
//! (1 - sig_level / 2 when two-sided), which rounds it to a multiple of 2^-53:
//! by up to 1.1e-8 (relative) at sig_level = 1e-8 and 1.1e-4 at 1e-12. Levels
//! at or below 2^-53 (about 1.1e-16), where it rounds to 1 and the critical
//! value would be infinite, are refused.

use crate::errors::{EconError, EconResult};
use crate::resource::checkpoint;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, FisherSnedecor, Normal, StudentsT};

// ============================================================================
// Input validation
// ============================================================================

/// `Err` naming `name` unless `value` lies strictly between 0 and 1.
fn require_open_unit(name: &str, value: f64) -> EconResult<()> {
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("{name} must be between 0 and 1 (exclusive), got {value:?}"),
        })
    }
}

/// `Err` naming `name` unless `value` is finite and at least `min`.
fn require_finite_at_least(name: &str, value: f64, min: f64) -> EconResult<()> {
    if value.is_finite() && value >= min {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("{name} must be a finite number of at least {min}, got {value:?}"),
        })
    }
}

/// The significance level must be greater than this, 2^-53 (about 1.1e-16):
/// at or below it 1 - sig_level / 2 rounds to 1, the critical value is
/// infinite, and every finite non-centrality would get a power of 0.
const MIN_SIG_LEVEL: f64 = f64::EPSILON / 2.0;

/// `Err` unless `sig_level` lies strictly between [`MIN_SIG_LEVEL`] and 1.
fn require_sig_level(sig_level: f64) -> EconResult<()> {
    require_open_unit("sig_level", sig_level)?;
    if sig_level > MIN_SIG_LEVEL {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!(
                "sig_level must be greater than {MIN_SIG_LEVEL:?} (2^-53: at or below it \
                 1 - sig_level / 2 rounds to 1 and the critical value is infinite), \
                 got {sig_level:?}"
            ),
        })
    }
}

/// `Err` naming `name` unless `value` is finite and greater than 0.
fn require_finite_positive(name: &str, value: f64) -> EconResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("{name} must be a finite number greater than 0, got {value:?}"),
        })
    }
}

// ============================================================================
// Bounded t and F quantiles
// ============================================================================
//
// statrs 0.18 computes t and F quantiles by inverting the regularized
// incomplete beta function with AS 64/AS 109. That Newton loop has no
// iteration cap: it starts from a guess clamped to [1e-4, 1 - 1e-4] and scales
// each step by the reciprocal beta density there, which overflows once a shape
// parameter (df / 2) passes about 7e6. The step is then NaN and the loop never
// ends, so a t quantile at df = 1.5e7 or an F quantile at df2 = 2e7 hangs the
// thread, and no deadline can stop it from outside. Well before that the
// result also drifts: compared with R, statrs' t quantile is off by 3e-7 at
// df = 3e4 and 1e-4 at df = 1e6, and its F quantile by 3e-8 at (2, 1e4) and
// 1e-4 at (1, 5e5). Far in the lower tail the same overflow happens at small
// degrees of freedom (a t quantile at df = 1e3 and p = 1e-100 never returns),
// and below p = 1e-154 the starting point squares p to 0 and statrs panics.
//
// statrs folds the upper tail onto the lower one (the t quantile inverts the
// beta function at 2 (1 - p), and the inversion works on 1 - x when x > 0.5),
// so a probability near 1 reaches the same loop as the matching small tail
// probability. In the small tails statrs is inaccurate at small degrees of
// freedom even where it returns. Compared with R, its t quantile at df = 1 is
// 3.48e7 at tail 1e-8 (R's `qt`: 3.18e7) and never passes about 9.2e7 at
// smaller tails (R: 3.18e9 at 1e-10, 1.4e15 at 2^-52); at df = 1.05 it is
// still 4e-11 off at tail 1e-6. Its F quantile, at tails of 1e-7 to 1e-15, is
// off by up to 5e-7 in the upper tail with df1 < df2 <= 4, by up to 0.1 in the
// upper tail with df2 < df1 (and wholly wrong at df2 = 1), and by orders of
// magnitude in the lower tail at df1 = 1. Outside the upper tail with
// df1 < df2 it is off by 1e-2 to 5e-2 already at a tail of 1e-6 (at (1000, 1)
// in the upper tail and at (1, 1000) in the lower).
//
// The functions below therefore use statrs' `inverse_cdf` only at small
// degrees of freedom and with both tails (p and 1 - p) of at least
// STATRS_MIN_P = 1e-5, and the F quantile only in the upper tail with df1 < df2,
// the shape `power_anova_test` asks for. There its start is far from overflow,
// though its result is not everywhere close to full precision. Compared with
// R (quantiles polished by Newton steps on `pt` and `pf`), on grids of df
// from 1 to 1e3 and of tails from 1e-5 to 0.5, its t quantile is within 2e-12
// (relative) for tails up to 0.45 and within 2e-11 (absolute) nearer the
// median, except in a band of tails between about 0.12 and 0.21 that widens
// with df (0.145 to 0.163 at df = 50, 0.129 to 0.188 at 500, 0.12 to 0.207 at
// 1e3). In the band the error grows with df: 7e-12 at df = 50, 6.3e-11 at
// 150, 5.3e-10 at 400 and 4.7e-9 at 1e3 (at a tail of 0.207). Its F quantile
// is within 1e-8: F(1, df2) at p is the square of t(df2) at (1 + p) / 2, so at
// df1 = 1 the same band (F tails of 0.24 to 0.415) doubles the t error, to
// 9.4e-9 at (1, 1e3) and a tail of 0.415; it is within 3.6e-9 at df1 = 1.5,
// 1.3e-9 at df1 = 2 and 5e-10 from df1 = 3 on. Errors of that size move a
// power by about 1e-9.
//
// Everywhere else the functions use loops of their own with fixed bounds, on
// top of statrs' incomplete beta function (whose continued fraction stops
// after 140 terms), or closed-form large-df limits. (The power functions ask
// for 1 - sig_level and 1 - sig_level / 2, whose upper tails go down to
// 1.1e-16.)

/// Largest degrees of freedom for which [`t_quantile`] calls statrs.
const T_STATRS_MAX_DF: f64 = 1e3;

/// Largest degrees of freedom (both of them) for which [`f_quantile`] calls
/// statrs.
const F_STATRS_MAX_DF: f64 = 1e3;

/// Largest smaller degrees of freedom for which [`f_quantile`] uses the
/// bisection or the expansion; above it (both degrees of freedom large),
/// Paulson's approximation.
const F_SMALL_DF: f64 = 1e4;

/// Largest degrees of freedom at which [`f_quantile_by_bisection`] is used:
/// statrs' incomplete beta function, and so the bisection, is accurate to
/// about 2e-9 up to here when the other degrees of freedom are at most
/// [`F_SMALL_DF`].
const F_BISECTION_MAX_DF: f64 = 1e6;

/// Smallest tail probability, p or 1 - p, passed to statrs' quantiles: below
/// it statrs' t quantile is 4e-11 off at df = 1.05 (at a tail of 1e-6) and
/// grossly off further out (see above), while the bisection that replaces it
/// is within 3e-14 of R.
const STATRS_MIN_P: f64 = 1e-5;

/// Degrees of freedom above which [`t_cdf`] uses its large-df expansion
/// instead of statrs: statrs' t distribution function, through its incomplete
/// beta function, is 1e-8 off R's `pt` at df = 1e10 and 0.05 off at 1e15.
const T_CDF_EXPANSION_MIN_DF: f64 = 1e6;

/// Standard normal quantile (statrs' `erfc_inv`: rational approximations,
/// no iteration).
fn std_normal_quantile(p: f64) -> f64 {
    -std::f64::consts::SQRT_2 * statrs::function::erf::erfc_inv(2.0 * p)
}

/// `Err` unless `p` is a probability in [0, 1].
fn check_quantile_probability(p: f64) -> EconResult<()> {
    if (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(EconError::Computation(format!(
            "quantile probability must be in [0, 1], got {p:?}"
        )))
    }
}

/// The incomplete beta function I_x(a, b), as an [`EconResult`].
fn beta_reg(a: f64, b: f64, x: f64) -> EconResult<f64> {
    statrs::function::beta::checked_beta_reg(a, b, x)
        .map_err(|e| EconError::Computation(format!("incomplete beta function error: {}", e)))
}

/// The positive point where `above` turns from true to false, for a predicate
/// "the root lies above x" that is true below the root and false beyond it.
///
/// Brackets the root by doubling or halving from 1, then bisects. Every loop
/// has a fixed bound (at most 1,075 doublings or halvings, then 64 bisection
/// steps, enough to reach adjacent floats), so the search ends for any
/// predicate. A root beyond the range of `f64` comes back as infinity or 0.
fn bisect_root(above: impl Fn(f64) -> EconResult<bool>) -> EconResult<f64> {
    let (mut lo, mut hi);
    if above(1.0)? {
        (lo, hi) = (1.0_f64, 2.0_f64);
        while above(hi)? {
            lo = hi;
            hi *= 2.0;
            if hi.is_infinite() {
                return Ok(f64::INFINITY);
            }
        }
    } else {
        (lo, hi) = (0.5_f64, 1.0_f64);
        while !above(lo)? {
            hi = lo;
            lo *= 0.5;
            if lo == 0.0 {
                return Ok(0.0);
            }
        }
    }
    // the root lies in (lo, hi]
    for _ in 0..64 {
        let mid = lo + 0.5 * (hi - lo);
        if mid <= lo || mid >= hi {
            break;
        }
        if above(mid)? {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(lo + 0.5 * (hi - lo))
}

/// Quantile of Student's t with `df` degrees of freedom at probability `p`.
///
/// Up to [`T_STATRS_MAX_DF`]: statrs' `inverse_cdf` when p and 1 - p are both
/// at least [`STATRS_MIN_P`], within 2e-12 (relative) of R for tails up to
/// 0.45 (2e-11 absolute nearer the median) except in a band of tails between
/// about 0.12 and 0.21, where it is up to 4.7e-9 off at df = 1e3 (see the
/// "Bounded t and F quantiles" section), and otherwise bisection on the
/// distribution function, within 3e-14. Above it,
/// the Cornish-Fisher expansion about the normal quantile (Abramowitz &
/// Stegun 26.7.5) to the df^-4 term: just above the switch (df = 1001) within
/// 7e-13 for tails down to 1e-6, 3e-11 at 1e-12 and 1.2e-10 at 1.1e-16, the
/// smallest tail the power functions ask for, and within 1e-14 from df = 1e4
/// on.
///
/// The bisection works on df / (df + m^2), so m^2 must stay finite and the
/// magnitude it returns saturates at about 1.3e154. At df below about 2.1 that
/// clamps every tail below a floor of about 2.4e-155 at df = 1, 2.8e-309 at
/// df = 2, falling below the smallest subnormal near df = 2.1:
/// `t_quantile(1.0, 1e-300)` is -1.3e154 where R gives -3.2e299. The power
/// functions never ask for tails below 1.1e-16.
fn t_quantile(df: f64, p: f64) -> EconResult<f64> {
    check_quantile_probability(p)?;
    if df.is_nan() || df <= 0.0 {
        return Err(EconError::Computation(format!(
            "t distribution needs positive degrees of freedom, got {df:?}"
        )));
    }
    if p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    if df <= T_STATRS_MAX_DF {
        // 1 - p is exact for p > 0.5, so both tails are matched alike
        let tail = p.min(1.0 - p);
        if tail < STATRS_MIN_P {
            // P(T < -m) = P(T > m) = I_{df / (df + m^2)}(df / 2, 1 / 2) / 2
            let magnitude =
                bisect_root(|m| Ok(0.5 * beta_reg(df / 2.0, 0.5, df / (df + m * m))? > tail))?;
            return Ok(if p > 0.5 { magnitude } else { -magnitude });
        }
        let t = StudentsT::new(0.0, 1.0, df)
            .map_err(|e| EconError::Computation(format!("t distribution error: {}", e)))?;
        return Ok(t.inverse_cdf(p));
    }
    let z = std_normal_quantile(p);
    let z2 = z * z;
    let g1 = z * (z2 + 1.0) / 4.0;
    let g2 = z * ((5.0 * z2 + 16.0) * z2 + 3.0) / 96.0;
    let g3 = z * (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) / 384.0;
    let g4 = z * ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) / 92160.0;
    Ok(z + (g1 + (g2 + (g3 + g4 / df) / df) / df) / df)
}

/// Central t distribution function, with the large-df expansion above
/// [`T_CDF_EXPANSION_MIN_DF`] degrees of freedom.
fn t_cdf(x: f64, df: f64) -> EconResult<f64> {
    // An infinite x must not reach the expansion, where it would give
    // 0 * inf = NaN.
    if x.is_infinite() {
        return Ok(if x > 0.0 { 1.0 } else { 0.0 });
    }
    if df > T_CDF_EXPANSION_MIN_DF {
        // F(x) = Phi(x) - phi(x) (x^3 + x) / (4 df) + O(df^-2)
        let density = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        return Ok(Normal::standard().cdf(x) - density * x * (x * x + 1.0) / (4.0 * df));
    }
    let t = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| EconError::Computation(format!("t distribution error: {}", e)))?;
    Ok(t.cdf(x))
}

/// Quantile of the F distribution with (`df1`, `df2`) degrees of freedom at
/// probability `p`.
///
/// - Both degrees of freedom at most [`F_STATRS_MAX_DF`], df1 < df2, p > 0.5
///   and 1 - p at least [`STATRS_MIN_P`] (the upper tail that
///   `power_anova_test` asks for): statrs' `inverse_cdf`.
/// - Otherwise, the smaller at most [`F_SMALL_DF`] and the larger at most
///   [`F_BISECTION_MAX_DF`]: bisection on the distribution function
///   ([`f_quantile_by_bisection`]).
/// - The smaller at most [`F_SMALL_DF`] and the larger above
///   [`F_BISECTION_MAX_DF`]: the quantile at [`F_BISECTION_MAX_DF`] carried to
///   the larger degrees of freedom by its expansion ([`f_quantile_large_df2`]).
/// - Both above [`F_SMALL_DF`]: Paulson's (1942) normal approximation
///   ([`paulson_f_quantile`]).
///
/// Compared with R (quantiles found by root-finding on `pf`) on a grid of
/// degrees of freedom and of tails (p or 1 - p) from 1.1e-16 to 0.5, the
/// result is within 1e-8 (relative) in the first case (9.4e-9 at (1, 1e3) and
/// a tail of 0.415, see the "Bounded t and F quantiles" section), within 2.2e-9
/// in the second (at (1, 1e6) and a tail of 0.09) and within 2e-6 in the third
/// (1.6e-6 at df1 = 1e4 and df2 = 1e15). Paulson's approximation is
/// within 1.5e-6 for tails down to 1e-6 and within 1e-5 down to 1.1e-16
/// (8.6e-6 at (1.1e4, 1.1e4) and tail 2^-53, just above the switch; its error
/// falls like 1 / df).
fn f_quantile(df1: f64, df2: f64, p: f64) -> EconResult<f64> {
    check_quantile_probability(p)?;
    if !(df1.is_finite() && df1 > 0.0 && df2.is_finite() && df2 > 0.0) {
        return Err(EconError::Computation(format!(
            "F distribution needs finite positive degrees of freedom, got ({df1:?}, {df2:?})"
        )));
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    let (small, large) = (df1.min(df2), df1.max(df2));
    // statrs only where it is accurate: the upper tail with df1 < df2, which is
    // what power_anova_test asks for (see the doc above)
    if large <= F_STATRS_MAX_DF && df1 < df2 && p > 0.5 && 1.0 - p >= STATRS_MIN_P {
        let f = FisherSnedecor::new(df1, df2)
            .map_err(|e| EconError::Computation(format!("F distribution error: {}", e)))?;
        return Ok(f.inverse_cdf(p));
    }
    if small > F_SMALL_DF {
        return Ok(paulson_f_quantile(df1, df2, std_normal_quantile(p)));
    }
    // 1 - p is exact for p >= 0.5, and the callees use it only there
    let q = 1.0 - p;
    if large <= F_BISECTION_MAX_DF {
        return f_quantile_by_bisection(df1, df2, p, q);
    }
    if df1 <= df2 {
        f_quantile_large_df2(df1, df2, p, q)
    } else {
        // F(df1, df2) at p is 1 / F(df2, df1) at 1 - p. Passing p as that
        // quantile's upper tail keeps a small p exact, where 1 - (1 - p)
        // would round it (by 0.08% at p = 1e-15).
        Ok(1.0 / f_quantile_large_df2(df2, df1, q, p)?)
    }
}

/// F quantile by bisection ([`bisect_root`]) on the distribution function,
/// computed from statrs' regularized incomplete beta function, at lower-tail
/// probability `p` and upper-tail probability `q` = 1 - p. When p > 0.5 the
/// quantile is matched on the survival function at `q`, otherwise on the
/// distribution function at `p`, so that neither tail loses digits to
/// cancellation; the caller passes both because it may hold the upper tail
/// exactly where 1 - p would round it.
fn f_quantile_by_bisection(df1: f64, df2: f64, p: f64, q: f64) -> EconResult<f64> {
    let (a, b) = (df1 / 2.0, df2 / 2.0);
    bisect_root(|x| {
        let t = df1 * x;
        let s = t + df2;
        if !s.is_finite() {
            return Ok(false);
        }
        Ok(if p > 0.5 {
            beta_reg(b, a, df2 / s)? > q
        } else {
            beta_reg(a, b, t / s)? < p
        })
    })
}

/// F quantile for `df2` above [`F_BISECTION_MAX_DF`] and `df1` at most
/// [`F_SMALL_DF`].
///
/// With c the chi-square(df1) quantile at p, the quantile expands as
/// F = (c / df1) (1 + (c - df1 + 2) / (2 df2)) + O(df2^-2). The quantile at
/// df2 = [`F_BISECTION_MAX_DF`] fixes the limit c / df1, which is then carried
/// to the requested df2. `p` and `q` = 1 - p are the lower and upper tail, as
/// in [`f_quantile_by_bisection`].
fn f_quantile_large_df2(df1: f64, df2: f64, p: f64, q: f64) -> EconResult<f64> {
    let anchor_df2 = F_BISECTION_MAX_DF;
    let at_anchor = f_quantile_by_bisection(df1, anchor_df2, p, q)?;
    // Solve at_anchor = limit (1 + (df1 limit - df1 + 2) / (2 anchor_df2)) by
    // fixed-point iteration; the map contracts by about
    // df1 limit / (2 anchor_df2), below 0.01 here.
    let mut limit = at_anchor;
    for _ in 0..8 {
        limit = at_anchor / (1.0 + (df1 * limit - df1 + 2.0) / (2.0 * anchor_df2));
    }
    Ok(limit * (1.0 + (df1 * limit - df1 + 2.0) / (2.0 * df2)))
}

/// Paulson's (1942) approximation to the F quantile at the standard normal
/// quantile `z`: with a = 2/(9 df1) and b = 2/(9 df2),
/// ((1 - b) u - (1 - a)) / sqrt(a + b u^2) is standard normal for u = F^(1/3).
/// Its error shrinks like 1/df; it is used with both degrees of freedom above
/// [`F_SMALL_DF`].
fn paulson_f_quantile(df1: f64, df2: f64, z: f64) -> f64 {
    let a = 2.0 / (9.0 * df1);
    let b = 2.0 / (9.0 * df2);
    let spread = (a * (1.0 - b).powi(2) + b * (1.0 - a).powi(2) - z * z * a * b).max(0.0);
    let u = ((1.0 - a) * (1.0 - b) + z * spread.sqrt()) / ((1.0 - b).powi(2) - z * z * b);
    u.powi(3)
}

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
/// A `PowerTTestResult` with all parameters filled in. The run time does not
/// depend on `n`: the t critical values come from quantile routines with
/// fixed loop bounds (see the module documentation).
///
/// # Errors
///
/// [`EconError::InvalidSpecification`], naming the parameter and its limit,
/// when
/// - not exactly one of the five parameters is `None`, or the one left out is
///   `sd` or `sig_level` (solving for them is not implemented);
/// - `n` is not finite or is below 2;
/// - `delta` is not finite;
/// - `sd` is not finite or not greater than 0, or, when solving for `delta`,
///   above `f64::MAX / 10` (the search runs up to `10 * sd`);
/// - `sig_level` is not strictly between 2^-53 (about 1.1e-16) and 1 (NaN
///   included): at or below 2^-53, 1 - sig_level / 2 rounds to 1 and the
///   critical value would be infinite;
/// - `power` is not strictly between 0 and 1 (NaN included).
///
/// [`EconError::Cancelled`] when the cooperative-cancellation budget
/// ([`crate::resource`]) runs out while solving for `n` or `delta`.
///
/// [`EconError::Computation`] if a distribution routine fails or the power
/// comes out undefined (NaN); not expected for inputs that pass the checks
/// above.
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
    let none_count = [
        n.is_none(),
        delta.is_none(),
        sd.is_none(),
        sig_level.is_none(),
        power.is_none(),
    ]
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

    // Validate parameters (NaN and infinities included: they would reach the
    // statrs quantiles, which panic or never return on them)
    if let Some(n_val) = n {
        require_finite_at_least("n", n_val, 2.0)?;
    }
    if let Some(d) = delta
        && !d.is_finite()
    {
        return Err(EconError::InvalidSpecification {
            message: format!("delta must be a finite number, got {d:?}"),
        });
    }
    require_finite_positive("sd", sd)?;
    if delta.is_none() && !(10.0 * sd).is_finite() {
        // the delta search runs up to 10 * sd
        let max_sd = f64::MAX / 10.0;
        return Err(EconError::InvalidSpecification {
            message: format!("sd must be at most {max_sd:e} when solving for delta, got {sd:?}"),
        });
    }
    require_sig_level(sig_level)?;
    if let Some(p) = power {
        require_open_unit("power", p)?;
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
            message:
                "Solving for sd or sig_level not yet implemented. Provide them as known values."
                    .to_string(),
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

    // Non-centrality |delta| / se, with the standard error se = sd sqrt(2 / n)
    // for two samples and sd / sqrt(n) otherwise, which R's power.t.test
    // writes as sqrt(n / tsample) delta / sd. Formed as |delta| / sd times
    // sqrt(n / tsample), with n >= 2: |delta| / sd underflows only when the
    // non-centrality is negligible and overflows only when it is beyond
    // f64::MAX, and delta = 0 gives exactly 0. Dividing by se instead turns
    // its underflow (delta = 5e-324, sd = 1e-174, n = 1e300) into an infinite
    // non-centrality where the true one is 3.5, and delta = 0 into 0 / 0.
    let tsample = match test_type {
        TTestType::TwoSample => 2.0,
        TTestType::OneSample | TTestType::Paired => 1.0,
    };
    let ncp = delta.abs() / sd * (n / tsample).sqrt();

    // Critical value
    let t_crit = match alternative {
        PowerAlternative::TwoSided => t_quantile(df, 1.0 - sig_level / 2.0)?,
        PowerAlternative::OneSided => t_quantile(df, 1.0 - sig_level)?,
    };
    let power = match alternative {
        PowerAlternative::TwoSided => {
            // Power = P(|T| > t_crit) under alternative
            // Using non-central t distribution approximation
            let nct_upper = 1.0 - nct_cdf(t_crit, df, ncp)?;
            let nct_lower = nct_cdf(-t_crit, df, ncp)?;
            nct_upper + nct_lower
        }
        PowerAlternative::OneSided => {
            if delta >= 0.0 {
                1.0 - nct_cdf(t_crit, df, ncp)?
            } else {
                nct_cdf(-t_crit, df, -ncp)?
            }
        }
    };

    // the clamp below would turn a NaN into 0 without a word
    if power.is_nan() {
        return Err(EconError::Computation(format!(
            "power is undefined for non-centrality {ncp:?} and critical value {t_crit:?}"
        )));
    }
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
        checkpoint()?;
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
    // Bisection search for delta; power_t_test keeps 10 * sd finite, and the
    // midpoint is formed so that low + high cannot overflow near f64::MAX
    let mut low = 0.0;
    let mut high = 10.0 * sd;

    for _ in 0..100 {
        checkpoint()?;
        let mid = 0.5 * low + 0.5 * high;
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

    Ok(0.5 * low + 0.5 * high)
}

/// Non-central t CDF approximation.
/// Uses the approximation from Lenth (1989).
fn nct_cdf(t: f64, df: f64, ncp: f64) -> EconResult<f64> {
    // For ncp = 0, this is just the central t
    if ncp.abs() < 1e-10 {
        return t_cdf(t, df);
    }

    // An infinite non-centrality (the true one is beyond f64::MAX) puts all
    // the mass at +inf or -inf. The limit holds for every t, an infinite one
    // included, where the approximations below would compute inf - inf = NaN.
    if ncp.is_infinite() {
        return Ok(if ncp > 0.0 { 0.0 } else { 1.0 });
    }

    // Use normal approximation for large df
    if df > 100.0 {
        let normal = Normal::new(ncp, 1.0)
            .map_err(|e| EconError::Computation(format!("Normal distribution error: {}", e)))?;
        return Ok(normal.cdf(t));
    }

    // Approximation using series expansion
    // P(T < t | ncp) ≈ Φ(z) where z is adjusted for non-centrality
    let z = t * (1.0 - 1.0 / (4.0 * df)).sqrt() - ncp;
    Ok(Normal::standard().cdf(z))
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
///
/// The run time does not depend on `n`.
///
/// # Errors
///
/// [`EconError::InvalidSpecification`], naming the parameter and its limit,
/// when
/// - `p2` is `None`, or not exactly one of `n` and `power` is `None`;
/// - `p1`, `p2` or `power` is not strictly between 0 and 1 (NaN included);
/// - `sig_level` is not strictly between 2^-53 (about 1.1e-16) and 1 (NaN
///   included): at or below 2^-53, 1 - sig_level / 2 rounds to 1 and the
///   critical value would be infinite;
/// - `n` is not finite or is below 1.
///
/// [`EconError::Cancelled`] when the cooperative-cancellation budget
/// ([`crate::resource`]) runs out while solving for `n`.
///
/// [`EconError::Computation`] if a distribution routine fails or the power
/// comes out undefined (NaN); not expected for inputs that pass the checks
/// above.
pub fn power_prop_test(
    n: Option<f64>,
    p1: f64,
    p2: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    alternative: PowerAlternative,
) -> EconResult<PowerPropTestResult> {
    let sig_level = sig_level.unwrap_or(0.05);

    // Validate (NaN included: the comparisons below would let it through)
    require_open_unit("p1", p1)?;

    let p2 = p2.ok_or_else(|| EconError::InvalidSpecification {
        message: "p2 must be provided".to_string(),
    })?;

    require_open_unit("p2", p2)?;
    require_sig_level(sig_level)?;
    if let Some(p) = power {
        require_open_unit("power", p)?;
    }
    if let Some(n_val) = n {
        // as R's power.prop.test: at least one observation per group
        require_finite_at_least("n", n_val, 1.0)?;
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
    let _h = 2.0 * (p1.sqrt().asin() - p2.sqrt().asin());

    // Non-centrality |p1 - p2| / se, with the standard error under the null
    // se = sqrt(2 p q / n) (p the pooled proportion, q = 1 - p), formed
    // without se: |p1 - p2| / sqrt(2 p q) is at most sqrt(2) and its
    // denominator at least 3e-162, so the non-centrality stays finite (below
    // 2e154) for any n and is exactly 0 when p1 == p2. Dividing by se instead
    // turns its underflow to 0 (p1 = 5e-324, p2 = 1e-323, n = 100) into an
    // infinite non-centrality where the true one is about 1e-161.
    let p_pooled = (p1 + p2) / 2.0;
    let ncp = (p1 - p2).abs() / (2.0 * p_pooled * (1.0 - p_pooled)).sqrt() * n.sqrt();

    let normal = Normal::standard();
    let alt = Normal::new(ncp, 1.0)
        .map_err(|e| EconError::Computation(format!("Normal distribution error: {e}")))?;

    let z_crit = match alternative {
        PowerAlternative::TwoSided => normal.inverse_cdf(1.0 - sig_level / 2.0),
        PowerAlternative::OneSided => normal.inverse_cdf(1.0 - sig_level),
    };
    let power = match alternative {
        PowerAlternative::TwoSided => {
            let power_upper = 1.0 - alt.cdf(z_crit);
            let power_lower = alt.cdf(-z_crit);
            power_upper + power_lower
        }
        PowerAlternative::OneSided => 1.0 - alt.cdf(z_crit),
    };

    // the clamp below would turn a NaN into 0 without a word
    if power.is_nan() {
        return Err(EconError::Computation(format!(
            "power is undefined for non-centrality {ncp:?} and critical value {z_crit:?}"
        )));
    }
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
        checkpoint()?;
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
///
/// The run time does not depend on `n` or `groups`: the F critical values
/// come from quantile routines with fixed loop bounds (see the module
/// documentation).
///
/// # Errors
///
/// [`EconError::InvalidSpecification`], naming the parameter and its limit,
/// when
/// - `groups` is below 2, or not exactly one of `n` and `power` is `None`;
/// - `n` is not finite, is below 2, or is so large that the error degrees of
///   freedom, `groups * (n - 1)`, are not finite (above about
///   `f64::MAX / groups`; the message names the largest n accepted);
/// - `between_var` is not finite or is negative;
/// - `within_var` is not finite or not greater than 0;
/// - `sig_level` is not strictly between 2^-53 (about 1.1e-16) and 1 (NaN
///   included): at or below 2^-53, as in [`power_t_test`] and
///   [`power_prop_test`], the critical value can be infinite;
/// - `power` is not strictly between 0 and 1 (NaN included).
///
/// [`EconError::Cancelled`] when the cooperative-cancellation budget
/// ([`crate::resource`]) runs out while solving for `n`.
///
/// [`EconError::Computation`] if a distribution routine fails; not expected
/// for inputs that pass the checks above.
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
            message: format!("groups must be at least 2, got {groups}"),
        });
    }

    let sig_level = sig_level.unwrap_or(0.05);

    // Validate (NaN and infinities included: they would reach the statrs
    // quantiles, which panic on a probability outside [0, 1])
    if let Some(n_val) = n {
        require_finite_at_least("n", n_val, 2.0)?;
        let g = groups as f64;
        if !(g * (n_val - 1.0)).is_finite() {
            // the largest n the check above accepts: f64::MAX / g itself fails
            // it when g * (f64::MAX / g - 1) rounds up to infinity (groups not
            // a power of two, e.g. 3), and one step down is then enough
            let mut max_n = f64::MAX / g;
            if !(g * (max_n - 1.0)).is_finite() {
                max_n = max_n.next_down();
            }
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "n must be at most {max_n:e} for groups = {groups} (groups * (n - 1), \
                     the error degrees of freedom, must be finite), got {n_val:e}"
                ),
            });
        }
    }
    require_finite_at_least("between_var", between_var, 0.0)?;
    require_finite_positive("within_var", within_var)?;
    require_sig_level(sig_level)?;
    if let Some(p) = power {
        require_open_unit("power", p)?;
    }

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
    let f_crit = f_quantile(df1, df2, 1.0 - sig_level)?;

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
        checkpoint()?;
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
        _ => {
            return Err(EconError::InvalidSpecification {
                message: format!("Unknown test type: {}", test_type),
            });
        }
    };

    let alternative = match alternative.to_lowercase().as_str() {
        "two.sided" | "two_sided" | "twosided" => PowerAlternative::TwoSided,
        "one.sided" | "one_sided" | "onesided" => PowerAlternative::OneSided,
        _ => {
            return Err(EconError::InvalidSpecification {
                message: format!("Unknown alternative: {}", alternative),
            });
        }
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
        _ => {
            return Err(EconError::InvalidSpecification {
                message: format!("Unknown alternative: {}", alternative),
            });
        }
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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

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
        )
        .unwrap();

        // Should need around 100 per group
        assert!(result.n > 50.0 && result.n < 150.0);
    }

    #[test]
    fn test_power_anova() {
        // Power for 3-group ANOVA
        let result = power_anova_test(
            3,
            Some(20.0),
            0.25, // between variance
            1.0,  // within variance
            Some(0.05),
            None,
        )
        .unwrap();

        assert!(result.power > 0.0 && result.power < 1.0);
    }

    #[test]
    fn test_power_anova_solve_n() {
        let result = power_anova_test(3, None, 0.25, 1.0, Some(0.05), Some(0.8)).unwrap();

        // Should need moderate n per group
        assert!(result.n > 10.0);
    }

    // =========================================================================
    // Validation tests against R
    // =========================================================================

    #[test]
    fn test_validate_power_t_test_compute_power() {
        // R: power.t.test(n = 30, delta = 0.5, sd = 1, sig.level = 0.05, type = "two.sample")
        // R result: power = 0.477841
        let result = power_t_test(
            Some(30.0),
            Some(0.5),
            Some(1.0),
            Some(0.05),
            None,
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        )
        .unwrap();

        // Allow 10% tolerance due to approximation differences
        let expected = 0.477841;
        assert!(
            (result.power - expected).abs() < 0.1,
            "Power mismatch: Rust={:.6}, R={:.6}",
            result.power,
            expected
        );
    }

    #[test]
    fn test_validate_power_t_test_solve_n() {
        // R: power.t.test(delta = 0.5, sd = 1, sig.level = 0.05, power = 0.8, type = "two.sample")
        // R result: n = 63.8
        let result = power_t_test(
            None,
            Some(0.5),
            Some(1.0),
            Some(0.05),
            Some(0.8),
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        )
        .unwrap();

        let expected = 63.8;
        // Allow wider tolerance since we're solving iteratively
        assert!(
            (result.n - expected).abs() < 15.0,
            "n mismatch: Rust={:.1}, R={:.1}",
            result.n,
            expected
        );
    }

    #[test]
    fn test_validate_power_prop_test_compute_power() {
        // R: power.prop.test(n = 100, p1 = 0.3, p2 = 0.5, sig.level = 0.05)
        // R result: power = 0.828109
        let result = power_prop_test(
            Some(100.0),
            0.3,
            Some(0.5),
            Some(0.05),
            None,
            PowerAlternative::TwoSided,
        )
        .unwrap();

        let expected = 0.828109;
        assert!(
            (result.power - expected).abs() < 0.15,
            "Power mismatch: Rust={:.6}, R={:.6}",
            result.power,
            expected
        );
    }

    #[test]
    fn test_validate_power_prop_test_solve_n() {
        // R: power.prop.test(p1 = 0.3, p2 = 0.5, sig.level = 0.05, power = 0.8)
        // R result: n = 93.0
        let result = power_prop_test(
            None,
            0.3,
            Some(0.5),
            Some(0.05),
            Some(0.8),
            PowerAlternative::TwoSided,
        )
        .unwrap();

        let expected = 93.0;
        assert!(
            (result.n - expected).abs() < 20.0,
            "n mismatch: Rust={:.1}, R={:.1}",
            result.n,
            expected
        );
    }

    #[test]
    fn test_validate_power_anova_structure() {
        // Test that ANOVA power function returns reasonable structure
        // The non-central F approximation may differ from R's exact implementation
        let result = power_anova_test(3, Some(20.0), 0.25, 1.0, Some(0.05), None).unwrap();

        // Check structure
        assert_eq!(result.groups, 3);
        assert!((result.n - 20.0).abs() < 1e-10);
        assert!((result.between_var - 0.25).abs() < 1e-10);
        assert!((result.within_var - 1.0).abs() < 1e-10);
        assert!((result.sig_level - 0.05).abs() < 1e-10);

        // Power should be between 0 and 1
        assert!(
            result.power >= 0.0 && result.power <= 1.0,
            "Power should be in [0,1]: {:.6}",
            result.power
        );

        // Method description should be filled
        assert!(!result.method.is_empty());
    }

    #[test]
    fn test_validate_power_anova_positive_power() {
        // Power should be positive and between 0 and 1
        let result = power_anova_test(3, Some(50.0), 0.25, 1.0, Some(0.05), None).unwrap();

        // Power should be in valid range
        assert!(
            result.power > 0.0 && result.power <= 1.0,
            "Power should be in (0, 1]: got {:.4}",
            result.power
        );

        // Result should have correct structure
        assert_eq!(result.groups, 3);
        assert!((result.n - 50.0).abs() < 1e-10);
    }

    // =========================================================================
    // Extreme sample sizes and group counts (PR-01): these used to spin
    // forever inside statrs' t and F quantiles once the degrees of freedom
    // passed about 1.4e7.
    // =========================================================================

    fn two_sample(n: f64, delta: f64, alternative: PowerAlternative) -> PowerTTestResult {
        power_t_test(
            Some(n),
            Some(delta),
            Some(1.0),
            Some(0.05),
            None,
            TTestType::TwoSample,
            alternative,
        )
        .unwrap()
    }

    #[test]
    fn test_validate_power_t_test_ten_million_per_group() {
        // R: power.t.test(n = 1e7, delta = 0.001, sd = 1, strict = TRUE)$power
        let r = two_sample(1e7, 0.001, PowerAlternative::TwoSided);
        assert!(
            (r.power - 0.608_779_443_418_653_6).abs() < 1e-6,
            "{}",
            r.power
        );

        // R: power.t.test(n = 1e7, delta = 0.001, sd = 1, alternative = "one.sided")$power
        let r = two_sample(1e7, 0.001, PowerAlternative::OneSided);
        assert!(
            (r.power - 0.722_811_570_357_799_8).abs() < 1e-6,
            "{}",
            r.power
        );

        // R: power.t.test(n = 1e7, delta = 0.001, sd = 1, type = "one.sample", strict = TRUE)$power
        let r = power_t_test(
            Some(1e7),
            Some(0.001),
            Some(1.0),
            Some(0.05),
            None,
            TTestType::OneSample,
            PowerAlternative::TwoSided,
        )
        .unwrap();
        assert!(
            (r.power - 0.885_379_081_953_102).abs() < 1e-6,
            "{}",
            r.power
        );
    }

    #[test]
    fn test_validate_power_t_test_solve_delta_ten_million_per_group() {
        // R: power.t.test(n = 1e7, sd = 1, power = 0.8, strict = TRUE, tol = 1e-14)$delta
        // (R's default tol, 1.2e-4 on delta, is too loose for a delta this small)
        let r = power_t_test(
            Some(1e7),
            None,
            Some(1.0),
            Some(0.05),
            Some(0.8),
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        )
        .unwrap();
        let expected = 0.001_252_905_524_219_861_2;
        assert!(
            ((r.delta - expected) / expected).abs() < 1e-5,
            "{}",
            r.delta
        );
    }

    #[test]
    fn test_power_t_test_astronomical_n_is_full_power() {
        // R: power.t.test(n = 1e300, delta = 0.5)$power == 1
        for n in [1e9, 1e18, 1e300] {
            let r = two_sample(n, 0.5, PowerAlternative::TwoSided);
            assert!((r.power - 1.0).abs() < 1e-12, "n = {n}: {}", r.power);
        }
    }

    // The ANOVA power values themselves are off (STAT-34, a separate item:
    // the noncentral-F approximation is mis-derived), so the ANOVA tests here
    // check that every call returns, with a power or n in range; the F critical
    // value they use is checked against R in the f_quantile tests below. At
    // n = 1e300 the non-centrality is so large that Patnaik's h overflows,
    // FisherSnedecor::new rejects it and ncf_cdf returns its 0.5 fallback
    // (power 0.5, where the true value is 1; STAT-34): that case only checks
    // that the call terminates.

    fn assert_power_in_range(r: &PowerAnovaTestResult) {
        assert!((0.0..=1.0).contains(&r.power), "power {}", r.power);
    }

    #[test]
    fn test_power_anova_ten_million_per_group_returns() {
        for n in [1e7, 1e12, 1e300] {
            let r = power_anova_test(3, Some(n), 0.25, 1.0, Some(0.05), None).unwrap();
            assert_power_in_range(&r);
        }
    }

    #[test]
    fn test_power_anova_many_groups_solve_n_returns() {
        // 50 groups: the first bisection step tries n = 5e5, so df2 = 2.5e7.
        let r = power_anova_test(50, None, 0.001, 1.0, Some(0.05), Some(0.8)).unwrap();
        assert!((2.0..=1e6).contains(&r.n), "{}", r.n);
    }

    #[test]
    fn test_power_anova_astronomical_group_count_returns() {
        let r = power_anova_test(1usize << 62, Some(10.0), 1.0, 1.0, Some(0.05), None).unwrap();
        assert_power_in_range(&r);
        let r = power_anova_test(1usize << 62, None, 1.0, 1.0, Some(0.05), Some(0.8)).unwrap();
        assert!((2.0..=1e6).contains(&r.n), "{}", r.n);
    }

    #[test]
    fn test_power_solvers_stop_on_a_cancelled_budget() {
        use crate::resource::{Budget, CancelToken, with_budget};

        let token = CancelToken::new();
        token.cancel();
        let cancelled = |e: EconError| matches!(e, EconError::Cancelled(_));
        with_budget(Budget::new(None, token), || {
            let n = power_t_test(
                None,
                Some(0.5),
                Some(1.0),
                Some(0.05),
                Some(0.8),
                TTestType::TwoSample,
                PowerAlternative::TwoSided,
            );
            assert!(n.is_err_and(cancelled), "solve n (t)");
            let delta = power_t_test(
                Some(50.0),
                None,
                Some(1.0),
                Some(0.05),
                Some(0.8),
                TTestType::TwoSample,
                PowerAlternative::TwoSided,
            );
            assert!(delta.is_err_and(cancelled), "solve delta (t)");
            let prop = power_prop_test(
                None,
                0.3,
                Some(0.5),
                Some(0.05),
                Some(0.8),
                PowerAlternative::TwoSided,
            );
            assert!(prop.is_err_and(cancelled), "solve n (prop)");
            let anova = power_anova_test(3, None, 0.25, 1.0, Some(0.05), Some(0.8));
            assert!(anova.is_err_and(cancelled), "solve n (anova)");
        });
    }

    /// A refusal of parameter `name`: every validation message starts with
    /// "<name> must".
    fn assert_refused<T: std::fmt::Debug>(r: EconResult<T>, name: &str) {
        assert_refused_containing(r, name, "");
    }

    /// As [`assert_refused`], and the message also contains `detail`.
    fn assert_refused_containing<T: std::fmt::Debug>(r: EconResult<T>, name: &str, detail: &str) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(
                    message.starts_with(&format!("{name} must")),
                    "message {message:?} should start with \"{name} must\""
                );
                assert!(
                    message.contains(detail),
                    "message {message:?} should contain {detail:?}"
                );
            }
            other => panic!("expected a refusal naming {name}, got {other:?}"),
        }
    }

    #[test]
    fn test_power_t_test_rejects_non_finite_and_out_of_range_inputs() {
        let t = |n: Option<f64>, delta: Option<f64>, sd: f64, sig: f64, power: Option<f64>| {
            power_t_test(
                n,
                delta,
                Some(sd),
                Some(sig),
                power,
                TTestType::TwoSample,
                PowerAlternative::TwoSided,
            )
        };
        assert_refused(t(Some(f64::INFINITY), Some(0.5), 1.0, 0.05, None), "n");
        assert_refused(t(Some(f64::NAN), Some(0.5), 1.0, 0.05, None), "n");
        assert_refused(t(Some(30.0), Some(f64::NAN), 1.0, 0.05, None), "delta");
        assert_refused(t(Some(30.0), Some(f64::INFINITY), 1.0, 0.05, None), "delta");
        assert_refused(t(Some(30.0), Some(0.5), 0.0, 0.05, None), "sd");
        assert_refused(t(Some(30.0), Some(0.5), f64::NAN, 0.05, None), "sd");
        assert_refused(t(Some(30.0), Some(0.5), 1.0, f64::NAN, None), "sig_level");
        assert_refused(t(None, Some(0.5), 1.0, 0.05, Some(f64::NAN)), "power");
        // Solving for delta searches up to 10 * sd, which must stay finite
        assert_refused_containing(
            t(Some(30.0), None, 1e308, 0.05, Some(0.8)),
            "sd",
            "solving for delta",
        );
        // Large values are printed in exponent form, not digit by digit
        assert_refused_containing(
            t(Some(30.0), None, 1e308, 0.05, Some(0.8)),
            "sd",
            "at most 1.7976931348623158e307 when solving for delta, got 1e308",
        );
        assert_refused_containing(
            t(Some(30.0), Some(0.5), 1.0, 1e300, None),
            "sig_level",
            "got 1e300",
        );
        assert_refused_containing(
            t(Some(1e300), Some(0.5), -1e300, 0.05, None),
            "sd",
            "got -1e300",
        );
    }

    #[test]
    fn test_validate_power_t_test_infinite_non_centrality_is_full_power() {
        // R: power.t.test(n = 1e300, delta = 1, sd = 1e-300, sig.level = s,
        // strict = TRUE)$power == 1, for s = 0.05 and for the smallest level
        // accepted, just above 2^-53. The non-centrality, sqrt(n / 2) delta / sd,
        // is beyond f64::MAX, and the power is its limit. (At sig_level = 1e-17
        // the critical value is infinite as well, and the power once came out
        // there as inf - inf = NaN, clamped to 0. Levels at or below 2^-53 are
        // now refused, see
        // test_power_functions_refuse_a_sig_level_at_or_below_2_pow_minus_53.)
        let cases = [
            (1.0, PowerAlternative::TwoSided),
            (1.0, PowerAlternative::OneSided),
            (-1.0, PowerAlternative::OneSided),
        ];
        for (delta, alternative) in cases {
            for sig_level in [MIN_SIG_LEVEL.next_up(), 0.05] {
                let r = power_t_test(
                    Some(1e300),
                    Some(delta),
                    Some(1e-300),
                    Some(sig_level),
                    None,
                    TTestType::TwoSample,
                    alternative,
                )
                .unwrap();
                assert_eq!(
                    r.power, 1.0,
                    "{alternative:?}, delta {delta}, sig {sig_level}"
                );
            }
        }
    }

    #[test]
    fn test_validate_power_t_test_non_centrality_beside_an_underflowing_standard_error() {
        // The standard error sd sqrt(2 / n) underflows to 0 at sd = 1e-174 and
        // n = 1e300, but the non-centrality sqrt(n / 2) delta / sd is 3.5:
        // dividing delta by the underflowed standard error used to give an
        // infinite non-centrality and a power of 1.
        // R: power.t.test(n = 1e300, delta = 5e-324, sd = 1e-174, type = t,
        // strict = TRUE)$power
        let cases = [
            (TTestType::TwoSample, 0.937_436_928_340_223_3),
            (TTestType::OneSample, 0.998_562_012_785_837_1),
            (TTestType::Paired, 0.998_562_012_785_837_1),
        ];
        for (test_type, expected) in cases {
            let r = power_t_test(
                Some(1e300),
                Some(5e-324),
                Some(1e-174),
                Some(0.05),
                None,
                test_type,
                PowerAlternative::TwoSided,
            )
            .unwrap();
            assert!(
                (r.power - expected).abs() < 1e-6,
                "{test_type:?}: {}",
                r.power
            );
        }
    }

    #[test]
    fn test_power_functions_refuse_a_sig_level_at_or_below_2_pow_minus_53() {
        // At or below 2^-53, 1 - sig_level / 2 rounds to 1, the critical value
        // is infinite, and the power came out as 0 for any finite
        // non-centrality (R: power.prop.test(n = 1e300, p1 = 0.3, p2 = 0.5,
        // sig.level = 1e-300)$power is 1).
        let limit = "greater than 1.1102230246251565e-16";
        let two = PowerAlternative::TwoSided;
        for sig in [1e-300, 1e-17, MIN_SIG_LEVEL] {
            let t = power_t_test(
                Some(1e6),
                Some(1.0),
                Some(1.0),
                Some(sig),
                None,
                TTestType::TwoSample,
                two,
            );
            assert_refused_containing(t, "sig_level", limit);
            let t_n = power_t_test(
                None,
                Some(1.0),
                Some(1.0),
                Some(sig),
                Some(0.8),
                TTestType::OneSample,
                PowerAlternative::OneSided,
            );
            assert_refused_containing(t_n, "sig_level", limit);
            let prop = power_prop_test(Some(1e300), 0.3, Some(0.5), Some(sig), None, two);
            assert_refused_containing(prop, "sig_level", limit);
            let prop_n = power_prop_test(None, 0.3, Some(0.5), Some(sig), Some(0.8), two);
            assert_refused_containing(prop_n, "sig_level", limit);
            let anova = power_anova_test(3, Some(1e6), 0.25, 1.0, Some(sig), None);
            assert_refused_containing(anova, "sig_level", limit);
        }
        // Just above the limit the critical values are finite (tail 2^-53),
        // and a large effect still has full power
        let sig = MIN_SIG_LEVEL.next_up();
        let t = power_t_test(
            Some(1e6),
            Some(1.0),
            Some(1.0),
            Some(sig),
            None,
            TTestType::TwoSample,
            two,
        )
        .unwrap();
        assert!((t.power - 1.0).abs() < 1e-12, "t: {}", t.power);
        let prop = power_prop_test(Some(1e300), 0.3, Some(0.5), Some(sig), None, two).unwrap();
        assert!((prop.power - 1.0).abs() < 1e-12, "prop: {}", prop.power);
        let anova = power_anova_test(3, Some(1e6), 0.25, 1.0, Some(sig), None).unwrap();
        assert_power_in_range(&anova);
    }

    #[test]
    fn test_power_t_test_solve_delta_scales_up_to_the_largest_sd() {
        // The solved delta is proportional to sd, up to the largest sd the
        // delta search accepts. The one-sample n = 2 case has its root above
        // 5 * sd, where low + high in the bisection used to overflow to inf.
        let sd_max = f64::MAX / 10.0;
        for (n, test_type) in [(30.0, TTestType::TwoSample), (2.0, TTestType::OneSample)] {
            let solve = |sd: f64| {
                power_t_test(
                    Some(n),
                    None,
                    Some(sd),
                    Some(0.05),
                    Some(0.99),
                    test_type,
                    PowerAlternative::TwoSided,
                )
                .unwrap()
                .delta
            };
            let unit = solve(1.0);
            assert!(
                unit < 10.0,
                "{test_type:?}: the root {unit} lies in the bracket"
            );
            let scaled = solve(sd_max) / sd_max;
            assert!(
                ((scaled - unit) / unit).abs() < 1e-6,
                "{test_type:?}: delta / sd is {scaled} at sd = {sd_max:e}, {unit} at sd = 1"
            );
        }
    }

    #[test]
    fn test_validate_power_prop_test_equal_proportions_at_the_smallest_f64() {
        // With p1 == p2 the power is the significance level (R:
        // power.prop.test(n = 100, p1 = 0.3, p2 = 0.3, strict = TRUE)$power is
        // 0.05). At p = 5e-324 the standard error underflows to 0, and the
        // non-centrality 0 / 0 used to fall back to a power of 1.
        for p in [0.3, 5e-324] {
            let r = power_prop_test(
                Some(100.0),
                p,
                Some(p),
                Some(0.05),
                None,
                PowerAlternative::TwoSided,
            )
            .unwrap();
            assert!((r.power - 0.05).abs() < 1e-9, "p = {p:e}: {}", r.power);
        }
        // Distinct proportions there: the standard error sqrt(2 p q / n)
        // underflows to 0, but the non-centrality |p1 - p2| / sqrt(2 p q / n)
        // is about 1e-161, so the power is the significance level. Dividing by
        // the underflowed standard error used to give an infinite
        // non-centrality and a power of 1. R: power.prop.test(n = 100,
        // p1 = 5e-324, p2 = 1e-323, alternative = "one.sided")$power is
        // 0.049999999999999989.
        let r = power_prop_test(
            Some(100.0),
            5e-324,
            Some(1e-323),
            Some(0.05),
            None,
            PowerAlternative::OneSided,
        )
        .unwrap();
        assert!(
            (r.power - 0.049_999_999_999_999_99).abs() < 1e-9,
            "{}",
            r.power
        );
    }

    #[test]
    fn test_power_prop_test_rejects_non_finite_and_out_of_range_inputs() {
        let one_sided = PowerAlternative::OneSided;
        assert_refused(
            power_prop_test(Some(100.0), 0.3, Some(0.5), Some(1.5), None, one_sided),
            "sig_level",
        );
        assert_refused(
            power_prop_test(None, 0.3, Some(0.5), Some(0.05), Some(1.5), one_sided),
            "power",
        );
        assert_refused(
            power_prop_test(Some(f64::NAN), 0.3, Some(0.5), Some(0.05), None, one_sided),
            "n",
        );
        assert_refused(
            power_prop_test(Some(0.0), 0.3, Some(0.5), Some(0.05), None, one_sided),
            "n",
        );
        assert_refused(
            power_prop_test(
                Some(100.0),
                f64::NAN,
                Some(0.5),
                Some(0.05),
                None,
                one_sided,
            ),
            "p1",
        );
    }

    #[test]
    fn test_power_anova_rejects_non_finite_and_out_of_range_inputs() {
        let a = |n: Option<f64>, b: f64, w: f64, sig: f64, power: Option<f64>| {
            power_anova_test(3, n, b, w, Some(sig), power)
        };
        assert_refused(a(Some(f64::INFINITY), 0.25, 1.0, 0.05, None), "n");
        assert_refused(a(Some(f64::NAN), 0.25, 1.0, 0.05, None), "n");
        assert_refused(a(Some(1.0), 0.25, 1.0, 0.05, None), "n");
        // the limit is f64::MAX / groups, 3.898125604559999e289, for 2^62 groups
        assert_refused_containing(
            power_anova_test(1usize << 62, Some(1e300), 0.25, 1.0, Some(0.05), None),
            "n",
            "at most 3.898125604559999e289 for groups = 4611686018427387904 \
             (groups * (n - 1)",
        );
        assert_refused(a(Some(20.0), f64::NAN, 1.0, 0.05, None), "between_var");
        assert_refused(a(Some(20.0), -0.25, 1.0, 0.05, None), "between_var");
        assert_refused(a(Some(20.0), 0.25, 0.0, 0.05, None), "within_var");
        assert_refused(a(Some(20.0), 0.25, f64::INFINITY, 0.05, None), "within_var");
        assert_refused(a(Some(20.0), 0.25, 1.0, 1.5, None), "sig_level");
        assert_refused(a(None, 0.25, 1.0, 0.05, Some(1.0)), "power");
        assert_refused_containing(
            power_anova_test(1, Some(20.0), 0.25, 1.0, Some(0.05), None),
            "groups",
            "at least 2, got 1",
        );
        assert_refused_containing(
            a(Some(20.0), -1e300, 1.0, 0.05, None),
            "between_var",
            "got -1e300",
        );
    }

    #[test]
    fn test_power_anova_n_limit_is_the_largest_accepted_n() {
        // The refusal names the largest n the error-degrees-of-freedom check
        // accepts. For groups that are not a power of two, f64::MAX / groups
        // itself fails the check, since groups * (f64::MAX / groups - 1) rounds
        // up to infinity.
        let groups_list = (2..=40).chain([97, 1000, 1001, 199_999, 1usize << 62, usize::MAX]);
        for groups in groups_list {
            let anova = |n: f64| power_anova_test(groups, Some(n), 0.25, 1.0, Some(0.05), None);
            let message = match anova(f64::MAX) {
                Err(EconError::InvalidSpecification { message }) => message,
                other => panic!("groups = {groups}: expected a refusal, got {other:?}"),
            };
            let limit: f64 = message
                .strip_prefix("n must be at most ")
                .and_then(|rest| rest.split(' ').next())
                .and_then(|number| number.parse().ok())
                .unwrap_or_else(|| panic!("groups = {groups}: no limit in {message:?}"));
            assert!(anova(limit).is_ok(), "groups = {groups}: {limit:e} refused");
            assert_refused(anova(limit.next_up()), "n");
        }
    }

    #[test]
    fn test_validate_power_t_test_zero_effect_at_huge_n() {
        // With delta = 0 the power is the significance level
        // (R: power.t.test(n = 1e7, delta = 0, strict = TRUE)$power == 0.05).
        for n in [1e7, 1e15, 1e300] {
            let r = two_sample(n, 0.0, PowerAlternative::TwoSided);
            assert!((r.power - 0.05).abs() < 1e-9, "n = {n}: {}", r.power);
        }
        // R: power.t.test(n = 1e300, delta = 0, sd = 1e-300, strict = TRUE)$power
        // == 0.05. The standard error, sd * sqrt(2 / n), would underflow to 0
        // here, and 0 / 0 must not turn into a non-centrality of NaN.
        let r = power_t_test(
            Some(1e300),
            Some(0.0),
            Some(1e-300),
            Some(0.05),
            None,
            TTestType::TwoSample,
            PowerAlternative::TwoSided,
        )
        .unwrap();
        assert!((r.power - 0.05).abs() < 1e-9, "sd = 1e-300: {}", r.power);
    }

    #[test]
    fn test_t_cdf_at_infinite_x() {
        // An infinite x (the critical value of a sig_level at or below 2^-53,
        // which the power functions now refuse) used to give NaN in the
        // large-df expansion.
        for df in [5.0, 1e7] {
            assert_eq!(t_cdf(f64::INFINITY, df).unwrap(), 1.0, "df = {df}");
            assert_eq!(t_cdf(f64::NEG_INFINITY, df).unwrap(), 0.0, "df = {df}");
        }
    }

    // =========================================================================
    // The bounded quantiles against R
    // =========================================================================

    fn assert_rel(ours: f64, r: f64, tol: f64, what: &str) {
        assert!(
            ((ours - r) / r).abs() < tol,
            "{what}: ours {ours:.17}, R {r:.17}, relative error {:.2e}",
            ((ours - r) / r).abs()
        );
    }

    #[test]
    fn test_validate_t_quantile_large_df() {
        // R: qt(p, df)
        let cases = [
            (1001.0, 0.975, 1.962_336_705_280_879_1),
            (1001.0, 0.999, 3.098_393_982_491_397_3),
            (2000.0, 0.975, 1.961_150_826_099_437_5),
            (3e4, 0.95, 1.644_904_420_834_430_6),
            (1e5, 0.975, 1.959_987_707_534_608_8),
            (2e6, 0.975, 1.959_965_170_676_374),
            (2e6, 0.999, 3.090_236_381_237_242_6),
            (2e6, 0.999_999_999_999, 7.034_531_301_421_679),
            (2e7, 0.95, 1.644_853_703_139_932_4),
            (1e9, 0.999, 3.090_232_314_317_942),
            (1e15, 0.975, 1.959_963_984_540_056),
        ];
        for (df, p, r) in cases {
            assert_rel(
                t_quantile(df, p).unwrap(),
                r,
                1e-12,
                &format!("qt({p}, {df})"),
            );
        }
        // Far tail just above the switch, at the smallest tail the power
        // functions ask for (1 - 2^-53 is exact): R qt(2^-53, 1001, lower.tail = FALSE)
        let p = 1.0 - f64::EPSILON / 2.0;
        assert_rel(
            t_quantile(1001.0, p).unwrap(),
            8.351_823_973_969_251,
            2e-10,
            "df 1001",
        );
        // statrs is still used, unchanged, up to 1e3 degrees of freedom
        for df in [1.0, 5.0, 57.0, 1000.0] {
            let statrs = StudentsT::new(0.0, 1.0, df).unwrap().inverse_cdf(0.975);
            assert_eq!(t_quantile(df, 0.975).unwrap(), statrs);
        }
        assert_eq!(t_quantile(1e7, 1.0).unwrap(), f64::INFINITY);
        assert!(t_quantile(1e7, 1.5).is_err() && t_quantile(f64::NAN, 0.5).is_err());
    }

    #[test]
    fn test_validate_statrs_quantiles_in_their_error_band() {
        // statrs' quantiles are used unchanged where they are, and in a band
        // of t tails between about 0.12 and 0.21 they are off by up to 4.7e-9
        // (9.4e-9 for F at df1 = 1): within the bounds the module comment
        // states, not the 1e-12 they reach elsewhere.
        // R: qt(0.8, 1000) polished by Newton steps on pt (statrs: 2.8e-9 off)
        let t = t_quantile(1000.0, 0.8).unwrap();
        assert_eq!(
            t,
            StudentsT::new(0.0, 1.0, 1000.0).unwrap().inverse_cdf(0.8)
        );
        assert_rel(t, 0.841_980_822_162_429_1, 5e-9, "qt(0.8, 1000)");
        // F(1, 1e3) at p is t(1e3) at (1 + p) / 2, squared.
        // R: qf(0.6, 1, 1000) polished by Newton steps on pf (statrs: 5.5e-9 off)
        let f = f_quantile(1.0, 1000.0, 0.6).unwrap();
        assert_eq!(
            f,
            FisherSnedecor::new(1.0, 1000.0).unwrap().inverse_cdf(0.6)
        );
        assert_rel(f, 0.708_931_704_889_319_6, 1e-8, "qf(0.6, 1, 1000)");
    }

    #[test]
    fn test_validate_f_quantile_every_branch() {
        // R: the root of pf(x, df1, df2) = p (lower or upper tail), tol 1e-15
        let cases = [
            // both <= 1e3: statrs (upper tail, df1 < df2), else bisection
            (2.0, 57.0, 0.95, 3.158_842_719_260_644_7, 1e-12),
            (1e3, 1e3, 0.95, 1.109_688_290_242_986_3, 1e-12),
            // statrs is 1.1e-2 off at (1e3, 1), 5.2e-2 at (1, 1e3) (where R's
            // own qf is 1.1e-2 off), 8.9e-5 at (2, 1)
            (1e3, 1.0, 1.0 - 1e-6, 636_301_542_061.701_9, 1e-12),
            (1.0, 1e3, 1e-6, 1.571_581_921_210_43e-12, 1e-12),
            (2.0, 1.0, 1.0 - 1e-6, 499_999_999_970.743_8, 1e-12),
            // one <= 1e4, the other <= 1e6: bisection (statrs is 2.5e-8 off at (2, 1e4))
            (2.0, 1e4, 0.95, 2.996_629_893_999_222_6, 1e-11),
            (10.0, 1e4, 0.05, 0.393_949_963_490_444_87, 1e-11),
            (1e4, 1e4, 0.95, 1.033_445_780_480_156_5, 1e-11),
            (1.0, 1e5, 0.95, 3.841_551_813_686_773_5, 1e-9),
            (2.0, 3e5, 0.95, 2.995_762_188_459_321_5, 1e-9),
            (94.0, 1e6, 0.95, 1.251_416_581_585_954_4, 1e-9),
            (1.0, 1e6, 0.05, 0.003_932_141_973_820_885, 1e-9),
            (1e4, 1e6, 0.999, 1.044_504_725_077_492, 1e-9),
            // one <= 1e4, the other > 1e6: expansion
            (2.0, 3e7, 0.95, 2.995_732_572_701_071_3, 1e-9),
            (1.0, 1e15, 0.95, 3.841_458_820_694_136, 1e-9),
            (94.0, 1e9, 0.05, 0.772_763_479_603_439_6, 1e-8),
            (1e3, 1e7, 0.999, 1.143_925_438_306_772_5, 1e-7),
            (1e8, 10.0, 0.95, 2.537_878_434_487_179_6, 1e-8),
            (2e6, 3.0, 0.05, 0.383_889_871_291_538_4, 1e-8),
            // as 1 / F(10, 1e8) at upper tail 1e-15, which 1 - (1 - 1e-15)
            // used to round by 0.08% (a 1.9e-5 error in the quantile)
            (1e8, 10.0, 1e-15, 0.106_759_098_138_915_76, 1e-8),
            // both > 1e4: Paulson
            (3e4, 1e6, 0.95, 1.013_672_233_527_664, 1e-8),
            (1e5, 1e7, 0.999, 1.013_947_232_510_641_8, 1e-6),
            (9e5, 2e6, 0.95, 1.002_956_005_739_645_2, 1e-8),
            (1e6, 1e9, 0.5, 0.999_999_334_000_078_6, 1e-12),
            // the far upper tail just above the switch, tail 2^-53
            (
                1.1e4,
                1.1e4,
                1.0 - f64::EPSILON / 2.0,
                1.169_566_557_883_788_6,
                1e-5,
            ),
        ];
        for (df1, df2, p, r, tol) in cases {
            let what = format!("qf({p}, {df1}, {df2})");
            assert_rel(f_quantile(df1, df2, p).unwrap(), r, tol, &what);
        }
        // statrs is still used, unchanged, for the upper tail with
        // df1 < df2 <= 1e3, the shape power_anova_test asks for
        for (df1, df2) in [(1.0, 2.0), (2.0, 57.0), (3.0, 76.0), (999.0, 1e3)] {
            let statrs = FisherSnedecor::new(df1, df2).unwrap().inverse_cdf(0.95);
            assert_eq!(f_quantile(df1, df2, 0.95).unwrap(), statrs);
        }
        assert_eq!(f_quantile(3.0, 3e7, 1.0).unwrap(), f64::INFINITY);
        assert_eq!(f_quantile(3.0, 3e7, 0.0).unwrap(), 0.0);
        assert!(f_quantile(3.0, 3e7, f64::NAN).is_err());
        assert!(f_quantile(3.0, f64::INFINITY, 0.5).is_err());
    }

    #[test]
    fn test_validate_quantiles_upper_tail_below_statrs_floor() {
        // p = 1 - 2e-16 rounds to 1 - 2^-52. statrs folds that onto a tail
        // probability of 2.2e-16, below STATRS_MIN_P, so these go to the
        // bisection, as the same tail does at the bottom.
        let p = 1.0 - 2e-16;
        // R: qt(2^-52, df, lower.tail = FALSE)
        let t_cases = [
            (1.0, 1_433_540_284_805_664.8),
            (3.0, 170_608.307_621_525_75),
            (1000.0, 8.264_015_607_871_633),
        ];
        for (df, r) in t_cases {
            let upper = t_quantile(df, p).unwrap();
            assert_rel(upper, r, 1e-12, &format!("qt(1 - 2^-52, {df})"));
            // 1 - p is exact, so both tails see the same tail probability
            assert_eq!(upper, -t_quantile(df, 1.0 - p).unwrap(), "df = {df}");
        }
        // R: qf(2^-52, df1, df2, lower.tail = FALSE)
        let f_cases = [
            (2.0, 57.0, 72.446_803_480_605_54),
            (4.0, 245.0, 23.370_948_266_418_868),
            (1e3, 1e3, 1.676_802_747_107_059_7),
        ];
        for (df1, df2, r) in f_cases {
            let what = format!("qf(1 - 2^-52, {df1}, {df2})");
            assert_rel(f_quantile(df1, df2, p).unwrap(), r, 1e-12, &what);
        }
    }

    #[test]
    fn test_validate_quantiles_far_tail_at_small_df() {
        // Tails below STATRS_MIN_P = 1e-5, where statrs is off (its t quantile
        // at df = 1 never passes 9.2e7 in magnitude, and it is 3.9e-11 off at
        // df = 1.05 and tail 1e-6), go to the bisection.
        // R: qt(tail, df, lower.tail = FALSE), taken at the exact tail
        let t_cases = [
            (1.0, 1e-10, 3_183_098_861.837_906_4),
            (1.5, 1e-12, 52_194_694.273_446_366),
            (1.75, 1e-14, 61_668_983.346_367_51),
            (1.05, 1e-6, 175_852.477_391_897_6),
        ];
        for (df, tail, r) in t_cases {
            let what = format!("qt({tail:e}, {df}, lower.tail = FALSE)");
            assert_rel(-t_quantile(df, tail).unwrap(), r, 1e-12, &what);
        }
        // R: qt(1 - tail, df). 1 - tail rounds (1 - (1 - 1e-10) is
        // 1.000000082740371e-10), and R and t_quantile then both work with
        // that rounded tail, so these differ from the values above.
        let t_upper = [
            (1.0, 1.0 - 1e-10, 3_183_098_598.467_148),
            (1.5, 1.0 - 1e-12, 52_195_464.045_249_306),
        ];
        for (df, p, r) in t_upper {
            assert_rel(
                t_quantile(df, p).unwrap(),
                r,
                1e-12,
                &format!("qt({p}, {df})"),
            );
        }
        // R: qf(1 - tail, df1, df2); for (1, 2) also the closed form
        // 2 (1 - q)^2 / (q (2 - q)) at q = 1 - p
        let f_upper = [
            (1.0, 2.0, 1.0 - 1e-12, 1_000_022_122_208.002_9),
            (3.0, 4.0, 1.0 - 1e-13, 5_772_603.724_181_895),
        ];
        for (df1, df2, p, r) in f_upper {
            let what = format!("qf({p}, {df1}, {df2})");
            assert_rel(f_quantile(df1, df2, p).unwrap(), r, 1e-12, &what);
        }
        // At the exact tails: F(df1, df2) at p is 1 / F(df2, df1) at 1 - p,
        // R: qf(tail, df2, df1, lower.tail = FALSE)
        let f_lower = [
            (2.0, 1.0, 1e-12, 999_999_999_998.5),
            (4.0, 3.0, 1e-13, 5_773_501.136_340_68),
        ];
        for (df1, df2, tail, r) in f_lower {
            let what = format!("1 / qf({tail:e}, {df2}, {df1}, lower.tail = FALSE)");
            assert_rel(1.0 / f_quantile(df1, df2, tail).unwrap(), r, 1e-12, &what);
        }
    }

    #[test]
    fn test_quantiles_are_monotone_and_finite_at_any_df() {
        // Every branch, including astronomically large degrees of freedom,
        // returns an ordered, non-NaN quantile for every probability.
        let dfs = [1.0, 3.0, 57.0, 1e3, 1e4, 3e4, 1e6, 1e7, 1e12, 4.6e18, 1e300];
        let ps = [
            1e-300,
            1e-100,
            1e-9,
            0.05,
            0.5,
            0.95,
            0.999,
            1.0 - 1e-12,
            1.0 - 2e-16,
        ];
        for &df in &dfs {
            // The t sweep leaves out p = 1e-300: there the bisection saturates
            // at about -1.3e154 for df < 2 (see t_quantile and the check
            // below), which is ordered but not the quantile.
            let t: Vec<f64> = ps[1..]
                .iter()
                .map(|&p| t_quantile(df, p).unwrap())
                .collect();
            assert!(t.iter().all(|v| !v.is_nan()), "t df={df}: {t:?}");
            assert!(t.windows(2).all(|w| w[0] <= w[1]), "t df={df}: {t:?}");
            for &df2 in &dfs {
                let f: Vec<f64> = ps
                    .iter()
                    .map(|&p| f_quantile(df, df2, p).unwrap())
                    .collect();
                assert!(
                    f.iter().all(|v| !v.is_nan() && *v >= 0.0),
                    "F({df}, {df2}): {f:?}"
                );
                assert!(f.windows(2).all(|w| w[0] <= w[1]), "F({df}, {df2}): {f:?}");
            }
        }
        // What t_quantile gives at p = 1e-300 and df = 1 (R: -3.2e299): the
        // magnitude saturates where m^2 overflows, as its doc says
        let saturated = t_quantile(1.0, 1e-300).unwrap();
        assert_rel(saturated, -f64::MAX.sqrt(), 1e-12, "t_quantile(1, 1e-300)");
    }
}
