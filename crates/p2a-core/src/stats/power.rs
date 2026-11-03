//! Power analysis for statistical tests: R's `power.t.test`, `power.prop.test` and
//! `power.anova.test`, solved for any one parameter.
//!
//! The power of the t test and of the one-way ANOVA comes from the exact noncentral t and F
//! distributions, written here from the published algorithms: Lenth's (1989) AS 243 series for the
//! noncentral t, Lenth's (1987) AS 226 series with Frick's (1990) AS R84 starting point for the
//! noncentral beta (and so the noncentral F), with the constants R uses (error bounds 1e-12 and 1e-9,
//! iteration caps 1000 and 10000). Outside the range of the series the documented limits apply: the
//! normal approximation of Abramowitz & Stegun 26.7.10 for the t (df above 4e5, or a noncentrality
//! whose `exp(-lambda / 2)` underflows) and the noncentral chi-square for the F (df2 above 1e8). A
//! series that stops at its cap, or a value whose precision R would warn about, is reported as a
//! result warning at the returned value (R prints such warnings for intermediate values of its root
//! search as well; p2a reports only the answer's).
//!
//! The defaults and conventions are R's: `sd = 1` and `sig_level = 0.05` unless they are the unknown;
//! the one-tail power (`strict = false`) unless `strict` asks for both tails of a two-sided test; the
//! proportion test's variance `p1 q1 + p2 q2`; the ANOVA noncentrality
//! `(groups - 1) n between_var / within_var`. The unknown parameter is found by Brent's zeroin over R's
//! search intervals, with R's rules for extending them (the table in `validation/stats/power.md`) and
//! R's default tolerance [`R_UNIROOT_TOL`]; solutions are fractional, as R reports them.
//!
//! Critical values come from the shared quantile and tail helpers of
//! [`crate::traits::estimator`]: `qt` is [`t_quantile`] at the exact tail `sig_level / tside`, `qnorm`
//! is [`normal_quantile`], and the F critical value is the root of [`f_test_p_value`] (in log space),
//! with R's chi-square limits for large degrees of freedom. No tail is formed as `1 - sig_level`.
//!
//! Every root search checks the cooperative-cancellation budget
//! ([`crate::resource::checkpoint`]) once per power evaluation, and every loop has a fixed bound.

use crate::errors::{EconError, EconResult};
use crate::options::Choices;
use crate::resource::checkpoint;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{
    chi_squared_p_value, chi_squared_quantile, f_test_p_value, normal_quantile, t_quantile,
};
use serde::{Deserialize, Serialize};
use std::fmt;

/// R's uniroot default tolerance, `.Machine$double.eps^0.25 = 2^-13`.
pub const R_UNIROOT_TOL: f64 = 1.220_703_125e-4;

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

/// The significance level must be greater than this, 2^-53 (about 1.1e-16), in every test: at or
/// below it the confidence level 1 - sig_level rounds to 1 (or to 1 - 2^-53) in double precision, so
/// the level cannot be told from its complement.
const MIN_SIG_LEVEL: f64 = f64::EPSILON / 2.0;

/// `Err` unless `sig_level` lies strictly between [`MIN_SIG_LEVEL`] and 1 (NaN fails both
/// comparisons). The message gives the floor's reason only for a level from 0 up to the floor, the
/// only values it describes; a negative level, NaN, or a level of 1 or more gets the limits alone.
fn require_sig_level(sig_level: f64) -> EconResult<()> {
    if sig_level > MIN_SIG_LEVEL && sig_level < 1.0 {
        return Ok(());
    }
    let mut message = format!(
        "sig_level must be between {MIN_SIG_LEVEL:?} (2^-53) and 1 (exclusive), \
         got {sig_level:?}"
    );
    if (0.0..=MIN_SIG_LEVEL).contains(&sig_level) {
        message.push_str(
            ": every test uses this floor, since at or below 2^-53 the confidence level \
             1 - sig_level rounds to 1 or to 1 - 2^-53 in double precision",
        );
    }
    Err(EconError::InvalidSpecification { message })
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

fn require_tol(tol: f64) -> EconResult<()> {
    if tol.is_finite() && tol > 0.0 {
        Ok(())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("tol must be a finite number greater than 0, got {tol:?}"),
        })
    }
}

/// A whole number below 2^64 with all its digits (4611686018427387904 rather than f64's shortest
/// form 4611686018427388000), anything else as f64 prints it.
fn count_text(v: f64) -> String {
    if v.fract() == 0.0 && (0.0..18_446_744_073_709_551_616.0).contains(&v) {
        format!("{}", v as u64)
    } else {
        format!("{v}")
    }
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

// ============================================================================
// Noncentral t and F
// ============================================================================

/// A distribution function value and whether it may have lost precision (the series stopped at its
/// iteration cap, or the value sits where R warns that full precision may not have been achieved).
#[derive(Debug, Clone, Copy, PartialEq)]
struct NcValue {
    value: f64,
    imprecise: bool,
}

impl NcValue {
    fn exact(value: f64) -> Self {
        NcValue {
            value,
            imprecise: false,
        }
    }
    fn nan() -> Self {
        Self::exact(f64::NAN)
    }
}

/// sqrt(2 / pi)
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;
/// ln(sqrt(pi))
const LN_SQRT_PI: f64 = 0.572_364_942_924_700_1;

/// ln Gamma(x) for x > 0 (R's lgammafn, the same function R's series use).
fn ln_gamma(x: f64) -> f64 {
    // SAFETY: lgammafn is a pure C function of one double with no global state; for the positive,
    // moderate arguments passed here it neither fails nor warns.
    unsafe { r_mathlib::ffi::lgammafn(x) }
}

/// Noncentral t distribution function P(T <= t) (or the upper tail), T ~ t(df, ncp).
///
/// Lenth (1989), Algorithm AS 243: with x = t^2 / (t^2 + df) the distribution is a Poisson mixture of
/// incomplete beta functions in x, summed by the recurrences of the paper, plus Phi(-ncp). The error
/// bound and iteration cap are R's (1e-12, 1000). A negative t is reflected (t -> -t, ncp -> -ncp).
/// Large df or a noncentrality whose Poisson weights underflow use the normal approximation of
/// Abramowitz & Stegun 26.7.10.
fn pnt(t: f64, df: f64, ncp: f64, lower_tail: bool) -> NcValue {
    const ITER_MAX: usize = 1000;
    const ERR_MAX: f64 = 1e-12;
    let (tail0, tail1) = if lower_tail { (0.0, 1.0) } else { (1.0, 0.0) };
    if t.is_nan() || df.is_nan() || ncp.is_nan() || df <= 0.0 {
        return NcValue::nan();
    }
    if ncp == 0.0 {
        return NcValue::exact(r_mathlib::students_t_cdf(t, df, lower_tail, false));
    }
    if df.is_infinite() {
        return NcValue::exact(r_mathlib::normal_cdf(t, ncp, 1.0, lower_tail, false));
    }
    if t.is_infinite() {
        return NcValue::exact(if t < 0.0 { tail0 } else { tail1 });
    }
    // reflect a negative t: P(T <= t | ncp) = 1 - P(T <= -t | -ncp)
    let (reflected, tt, del) = if t >= 0.0 {
        (false, t, ncp)
    } else {
        if ncp > 40.0 {
            // the mass below a negative t is below 1e-350
            return NcValue::exact(tail0);
        }
        (true, -t, -ncp)
    };
    let lower = lower_tail != reflected;
    // exp(-del^2 / 2) underflows past 2 ln 2 * 1021 (the smallest normal exponent)
    if df > 4e5 || del * del > 2.0 * std::f64::consts::LN_2 * 1021.0 {
        let s = 1.0 / (4.0 * df);
        let mean_scaled = tt * (1.0 - s);
        let spread = (1.0 + tt * tt * 2.0 * s).sqrt();
        return NcValue::exact(r_mathlib::normal_cdf(
            mean_scaled,
            del,
            spread,
            lower,
            false,
        ));
    }
    let mut imprecise = false;
    let t2 = tt * tt;
    let x = t2 / (t2 + df);
    let mut sum = 0.0;
    if x > 0.0 {
        let lambda = del * del;
        // Poisson weights of the odd (p) and even (q) terms
        let mut p = 0.5 * (-0.5 * lambda).exp();
        if p == 0.0 {
            return NcValue {
                value: tail0,
                imprecise: true,
            };
        }
        let mut q = SQRT_2_OVER_PI * p * del;
        // s: the Poisson mass not yet summed (for the error bound)
        let mut s = 0.5 - p;
        if s < 1e-7 {
            s = -0.5 * (-0.5 * lambda).exp_m1();
        }
        let mut a = 0.5;
        let b = 0.5 * df;
        // (1 - x)^b, formed from df / (t^2 + df) to keep its precision
        let rxb = (df / (t2 + df)).powf(b);
        let ln_beta_half = LN_SQRT_PI + ln_gamma(b) - ln_gamma(0.5 + b);
        let mut x_odd = r_mathlib::beta_cdf(x, a, b, true, false);
        let mut g_odd = 2.0 * rxb * (a * x.ln() - ln_beta_half).exp();
        let bx = b * x;
        let mut x_even = if bx < f64::EPSILON { bx } else { 1.0 - rxb };
        let mut g_even = bx * rxb;
        sum = p * x_odd + q * x_even;
        let mut finished = false;
        for k in 1..=ITER_MAX {
            a += 1.0;
            x_odd -= g_odd;
            x_even -= g_even;
            g_odd *= x * (a + b - 1.0) / a;
            g_even *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2 * k) as f64;
            q *= lambda / (2 * k + 1) as f64;
            sum += p * x_odd + q * x_even;
            s -= p;
            if s < -1e-10 {
                // the weights summed past 1 by more than rounding
                imprecise = true;
                finished = true;
                break;
            }
            if s <= 0.0 && k > 1 {
                finished = true;
                break;
            }
            let err_bound = 2.0 * s * (x_odd - g_odd);
            if err_bound.abs() < ERR_MAX {
                finished = true;
                break;
            }
        }
        if !finished {
            imprecise = true;
        }
    }
    sum += r_mathlib::normal_cdf(-del, 0.0, 1.0, true, false);
    if lower && sum > 1.0 - 1e-10 {
        imprecise = true;
    }
    let sum = sum.min(1.0);
    NcValue {
        value: if lower { sum } else { 0.5 - sum + 0.5 },
        imprecise,
    }
}

/// Noncentral beta distribution function (lower tail) at x, with o_x = 1 - x given separately.
///
/// Lenth (1987), Algorithm AS 226, started at Frick's (1990, AS R84) index
/// x0 = floor(max(0, c - 7 sqrt(c))), c = ncp / 2: the Poisson(c) mixture of incomplete beta
/// functions I_x(a + j, b), j >= x0, summed by the recurrence of the paper until the error bound
/// (temp - gx) * (1 - sum of weights) is below 1e-9 or 10000 terms have been added (R's constants).
fn pnbeta_lower(x: f64, o_x: f64, a: f64, b: f64, ncp: f64) -> NcValue {
    const ERR_MAX: f64 = 1.0e-9;
    const ITER_MAX: f64 = 10000.0;
    if ncp < 0.0 || a <= 0.0 || b <= 0.0 {
        return NcValue::nan();
    }
    if x < 0.0 || o_x > 1.0 || (x == 0.0 && o_x == 1.0) {
        return NcValue::exact(0.0);
    }
    if x > 1.0 || o_x < 0.0 || (x == 1.0 && o_x == 0.0) {
        return NcValue::exact(1.0);
    }
    let c = ncp / 2.0;
    let x0 = (c - 7.0 * c.sqrt()).max(0.0).floor();
    let a0 = a + x0;
    let ln_beta = ln_gamma(a0) + ln_gamma(b) - ln_gamma(a0 + b);
    let mut temp = r_mathlib::beta_cdf(x, a0, b, true, false);
    let ln_1mx = if x < 0.5 { (-x).ln_1p() } else { o_x.ln() };
    let mut gx = (a0 * x.ln() + b * ln_1mx - ln_beta - a0.ln()).exp();
    let mut q = if a0 > a {
        (-c + x0 * c.ln() - ln_gamma(x0 + 1.0)).exp()
    } else {
        (-c).exp()
    };
    let mut sum_q = 1.0 - q;
    let mut sum = q * temp;
    let mut j = x0;
    let mut err_bound;
    loop {
        j += 1.0;
        temp -= gx;
        gx *= x * (a + b + j - 1.0) / (a + j);
        q *= c / j;
        sum_q -= q;
        sum += temp * q;
        err_bound = (temp - gx) * sum_q;
        if !(err_bound > ERR_MAX && j < ITER_MAX + x0) {
            break;
        }
    }
    NcValue {
        value: sum,
        imprecise: err_bound > ERR_MAX || j >= ITER_MAX + x0,
    }
}

/// Noncentral chi-square distribution function P(X <= x) (or the upper tail), X ~ chi2(df, ncp):
/// the Poisson(ncp / 2) mixture of central chi-squares with df + 2j degrees of freedom, summed from
/// Frick's starting index as [`pnbeta_lower`] (with the regularized gamma function in place of the
/// beta). Beyond 1e5 for ncp / 2 or 1e10 for df, where the series would need more terms than its
/// cap, the cube-root normal approximation of Abramowitz & Stegun 26.4.32 is used.
fn pnchisq(x: f64, df: f64, ncp: f64, lower_tail: bool) -> NcValue {
    const ERR_MAX: f64 = 1.0e-12;
    const ITER_MAX: f64 = 10000.0;
    if x.is_nan() || df.is_nan() || ncp.is_nan() || df <= 0.0 || ncp < 0.0 {
        return NcValue::nan();
    }
    if x <= 0.0 {
        return NcValue::exact(if lower_tail { 0.0 } else { 1.0 });
    }
    if x.is_infinite() {
        return NcValue::exact(if lower_tail { 1.0 } else { 0.0 });
    }
    let c = ncp / 2.0;
    if c > 1e5 || df > 1e10 {
        // h^2 = 2 (df + 2 ncp) / (9 k^2), k = df + ncp, formed without overflow
        let k = df + ncp;
        let h2 = 2.0 / 9.0 * (df / k + 2.0 * (ncp / k)) / k / k;
        let z = ((x / k).cbrt() - (1.0 - h2)) / h2.sqrt();
        return NcValue::exact(r_mathlib::normal_cdf(z, 0.0, 1.0, lower_tail, false));
    }
    let y = x / 2.0;
    let x0 = (c - 7.0 * c.sqrt()).max(0.0).floor();
    let mut shape = df / 2.0 + x0;
    // P(shape, y) and the term y^shape e^-y / Gamma(shape + 1) of P(shape, y) - P(shape + 1, y)
    let mut temp = r_mathlib::gamma_cdf(y, shape, 1.0, true, false);
    let mut gx = (shape * y.ln() - y - ln_gamma(shape + 1.0)).exp();
    let mut q = if x0 > 0.0 {
        (-c + x0 * c.ln() - ln_gamma(x0 + 1.0)).exp()
    } else {
        (-c).exp()
    };
    let mut sum_q = 1.0 - q;
    let mut sum = q * temp;
    let mut j = x0;
    let mut err_bound;
    loop {
        j += 1.0;
        temp -= gx;
        shape += 1.0;
        gx *= y / shape;
        q *= c / j;
        sum_q -= q;
        sum += temp * q;
        err_bound = (temp - gx) * sum_q;
        if !(err_bound > ERR_MAX && j < ITER_MAX + x0) {
            break;
        }
    }
    let imprecise = err_bound > ERR_MAX || j >= ITER_MAX + x0;
    let sum = sum.clamp(0.0, 1.0);
    NcValue {
        value: if lower_tail { sum } else { 0.5 - sum + 0.5 },
        imprecise,
    }
}

/// Noncentral F distribution function P(F <= x) (or the upper tail), F ~ F(df1, df2, ncp), through
/// the noncentral beta at y / (1 + y), y = df1 x / df2; the noncentral chi-square limit for
/// df2 > 1e8 (R's switch).
fn pnf(x: f64, df1: f64, df2: f64, ncp: f64, lower_tail: bool) -> NcValue {
    if x.is_nan() || df1.is_nan() || df2.is_nan() || ncp.is_nan() {
        return NcValue::nan();
    }
    if df1 <= 0.0 || df2 <= 0.0 || ncp < 0.0 || !ncp.is_finite() {
        return NcValue::nan();
    }
    if !df1.is_finite() && !df2.is_finite() {
        return NcValue::nan();
    }
    if x <= 0.0 {
        return NcValue::exact(if lower_tail { 0.0 } else { 1.0 });
    }
    if x == f64::INFINITY {
        return NcValue::exact(if lower_tail { 1.0 } else { 0.0 });
    }
    if df2 > 1e8 {
        return pnchisq(x * df1, df1, ncp, lower_tail);
    }
    let y = (df1 / df2) * x;
    let r = pnbeta_lower(y / (1.0 + y), 1.0 / (1.0 + y), df1 / 2.0, df2 / 2.0, ncp);
    if lower_tail {
        r
    } else {
        // an upper tail formed as 1 - (lower tail) keeps only about 1e-10 below 1e-10
        NcValue {
            value: 1.0 - r.value.min(1.0),
            imprecise: r.imprecise || r.value > 1.0 - 1e-10,
        }
    }
}

// ============================================================================
// Critical values
// ============================================================================

/// Root of a decreasing function of q >= 0 that is positive at 0: the bracket doubles from 1.
fn decreasing_root(what: &str, mut g: impl FnMut(f64) -> f64, g_at_zero: f64) -> EconResult<f64> {
    let mut hi = 1.0;
    let mut g_hi = g(hi);
    // hi stays a power of two below f64::MAX, then f64::MAX itself (2^1023 * 2 overflows)
    while g_hi > 0.0 && hi < f64::MAX {
        hi = if hi >= 2f64.powi(1023) {
            f64::MAX
        } else {
            hi * 2.0
        };
        g_hi = g(hi);
    }
    if g_hi > 0.0 {
        // the quantile is beyond f64::MAX (R's qf gives Inf there too)
        return Ok(f64::INFINITY);
    }
    if g_hi.is_nan() {
        return Err(EconError::Computation(format!(
            "{what}: no bracket for the critical value"
        )));
    }
    let mut f = |q: f64| -> EconResult<f64> { Ok(g(q)) };
    let (root, _, _) = zeroin(&mut f, 0.0, hi, g_at_zero, g_hi, 0.0, 1000)?;
    Ok(root)
}

/// Upper chi-square quantile: the q with P(chi2_df > q) = alpha (R's qchisq(alpha, df,
/// lower.tail = FALSE)). The Wilson-Hilferty cube-root approximation beyond df = 1e10, where it is
/// exact to double precision near the centre and the tail helper's series is near its cap.
fn chi_squared_upper_quantile(alpha: f64, df: f64) -> EconResult<f64> {
    if df > 1e10 {
        let z = -normal_quantile(alpha)?;
        let h = 2.0 / (9.0 * df);
        return Ok(df * (1.0 - h + z * h.sqrt()).powi(3));
    }
    let ln_alpha = alpha.ln();
    decreasing_root(
        "chi-square critical value",
        |q| chi_squared_p_value(q, df).ln() - ln_alpha,
        -ln_alpha,
    )
}

/// Lower chi-square quantile (R's qchisq(p, df)), Wilson-Hilferty beyond df = 1e10.
fn chi_squared_lower_quantile(p: f64, df: f64) -> EconResult<f64> {
    if df > 1e10 {
        let z = normal_quantile(p)?;
        let h = 2.0 / (9.0 * df);
        return Ok(df * (1.0 - h + z * h.sqrt()).powi(3));
    }
    chi_squared_quantile(p, df)
}

/// R's `qf(alpha, df1, df2, lower.tail = FALSE)`: the root of `f_test_p_value(q, df1, df2) = alpha`
/// (in log space, by zeroin at tol 0), with R's chi-square limits when the larger degrees of freedom
/// exceed 4e5. NaN for alpha outside (0, 1), as R.
fn f_upper_quantile(alpha: f64, df1: f64, df2: f64) -> EconResult<f64> {
    if !(alpha > 0.0 && alpha < 1.0) || df1.is_nan() || df1 <= 0.0 || df2.is_nan() || df2 <= 0.0 {
        return Ok(f64::NAN);
    }
    if df1 <= df2 && df2 > 4e5 {
        if df1.is_infinite() {
            return Ok(1.0);
        }
        return Ok(chi_squared_upper_quantile(alpha, df1)? / df1);
    }
    if df1 > 4e5 {
        return Ok(df2 / chi_squared_lower_quantile(alpha, df2)?);
    }
    let ln_alpha = alpha.ln();
    decreasing_root(
        "F critical value",
        |q| f_test_p_value(q, df1, df2).ln() - ln_alpha,
        -ln_alpha,
    )
}

// ============================================================================
// Root finding: Brent's zeroin and R's interval rules
// ============================================================================

/// Brent's (1973) zeroin on [ax, bx] with f(ax) = fa, f(bx) = fb of opposite signs: the root to
/// within `tol + 4 eps |root|`, with the number of iterations (`None` after `max_iter`) and the
/// final bracket width.
fn zeroin(
    f: &mut impl FnMut(f64) -> EconResult<f64>,
    ax: f64,
    bx: f64,
    fa: f64,
    fb: f64,
    tol: f64,
    max_iter: usize,
) -> EconResult<(f64, Option<usize>, f64)> {
    let (mut a, mut b, mut fa, mut fb) = (ax, bx, fa, fb);
    let (mut c, mut fc) = (a, fa);
    if fa == 0.0 {
        return Ok((a, Some(0), 0.0));
    }
    if fb == 0.0 {
        return Ok((b, Some(0), 0.0));
    }
    for iter in 0..=max_iter {
        let prev_step = b - a;
        // keep b the best estimate, c on the other side of the root
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut step = (c - b) / 2.0;
        if step.abs() <= tol_act || fb == 0.0 {
            return Ok((b, Some(iter), (c - b).abs()));
        }
        if iter == max_iter {
            break;
        }
        // interpolate when the last step was large enough and |f| decreased
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q);
            if a == c {
                // secant
                let s = fb / fa;
                p = cb * s;
                q = 1.0 - s;
            } else {
                // inverse quadratic interpolation
                let qa = fa / fc;
                let r = fb / fc;
                let s = fb / fa;
                p = s * (cb * qa * (qa - r) - (b - a) * (r - 1.0));
                q = (qa - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if p < 0.75 * cb * q - (tol_act * q).abs() / 2.0 && p < (prev_step * q / 2.0).abs() {
                step = p / q;
            }
        }
        if step.abs() < tol_act {
            step = if step > 0.0 { tol_act } else { -tol_act };
        }
        a = b;
        fa = fb;
        b += step;
        fb = f(b)?;
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    Ok((b, None, (c - b).abs()))
}

/// How a search interval may be extended when the function does not change sign on it (R's
/// `uniroot(extendInt = )`): not at all, both ends, or in the direction where an increasing
/// (`UpX`) or decreasing (`DownX`) function crosses zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendInt {
    No,
    Yes,
    DownX,
    UpX,
}

/// What the root search reports about a solved parameter (p2a's own diagnostics).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RootInfo {
    /// Function evaluations of the zeroin iteration plus the bracket-extension steps.
    pub iterations: usize,
    /// Steps that moved an end of the search interval (0 when it was not extended).
    pub extension_steps: usize,
    /// Width of the final bracket.
    pub estimated_precision: f64,
    /// Lower end of the bracket after extension.
    pub lower: f64,
    /// Upper end of the bracket after extension.
    pub upper: f64,
    /// `false` when zeroin stopped at 1000 iterations; the value is then the last iterate.
    pub converged: bool,
}

const ROOT_MAX_ITER: usize = 1000;

fn sign(v: f64) -> f64 {
    if v.is_nan() {
        f64::NAN
    } else if v > 0.0 {
        1.0
    } else if v < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// The root of `f` on [lower, upper] by zeroin at `tol`, after extending the interval as R's
/// `uniroot` does for `extend`. Errors name `param`.
fn uniroot(
    f: &mut impl FnMut(f64) -> EconResult<f64>,
    lower: f64,
    upper: f64,
    extend: ExtendInt,
    tol: f64,
    param: &str,
) -> EconResult<(f64, RootInfo)> {
    if lower.is_nan() || upper.is_nan() || lower >= upper {
        return Err(invalid(format!(
            "cannot solve for {param}: the search interval [{lower:e}, {upper:e}] is empty"
        )));
    }
    let truncate = |v: f64| {
        if v.is_nan() {
            v
        } else {
            v.clamp(-f64::MAX, f64::MAX)
        }
    };
    let undefined = || {
        invalid(format!(
            "cannot solve for {param}: the power is not defined at the end of the search interval"
        ))
    };
    let (mut lower, mut upper) = (lower, upper);
    let mut f_lower = f(lower)?;
    let mut f_upper = f(upper)?;
    if f_lower.is_nan() || f_upper.is_nan() {
        return Err(undefined());
    }
    f_lower = truncate(f_lower);
    f_upper = truncate(f_upper);
    let direction = match extend {
        ExtendInt::Yes => None,
        ExtendInt::No => Some(0.0),
        ExtendInt::DownX => Some(-1.0),
        ExtendInt::UpX => Some(1.0),
    };
    let extend_now = match direction {
        None => f_lower * f_upper > 0.0,
        Some(s) => s * f_lower > 0.0 || s * f_upper < 0.0,
    };
    let mut steps = 0usize;
    let no_sign_change = |steps: usize| {
        invalid(format!(
            "cannot solve for {param}: no sign change found in {} iterations",
            steps - 1
        ))
    };
    if extend_now {
        let width = |u: f64| 0.01 * u.abs().max(1e-4);
        match direction {
            None => {
                let (mut d_lower, mut d_upper) = (width(lower), width(upper));
                while f_lower * f_upper > 0.0 && (lower.is_finite() || upper.is_finite()) {
                    steps += 1;
                    if steps > ROOT_MAX_ITER {
                        return Err(no_sign_change(steps));
                    }
                    if lower.is_finite() {
                        let (old, old_f) = (lower, f_lower);
                        lower -= d_lower;
                        f_lower = f(lower)?;
                        if f_lower.is_nan() {
                            lower = old;
                            f_lower = old_f;
                            d_lower /= 4.0;
                        }
                    }
                    if upper.is_finite() {
                        let (old, old_f) = (upper, f_upper);
                        upper += d_upper;
                        f_upper = f(upper)?;
                        if f_upper.is_nan() {
                            upper = old;
                            f_upper = old_f;
                            d_upper /= 4.0;
                        }
                    }
                    d_lower *= 2.0;
                    d_upper *= 2.0;
                }
            }
            Some(s) => {
                let mut d = width(lower);
                while s * f_lower > 0.0 {
                    steps += 1;
                    if steps > ROOT_MAX_ITER {
                        return Err(no_sign_change(steps));
                    }
                    lower -= d;
                    if !lower.is_finite() {
                        break;
                    }
                    f_lower = f(lower)?;
                    d *= 2.0;
                }
                let mut d = width(upper);
                while s * f_upper < 0.0 {
                    steps += 1;
                    if steps > ROOT_MAX_ITER {
                        return Err(no_sign_change(steps));
                    }
                    upper += d;
                    if !upper.is_finite() {
                        break;
                    }
                    f_upper = f(upper)?;
                    d *= 2.0;
                }
            }
        }
    }
    let sign_product = sign(f_lower) * sign(f_upper);
    if sign_product.is_nan() || sign_product > 0.0 || !lower.is_finite() || !upper.is_finite() {
        return Err(invalid(format!(
            "cannot solve for {param}: the power does not reach the requested value on \
             [{lower:e}, {upper:e}]"
        )));
    }
    if extend_now && steps > 0 {
        f_lower = truncate(f_lower);
        f_upper = truncate(f_upper);
    }
    let (root, iterations, precision) =
        zeroin(f, lower, upper, f_lower, f_upper, tol, ROOT_MAX_ITER)?;
    Ok((
        root,
        RootInfo {
            iterations: iterations.unwrap_or(ROOT_MAX_ITER) + steps,
            extension_steps: steps,
            estimated_precision: precision,
            lower,
            upper,
            converged: iterations.is_some(),
        },
    ))
}

/// Solve `body(x) = target` and evaluate the body once more at the root.
fn solve_for_value(
    body: &mut impl FnMut(f64) -> EconResult<NcValue>,
    target: f64,
    interval: (f64, f64),
    extend: ExtendInt,
    tol: f64,
    param: &str,
) -> EconResult<(f64, RootInfo, NcValue)> {
    let mut f = |x: f64| -> EconResult<f64> { Ok(body(x)?.value - target) };
    let (root, info) = uniroot(&mut f, interval.0, interval.1, extend, tol, param)?;
    let at_root = body(root)?;
    Ok((root, info, at_root))
}

fn root_warnings(info: &RootInfo, param: &str, warnings: &mut Vec<String>) {
    if !info.converged {
        warnings.push(format!(
            "uniroot did not converge in {ROOT_MAX_ITER} iterations while solving for {param}; \
             the reported value is the last iterate"
        ));
    }
}

// ============================================================================
// Options
// ============================================================================

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

crate::option_enum!(
    TTestType,
    Choices {
        option: "type",
        values: &[
            (&["two.sample", "twosample", "two"], TTestType::TwoSample),
            (&["one.sample", "onesample", "one"], TTestType::OneSample),
            (&["paired"], TTestType::Paired),
        ],
    }
);

/// Alternative hypothesis specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PowerAlternative {
    /// Two-sided test (default)
    #[default]
    TwoSided,
    /// One-sided test
    OneSided,
}

crate::option_enum!(
    PowerAlternative,
    Choices {
        option: "alternative",
        values: &[
            (
                &["two.sided", "twosided", "two"],
                PowerAlternative::TwoSided
            ),
            (
                &["one.sided", "onesided", "one"],
                PowerAlternative::OneSided
            ),
        ],
    }
);

/// The parameter a t-test power calculation solves for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PowerTParam {
    N,
    Delta,
    Sd,
    SigLevel,
    #[default]
    Power,
}

crate::option_enum!(
    PowerTParam,
    Choices {
        option: "solve_for",
        values: &[
            (&["n"], PowerTParam::N),
            (&["delta"], PowerTParam::Delta),
            (&["sd"], PowerTParam::Sd),
            (&["sig.level", "alpha", "siglevel"], PowerTParam::SigLevel),
            (&["power"], PowerTParam::Power),
        ],
    }
);

/// The parameter a proportion-test power calculation solves for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PowerPropParam {
    N,
    P1,
    P2,
    SigLevel,
    #[default]
    Power,
}

crate::option_enum!(
    PowerPropParam,
    Choices {
        option: "solve_for",
        values: &[
            (&["n"], PowerPropParam::N),
            (&["p1"], PowerPropParam::P1),
            (&["p2"], PowerPropParam::P2),
            (
                &["sig.level", "alpha", "siglevel"],
                PowerPropParam::SigLevel
            ),
            (&["power"], PowerPropParam::Power),
        ],
    }
);

/// The parameter an ANOVA power calculation solves for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PowerAnovaParam {
    Groups,
    N,
    BetweenVar,
    WithinVar,
    SigLevel,
    #[default]
    Power,
}

crate::option_enum!(
    PowerAnovaParam,
    Choices {
        option: "solve_for",
        values: &[
            (&["groups"], PowerAnovaParam::Groups),
            (&["n"], PowerAnovaParam::N),
            (&["between.var", "betweenvar"], PowerAnovaParam::BetweenVar),
            (&["within.var", "withinvar"], PowerAnovaParam::WithinVar),
            (
                &["sig.level", "alpha", "siglevel"],
                PowerAnovaParam::SigLevel
            ),
            (&["power"], PowerAnovaParam::Power),
        ],
    }
);

/// Which of `params` (field name, value, has a default) is solved: `solve_for` when given (it must
/// be absent, every parameter without a default present), else the only absent one, else, after the
/// defaults, the only absent one without a default.
fn choose_unknown(
    params: &[(&'static str, Option<f64>, bool)],
    solve_for: Option<usize>,
) -> EconResult<usize> {
    if let Some(i) = solve_for {
        let (name, value, _) = params[i];
        if let Some(v) = value {
            return Err(invalid(format!(
                "solve_for = {name} but {name} = {v} was given; omit it"
            )));
        }
        for (j, (q, value, has_default)) in params.iter().enumerate() {
            if j != i && value.is_none() && !has_default {
                return Err(invalid(format!("{q} is required when solving for {name}")));
            }
        }
        return Ok(i);
    }
    let absent: Vec<usize> = (0..params.len())
        .filter(|&i| params[i].1.is_none())
        .collect();
    if absent.len() == 1 {
        return Ok(absent[0]);
    }
    let required: Vec<usize> = absent.iter().copied().filter(|&i| !params[i].2).collect();
    if required.len() == 1 {
        return Ok(required[0]);
    }
    let core: Vec<&str> = params.iter().filter(|p| !p.2).map(|p| p.0).collect();
    let defaulted: Vec<&str> = params.iter().filter(|p| p.2).map(|p| p.0).collect();
    let defaults = match defaulted.as_slice() {
        [one] => format!("{one} defaults to 0.05; set solve_for to solve for it"),
        _ => format!(
            "{} default to 1 and 0.05; set solve_for to solve for them",
            defaulted.join(" and ")
        ),
    };
    let omitted = if required.is_empty() {
        "nothing is omitted".to_string()
    } else {
        format!(
            "omitted: {}",
            required
                .iter()
                .map(|&i| params[i].0)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    Err(invalid(format!(
        "exactly one of {} must be omitted ({defaults}); {omitted}",
        core.join(", ")
    )))
}

// ============================================================================
// Display helpers
// ============================================================================

/// A number as R's `format(x, digits = 7)`: 7 significant digits, trailing zeros removed, in
/// fixed notation unless the scientific form is shorter (R's rule at `scipen = 0`).
fn format7(x: f64) -> String {
    if !x.is_finite() {
        return if x.is_nan() {
            "NaN".to_string()
        } else if x > 0.0 {
            "Inf".to_string()
        } else {
            "-Inf".to_string()
        };
    }
    if x == 0.0 {
        return "0".to_string();
    }
    let trim = |v: &str| -> String {
        if v.contains('.') {
            v.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            v.to_string()
        }
    };
    let sci = format!("{x:.6e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let mantissa = trim(mantissa);
    let sci = format!(
        "{mantissa}e{}{:02}",
        if exp < 0 { '-' } else { '+' },
        exp.abs()
    );
    let sig = mantissa.chars().filter(char::is_ascii_digit).count() as i32;
    let decimals = (sig - 1 - exp).max(0) as usize;
    let fixed = format!("{x:.decimals$}");
    if fixed.len() <= sci.len() { fixed } else { sci }
}

fn write_power_htest(
    f: &mut fmt::Formatter<'_>,
    method: &str,
    rows: &[(&str, String)],
    note: Option<&str>,
    notes: &dyn ResultNotes,
) -> fmt::Result {
    writeln!(f, "\n     {method}\n")?;
    for (name, value) in rows {
        writeln!(f, "{name:>15} = {value}")?;
    }
    if let Some(note) = note {
        writeln!(f, "\nNOTE: {note}")?;
    }
    write!(f, "{}", notes_text(notes))
}

// ============================================================================
// power.t.test
// ============================================================================

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
    /// Both tails counted for a two-sided test (R's `strict = TRUE`)
    #[serde(default)]
    pub strict: bool,
    /// The parameter that was computed
    #[serde(default)]
    pub solved_for: PowerTParam,
    /// The root search, when a parameter other than the power was solved
    #[serde(default)]
    pub root: Option<RootInfo>,
    /// Precision and convergence warnings at the reported values
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for PowerTTestResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for PowerTTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = [
            ("n", format7(self.n)),
            ("delta", format7(self.delta)),
            ("sd", format7(self.sd)),
            ("sig.level", format7(self.sig_level)),
            ("power", format7(self.power)),
            ("alternative", self.alternative.to_string()),
        ];
        write_power_htest(f, &self.method, &rows, self.note.as_deref(), self)
    }
}

/// A power.t.test call: any one of `n`, `delta`, `sd`, `sig_level`, `power` is solved for.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerTTestSpec {
    pub n: Option<f64>,
    pub delta: Option<f64>,
    pub sd: Option<f64>,
    pub sig_level: Option<f64>,
    pub power: Option<f64>,
    pub test_type: TTestType,
    pub alternative: PowerAlternative,
    /// R's `strict`: count both tails of a two-sided test (default false: the one tail R counts)
    pub strict: bool,
    /// Root-search tolerance (default [`R_UNIROOT_TOL`])
    pub tol: f64,
    /// The parameter to solve for; `None` infers it (see [`power_t_test_with`])
    pub solve_for: Option<PowerTParam>,
}

impl Default for PowerTTestSpec {
    fn default() -> Self {
        PowerTTestSpec {
            n: None,
            delta: None,
            sd: None,
            sig_level: None,
            power: None,
            test_type: TTestType::TwoSample,
            alternative: PowerAlternative::TwoSided,
            strict: false,
            tol: R_UNIROOT_TOL,
            solve_for: None,
        }
    }
}

/// R's power.t.test p.body.
fn t_power(
    n: f64,
    delta: f64,
    sd: f64,
    sig_level: f64,
    tsample: f64,
    tside: f64,
    strict: bool,
) -> EconResult<NcValue> {
    checkpoint()?;
    if n.is_nan() || delta.is_nan() || sd.is_nan() || sig_level.is_nan() {
        return Ok(NcValue::nan());
    }
    let nu = (n - 1.0).max(1e-7) * tsample;
    let tail = sig_level / tside;
    if !(tail > 0.0 && tail < 1.0) {
        return Ok(NcValue::nan());
    }
    let qu = -t_quantile(tail, nu)?;
    // R's order, unless sqrt(n / tsample) * delta overflows where the noncentrality does not
    let scaled = (n / tsample).sqrt() * delta;
    let ncp = if scaled.is_finite() {
        scaled / sd
    } else {
        (n / tsample).sqrt() * (delta / sd)
    };
    let upper = pnt(qu, nu, ncp, false);
    if strict && tside == 2.0 {
        let lower = pnt(-qu, nu, ncp, true);
        Ok(NcValue {
            value: upper.value + lower.value,
            imprecise: upper.imprecise || lower.imprecise,
        })
    } else {
        Ok(upper)
    }
}

const PNT_WARNING: &str = "the noncentral t probability at the reported values may not be fully \
                           accurate (R: full precision may not have been achieved in 'pnt')";
const PNF_WARNING: &str = "the noncentral F probability at the reported values may not be fully \
                           accurate (R: full precision may not have been achieved in 'pnbeta')";

/// Power of the one-sample, two-sample or paired t test, or the one parameter that gives the
/// requested power (R's `power.t.test`).
///
/// Which parameter is solved: `spec.solve_for` when given (it must be absent, and `n`, `delta`,
/// `power` other than it given); otherwise the only absent one of the five; otherwise `sd` and
/// `sig_level` take R's defaults (1 and 0.05) and exactly one of `n`, `delta`, `power` must be
/// absent. A two-sided `delta` is replaced by its absolute value, as R; a one-sided test counts the
/// upper tail whatever the sign of `delta` (R's convention).
///
/// The search intervals are R's: `n` on [2, 1e7] extended downwards if needed, `sd` on
/// `delta * [1e-7, 1e7]`, `delta` on `sd * [1e-7, 1e7]`, `sig_level` on [1e-10, 1 - 1e-10].
///
/// # Errors
///
/// [`EconError::InvalidSpecification`] for a parameter out of range (`n` not above 1, `sd` not
/// positive, `sig_level` not in (2^-53, 1), `power` not in (0, 1), `tol` not positive), for an
/// ambiguous call, and when the requested power is not reached on the search interval
/// ("cannot solve for ..."); [`EconError::Cancelled`] when the budget runs out;
/// [`EconError::Computation`] when the power is undefined.
pub fn power_t_test_with(spec: &PowerTTestSpec) -> EconResult<PowerTTestResult> {
    require_tol(spec.tol)?;
    if let Some(n) = spec.n
        && !(n.is_finite() && n > 1.0)
    {
        return Err(invalid(format!(
            "n must be greater than 1 (a t test needs a positive df), got {n:?}"
        )));
    }
    if let Some(d) = spec.delta
        && !d.is_finite()
    {
        return Err(invalid(format!("delta must be a finite number, got {d:?}")));
    }
    if let Some(sd) = spec.sd {
        require_finite_positive("sd", sd)?;
    }
    if let Some(s) = spec.sig_level {
        require_sig_level(s)?;
    }
    if let Some(p) = spec.power {
        require_open_unit("power", p)?;
    }
    const ORDER: [PowerTParam; 5] = [
        PowerTParam::N,
        PowerTParam::Delta,
        PowerTParam::Sd,
        PowerTParam::SigLevel,
        PowerTParam::Power,
    ];
    let params = [
        ("n", spec.n, false),
        ("delta", spec.delta, false),
        ("sd", spec.sd, true),
        ("sig_level", spec.sig_level, true),
        ("power", spec.power, false),
    ];
    let index = choose_unknown(
        &params,
        spec.solve_for
            .map(|p| ORDER.iter().position(|&q| q == p).unwrap_or(4)),
    )?;
    let unknown = ORDER[index];
    let tsample = if spec.test_type == TTestType::TwoSample {
        2.0
    } else {
        1.0
    };
    let two_sided = spec.alternative == PowerAlternative::TwoSided;
    let tside = if two_sided { 2.0 } else { 1.0 };
    let strict = spec.strict;
    let mut n = spec.n.unwrap_or(f64::NAN);
    let mut delta = spec
        .delta
        .map_or(f64::NAN, |d| if two_sided { d.abs() } else { d });
    let mut sd = match (spec.sd, unknown) {
        (Some(v), _) => v,
        (None, PowerTParam::Sd) => f64::NAN,
        (None, _) => 1.0,
    };
    let mut sig_level = match (spec.sig_level, unknown) {
        (Some(v), _) => v,
        (None, PowerTParam::SigLevel) => f64::NAN,
        (None, _) => 0.05,
    };
    let mut power = spec.power.unwrap_or(f64::NAN);
    let tol = spec.tol;
    let mut warnings = Vec::new();
    let mut root = None;
    let at_value = match unknown {
        PowerTParam::Power => {
            let v = t_power(n, delta, sd, sig_level, tsample, tside, strict)?;
            if v.value.is_nan() {
                return Err(EconError::Computation(format!(
                    "power is undefined for n = {n:?}, delta = {delta:?}, sd = {sd:?}"
                )));
            }
            power = v.value;
            v
        }
        PowerTParam::N => {
            let mut body = |x: f64| t_power(x, delta, sd, sig_level, tsample, tside, strict);
            let (x, info, v) =
                solve_for_value(&mut body, power, (2.0, 1e7), ExtendInt::UpX, tol, "n")?;
            n = x;
            root = Some(info);
            v
        }
        PowerTParam::Sd => {
            if delta.is_nan() || delta <= 0.0 {
                return Err(invalid(format!(
                    "solving for sd needs delta > 0 (R searches delta * [1e-7, 1e7]), got delta = {delta:?}"
                )));
            }
            let mut body = |x: f64| t_power(n, delta, x, sig_level, tsample, tside, strict);
            let interval = (delta * 1e-7, (delta * 1e7).min(f64::MAX));
            let (x, info, v) =
                solve_for_value(&mut body, power, interval, ExtendInt::DownX, tol, "sd")?;
            sd = x;
            root = Some(info);
            v
        }
        PowerTParam::Delta => {
            if !(10.0 * sd).is_finite() {
                let max_sd = f64::MAX / 10.0;
                return Err(invalid(format!(
                    "sd must be at most {max_sd:e} when solving for delta, got {sd:?}"
                )));
            }
            let mut body = |x: f64| t_power(n, x, sd, sig_level, tsample, tside, strict);
            let interval = (sd * 1e-7, (sd * 1e7).min(f64::MAX));
            let (x, info, v) =
                solve_for_value(&mut body, power, interval, ExtendInt::UpX, tol, "delta")?;
            delta = x;
            root = Some(info);
            v
        }
        PowerTParam::SigLevel => {
            let mut body = |x: f64| t_power(n, delta, sd, x, tsample, tside, strict);
            let (x, info, v) = solve_for_value(
                &mut body,
                power,
                (1e-10, 1.0 - 1e-10),
                ExtendInt::Yes,
                tol,
                "sig_level",
            )?;
            sig_level = x;
            root = Some(info);
            v
        }
    };
    if at_value.imprecise {
        warnings.push(PNT_WARNING.to_string());
    }
    if let Some(info) = &root {
        root_warnings(info, params[index].0, &mut warnings);
    }
    let method = match spec.test_type {
        TTestType::TwoSample => "Two-sample t test power calculation",
        TTestType::OneSample => "One-sample t test power calculation",
        TTestType::Paired => "Paired t test power calculation",
    };
    let note = match spec.test_type {
        TTestType::TwoSample => Some("n is number in *each* group".to_string()),
        TTestType::Paired => {
            Some("n is number of *pairs*, sd is std.dev. of *differences* within pairs".to_string())
        }
        TTestType::OneSample => None,
    };
    Ok(PowerTTestResult {
        n,
        delta,
        sd,
        sig_level,
        power,
        alternative: spec.alternative,
        test_type: spec.test_type,
        method: method.to_string(),
        note,
        strict,
        solved_for: unknown,
        root,
        warnings,
    })
}

/// Power of a t test, or the one parameter that gives the requested power (R's `power.t.test` with
/// `strict = FALSE` and R's default tolerance).
///
/// The parameter left `None` is solved; when more than one is `None`, `sd` and `sig_level` take R's
/// defaults 1 and 0.05 and exactly one of `n`, `delta`, `power` must be `None` (see
/// [`power_t_test_with`], which also takes `strict`, `tol` and `solve_for`).
///
/// A one-sided test here is taken in the direction of `delta` (the lower tail when `delta < 0`), so
/// the power depends only on |delta|; [`power_t_test_with`] follows R, which tests the upper tail
/// whatever the sign of delta (power 1.0e-6 at n = 20, delta = -1, one-sided).
///
/// # Errors
///
/// As [`power_t_test_with`].
pub fn power_t_test(
    n: Option<f64>,
    delta: Option<f64>,
    sd: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    test_type: TTestType,
    alternative: PowerAlternative,
) -> EconResult<PowerTTestResult> {
    let one_sided = alternative == PowerAlternative::OneSided;
    let spec = PowerTTestSpec {
        n,
        delta: if one_sided {
            delta.map(f64::abs)
        } else {
            delta
        },
        sd,
        sig_level,
        power,
        test_type,
        alternative,
        ..Default::default()
    };
    let mut r = power_t_test_with(&spec)?;
    if one_sided && let Some(d) = delta {
        r.delta = d;
    }
    Ok(r)
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
    /// Both tails counted for a two-sided test (R's `strict = TRUE`)
    #[serde(default)]
    pub strict: bool,
    /// The parameter that was computed
    #[serde(default)]
    pub solved_for: PowerPropParam,
    /// The root search, when a parameter other than the power was solved
    #[serde(default)]
    pub root: Option<RootInfo>,
    /// R's warnings (a solution outside [0, 1]) and convergence warnings
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for PowerPropTestResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for PowerPropTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = [
            ("n", format7(self.n)),
            ("p1", format7(self.p1)),
            ("p2", format7(self.p2)),
            ("sig.level", format7(self.sig_level)),
            ("power", format7(self.power)),
            ("alternative", self.alternative.to_string()),
        ];
        write_power_htest(f, &self.method, &rows, Some(&self.note), self)
    }
}

/// A power.prop.test call: any one of `n`, `p1`, `p2`, `sig_level`, `power` is solved for.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerPropTestSpec {
    pub n: Option<f64>,
    pub p1: Option<f64>,
    pub p2: Option<f64>,
    pub sig_level: Option<f64>,
    pub power: Option<f64>,
    pub alternative: PowerAlternative,
    /// R's `strict`
    pub strict: bool,
    /// Root-search tolerance (default [`R_UNIROOT_TOL`])
    pub tol: f64,
    /// The parameter to solve for; `None` infers it
    pub solve_for: Option<PowerPropParam>,
}

impl Default for PowerPropTestSpec {
    fn default() -> Self {
        PowerPropTestSpec {
            n: None,
            p1: None,
            p2: None,
            sig_level: None,
            power: None,
            alternative: PowerAlternative::TwoSided,
            strict: false,
            tol: R_UNIROOT_TOL,
            solve_for: None,
        }
    }
}

fn std_normal_cdf(x: f64, lower_tail: bool) -> f64 {
    r_mathlib::normal_cdf(x, 0.0, 1.0, lower_tail, false)
}

/// R's power.prop.test p.body.
fn prop_power(
    n: f64,
    p1: f64,
    p2: f64,
    sig_level: f64,
    tside: f64,
    strict: bool,
) -> EconResult<NcValue> {
    checkpoint()?;
    let tail = sig_level / tside;
    if !(tail > 0.0 && tail < 1.0) || n.is_nan() || p1.is_nan() || p2.is_nan() {
        return Ok(NcValue::nan());
    }
    let qu = -normal_quantile(tail)?;
    let value = if strict && tside == 2.0 {
        let d = (p1 - p2).abs();
        let q1 = 1.0 - p1;
        let q2 = 1.0 - p2;
        let pbar = (p1 + p2) / 2.0;
        let qbar = 1.0 - pbar;
        let v1 = p1 * q1;
        let v2 = p2 * q2;
        let vbar = pbar * qbar;
        let sd = (v1 + v2).sqrt();
        std_normal_cdf((n.sqrt() * d - qu * (2.0 * vbar).sqrt()) / sd, true)
            + std_normal_cdf((n.sqrt() * d + qu * (2.0 * vbar).sqrt()) / sd, false)
    } else {
        std_normal_cdf(
            (n.sqrt() * (p1 - p2).abs() - (qu * ((p1 + p2) * (1.0 - (p1 + p2) / 2.0)).sqrt()))
                / (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt(),
            true,
        )
    };
    Ok(NcValue::exact(value))
}

fn require_proportion(name: &str, p: f64) -> EconResult<()> {
    if (0.0..=1.0).contains(&p) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{name} must be a probability in [0, 1], got {p:?}"
        )))
    }
}

/// Power of the two-sample comparison of proportions, or the one parameter that gives the requested
/// power (R's `power.prop.test`).
///
/// Which parameter is solved: `solve_for` when given; otherwise the only absent one; otherwise
/// `sig_level` takes R's default 0.05 and exactly one of `n`, `p1`, `p2`, `power` must be absent.
/// R's intervals: `n` on [1, 1e7] (extended upwards), `p1` on [0, p2], `p2` on [p1, 1] (both
/// extended either way; R's warning when the solution leaves [0, 1]), `sig_level` on
/// [1e-10, 1 - 1e-10] (extended upwards).
///
/// # Errors
///
/// [`EconError::InvalidSpecification`] for a parameter out of range (`n` below 1, a proportion
/// outside [0, 1], both proportions 0 or both 1, `sig_level` not in (2^-53, 1), `power` not in
/// (0, 1)), for an ambiguous call and for an unreachable power; [`EconError::Cancelled`].
pub fn power_prop_test_with(spec: &PowerPropTestSpec) -> EconResult<PowerPropTestResult> {
    require_tol(spec.tol)?;
    if let Some(n) = spec.n {
        // as R's power.prop.test: at least one observation per group
        require_finite_at_least("n", n, 1.0)?;
    }
    if let Some(p) = spec.p1 {
        require_proportion("p1", p)?;
    }
    if let Some(p) = spec.p2 {
        require_proportion("p2", p)?;
    }
    if let (Some(a), Some(b)) = (spec.p1, spec.p2)
        && a == b
        && (a == 0.0 || a == 1.0)
    {
        return Err(invalid(format!(
            "p1 and p2 must not both be 0 or both be 1, got p1 = {a:?}, p2 = {b:?}"
        )));
    }
    if let Some(s) = spec.sig_level {
        require_sig_level(s)?;
    }
    if let Some(p) = spec.power {
        require_open_unit("power", p)?;
    }
    const ORDER: [PowerPropParam; 5] = [
        PowerPropParam::N,
        PowerPropParam::P1,
        PowerPropParam::P2,
        PowerPropParam::SigLevel,
        PowerPropParam::Power,
    ];
    let params = [
        ("n", spec.n, false),
        ("p1", spec.p1, false),
        ("p2", spec.p2, false),
        ("sig_level", spec.sig_level, true),
        ("power", spec.power, false),
    ];
    let index = choose_unknown(
        &params,
        spec.solve_for
            .map(|p| ORDER.iter().position(|&q| q == p).unwrap_or(4)),
    )?;
    let unknown = ORDER[index];
    let tside = if spec.alternative == PowerAlternative::TwoSided {
        2.0
    } else {
        1.0
    };
    let strict = spec.strict;
    let mut n = spec.n.unwrap_or(f64::NAN);
    let mut p1 = spec.p1.unwrap_or(f64::NAN);
    let mut p2 = spec.p2.unwrap_or(f64::NAN);
    let mut sig_level = match (spec.sig_level, unknown) {
        (Some(v), _) => v,
        (None, PowerPropParam::SigLevel) => f64::NAN,
        (None, _) => 0.05,
    };
    let mut power = spec.power.unwrap_or(f64::NAN);
    let tol = spec.tol;
    let mut warnings = Vec::new();
    let mut root = None;
    match unknown {
        PowerPropParam::Power => {
            let v = prop_power(n, p1, p2, sig_level, tside, strict)?;
            if v.value.is_nan() {
                return Err(EconError::Computation(format!(
                    "power is undefined for n = {n:?}, p1 = {p1:?}, p2 = {p2:?}"
                )));
            }
            power = v.value;
        }
        PowerPropParam::N => {
            let mut body = |x: f64| prop_power(x, p1, p2, sig_level, tside, strict);
            let (x, info, _) =
                solve_for_value(&mut body, power, (1.0, 1e7), ExtendInt::UpX, tol, "n")?;
            n = x;
            root = Some(info);
        }
        PowerPropParam::P1 => {
            let mut body = |x: f64| prop_power(n, x, p2, sig_level, tside, strict);
            let (x, info, _) =
                solve_for_value(&mut body, power, (0.0, p2), ExtendInt::Yes, tol, "p1")?;
            p1 = x;
            root = Some(info);
            if p1 < 0.0 {
                warnings
                    .push("No p1 in [0, p2] can be found to achieve the desired power".to_string());
            }
        }
        PowerPropParam::P2 => {
            let mut body = |x: f64| prop_power(n, p1, x, sig_level, tside, strict);
            let (x, info, _) =
                solve_for_value(&mut body, power, (p1, 1.0), ExtendInt::Yes, tol, "p2")?;
            p2 = x;
            root = Some(info);
            if p2 > 1.0 {
                warnings
                    .push("No p2 in [p1, 1] can be found to achieve the desired power".to_string());
            }
        }
        PowerPropParam::SigLevel => {
            let mut body = |x: f64| prop_power(n, p1, p2, x, tside, strict);
            let (x, info, _) = solve_for_value(
                &mut body,
                power,
                (1e-10, 1.0 - 1e-10),
                ExtendInt::UpX,
                tol,
                "sig_level",
            )?;
            sig_level = x;
            root = Some(info);
            if !(0.0..=1.0).contains(&sig_level) {
                warnings.push(
                    "No significance level [0, 1] can be found to achieve the desired power"
                        .to_string(),
                );
            }
        }
    }
    if let Some(info) = &root {
        root_warnings(info, params[index].0, &mut warnings);
    }
    Ok(PowerPropTestResult {
        n,
        p1,
        p2,
        sig_level,
        power,
        alternative: spec.alternative,
        method: "Two-sample comparison of proportions power calculation".to_string(),
        note: "n is number in *each* group".to_string(),
        strict,
        solved_for: unknown,
        root,
        warnings,
    })
}

/// Power of a two-sample proportion test, or the one parameter that gives the requested power (R's
/// `power.prop.test` with `strict = FALSE` and R's default tolerance; see
/// [`power_prop_test_with`]).
///
/// # Errors
///
/// As [`power_prop_test_with`].
pub fn power_prop_test(
    n: Option<f64>,
    p1: f64,
    p2: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    alternative: PowerAlternative,
) -> EconResult<PowerPropTestResult> {
    power_prop_test_with(&PowerPropTestSpec {
        n,
        p1: Some(p1),
        p2,
        sig_level,
        power,
        alternative,
        ..Default::default()
    })
}

// ============================================================================
// power.anova.test
// ============================================================================

/// Result of power.anova.test calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnovaTestResult {
    /// Number of groups (fractional when solved for, as R)
    pub groups: f64,
    /// Sample size per group
    pub n: f64,
    /// Between-group variance (the variance of the group means, divisor groups - 1)
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
    /// The parameter that was computed
    #[serde(default)]
    pub solved_for: PowerAnovaParam,
    /// The root search, when a parameter other than the power was solved
    #[serde(default)]
    pub root: Option<RootInfo>,
    /// Precision and convergence warnings at the reported values
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Cohen's f = sqrt((groups - 1) / groups * between_var / within_var)
    #[serde(default)]
    pub effect_size_f: f64,
}

impl ResultNotes for PowerAnovaTestResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for PowerAnovaTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rows = [
            ("groups", format7(self.groups)),
            ("n", format7(self.n)),
            ("between.var", format7(self.between_var)),
            ("within.var", format7(self.within_var)),
            ("sig.level", format7(self.sig_level)),
            ("power", format7(self.power)),
        ];
        write_power_htest(f, &self.method, &rows, Some(&self.note), self)
    }
}

/// A power.anova.test call: any one of the six parameters is solved for.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerAnovaTestSpec {
    pub groups: Option<f64>,
    pub n: Option<f64>,
    pub between_var: Option<f64>,
    pub within_var: Option<f64>,
    pub sig_level: Option<f64>,
    pub power: Option<f64>,
    /// Root-search tolerance (default [`R_UNIROOT_TOL`], which R's power.anova.test always uses)
    pub tol: f64,
    /// The parameter to solve for; `None` infers it
    pub solve_for: Option<PowerAnovaParam>,
}

impl Default for PowerAnovaTestSpec {
    fn default() -> Self {
        PowerAnovaTestSpec {
            groups: None,
            n: None,
            between_var: None,
            within_var: None,
            sig_level: None,
            power: None,
            tol: R_UNIROOT_TOL,
            solve_for: None,
        }
    }
}

/// R's power.anova.test p.body. An infinite noncentrality (beyond f64::MAX) has power 1, its limit.
fn anova_power(
    groups: f64,
    n: f64,
    between: f64,
    within: f64,
    sig_level: f64,
) -> EconResult<NcValue> {
    checkpoint()?;
    let df1 = groups - 1.0;
    let df2 = (n - 1.0) * groups;
    let lambda = (groups - 1.0) * n * (between / within);
    if df1.is_nan() || df2.is_nan() || lambda.is_nan() || df1 <= 0.0 || df2 <= 0.0 {
        return Ok(NcValue::nan());
    }
    if lambda == f64::INFINITY {
        return Ok(NcValue::exact(1.0));
    }
    let q = f_upper_quantile(sig_level, df1, df2)?;
    Ok(pnf(q, df1, df2, lambda, false))
}

/// Power of the balanced one-way ANOVA F test, or the one parameter that gives the requested power
/// (R's `power.anova.test`).
///
/// Which parameter is solved: `solve_for` when given; otherwise the only absent one; otherwise
/// `sig_level` takes R's default 0.05 and exactly one of `groups`, `n`, `between_var`, `within_var`,
/// `power` must be absent. The noncentrality is R's `(groups - 1) n between_var / within_var`, with
/// `between_var` the variance of the group means. R's intervals, never extended: `groups` [2, 100],
/// `n` [2, 1e5], `within_var` `between_var * [1e-7, 1e7]`, `between_var` `within_var * [1e-7, 1e7]`,
/// `sig_level` [1e-10, 1 - 1e-10]. Solutions are fractional (R reports 13.86 groups).
///
/// # Errors
///
/// [`EconError::InvalidSpecification`] for a parameter out of range (`groups` or `n` below 2, or
/// so large that `groups * (n - 1)` is not finite, `between_var` negative, `within_var` not
/// positive, `sig_level` not in (2^-53, 1), `power` not in (0, 1)), for an ambiguous call and for an
/// unreachable power ("cannot solve for ..."); [`EconError::Cancelled`];
/// [`EconError::Computation`] when the power or a critical value is undefined.
pub fn power_anova_test_with(spec: &PowerAnovaTestSpec) -> EconResult<PowerAnovaTestResult> {
    require_tol(spec.tol)?;
    if let Some(g) = spec.groups
        && !(g.is_finite() && g >= 2.0)
    {
        return Err(invalid(format!("groups must be at least 2, got {g}")));
    }
    if let Some(n) = spec.n {
        if !(n.is_finite() && n >= 2.0) {
            return Err(invalid(format!(
                "n must be at least 2 (observations in each group), got {n:?}"
            )));
        }
        if let Some(g) = spec.groups
            && !(g * (n - 1.0)).is_finite()
        {
            // the largest n the check accepts: f64::MAX / g itself fails it when
            // g * (f64::MAX / g - 1) rounds up to infinity (g not a power of two)
            let mut max_n = f64::MAX / g;
            if !(g * (max_n - 1.0)).is_finite() {
                max_n = max_n.next_down();
            }
            let g = count_text(g);
            return Err(invalid(format!(
                "n must be at most {max_n:e} for groups = {g} (groups * (n - 1), \
                 the error degrees of freedom, must be finite), got {n:e}"
            )));
        }
    }
    if let Some(b) = spec.between_var {
        require_finite_at_least("between_var", b, 0.0)?;
    }
    if let Some(w) = spec.within_var {
        require_finite_positive("within_var", w)?;
    }
    if let Some(s) = spec.sig_level {
        require_sig_level(s)?;
    }
    if let Some(p) = spec.power {
        require_open_unit("power", p)?;
    }
    const ORDER: [PowerAnovaParam; 6] = [
        PowerAnovaParam::Groups,
        PowerAnovaParam::N,
        PowerAnovaParam::BetweenVar,
        PowerAnovaParam::WithinVar,
        PowerAnovaParam::SigLevel,
        PowerAnovaParam::Power,
    ];
    let params = [
        ("groups", spec.groups, false),
        ("n", spec.n, false),
        ("between_var", spec.between_var, false),
        ("within_var", spec.within_var, false),
        ("sig_level", spec.sig_level, true),
        ("power", spec.power, false),
    ];
    let index = choose_unknown(
        &params,
        spec.solve_for
            .map(|p| ORDER.iter().position(|&q| q == p).unwrap_or(5)),
    )?;
    let unknown = ORDER[index];
    let mut groups = spec.groups.unwrap_or(f64::NAN);
    let mut n = spec.n.unwrap_or(f64::NAN);
    let mut between = spec.between_var.unwrap_or(f64::NAN);
    let mut within = spec.within_var.unwrap_or(f64::NAN);
    let mut sig_level = match (spec.sig_level, unknown) {
        (Some(v), _) => v,
        (None, PowerAnovaParam::SigLevel) => f64::NAN,
        (None, _) => 0.05,
    };
    let mut power = spec.power.unwrap_or(f64::NAN);
    let tol = spec.tol;
    let mut root = None;
    let no = ExtendInt::No;
    let at_value = match unknown {
        PowerAnovaParam::Power => {
            let v = anova_power(groups, n, between, within, sig_level)?;
            if v.value.is_nan() {
                return Err(EconError::Computation(format!(
                    "power is undefined for groups = {groups:?}, n = {n:?}"
                )));
            }
            power = v.value;
            v
        }
        PowerAnovaParam::Groups => {
            let mut body = |x: f64| anova_power(x, n, between, within, sig_level);
            let (x, info, v) = solve_for_value(&mut body, power, (2.0, 100.0), no, tol, "groups")?;
            groups = x;
            root = Some(info);
            v
        }
        PowerAnovaParam::N => {
            let mut body = |x: f64| anova_power(groups, x, between, within, sig_level);
            let (x, info, v) = solve_for_value(&mut body, power, (2.0, 1e5), no, tol, "n")?;
            n = x;
            root = Some(info);
            v
        }
        PowerAnovaParam::WithinVar => {
            if between.is_nan() || between <= 0.0 {
                return Err(invalid(format!(
                    "solving for within_var needs between_var > 0 (R searches between_var * [1e-7, 1e7]), got between_var = {between:?}"
                )));
            }
            let mut body = |x: f64| anova_power(groups, n, between, x, sig_level);
            let interval = (between * 1e-7, (between * 1e7).min(f64::MAX));
            let (x, info, v) = solve_for_value(&mut body, power, interval, no, tol, "within_var")?;
            within = x;
            root = Some(info);
            v
        }
        PowerAnovaParam::BetweenVar => {
            let mut body = |x: f64| anova_power(groups, n, x, within, sig_level);
            let interval = (within * 1e-7, (within * 1e7).min(f64::MAX));
            let (x, info, v) = solve_for_value(&mut body, power, interval, no, tol, "between_var")?;
            between = x;
            root = Some(info);
            v
        }
        PowerAnovaParam::SigLevel => {
            let mut body = |x: f64| anova_power(groups, n, between, within, x);
            let (x, info, v) =
                solve_for_value(&mut body, power, (1e-10, 1.0 - 1e-10), no, tol, "sig_level")?;
            sig_level = x;
            root = Some(info);
            v
        }
    };
    let mut warnings = Vec::new();
    if at_value.imprecise {
        warnings.push(PNF_WARNING.to_string());
    }
    if let Some(info) = &root {
        root_warnings(info, params[index].0, &mut warnings);
    }
    let effect_size_f = ((groups - 1.0) / groups * between / within).sqrt();
    Ok(PowerAnovaTestResult {
        groups,
        n,
        between_var: between,
        within_var: within,
        sig_level,
        power,
        method: "Balanced one-way analysis of variance power calculation".to_string(),
        note: "n is number in each group".to_string(),
        solved_for: unknown,
        root,
        warnings,
        effect_size_f,
    })
}

/// Power of a balanced one-way ANOVA, or the one of `n`, `sig_level` and `power` left `None` (R's
/// `power.anova.test`; see [`power_anova_test_with`] to solve for `groups`, `between_var` or
/// `within_var`).
///
/// # Errors
///
/// As [`power_anova_test_with`].
pub fn power_anova_test(
    groups: usize,
    n: Option<f64>,
    between_var: f64,
    within_var: f64,
    sig_level: Option<f64>,
    power: Option<f64>,
) -> EconResult<PowerAnovaTestResult> {
    power_anova_test_with(&PowerAnovaTestSpec {
        groups: Some(groups as f64),
        n,
        between_var: Some(between_var),
        within_var: Some(within_var),
        sig_level,
        power,
        ..Default::default()
    })
}

// ============================================================================
// MCP-friendly wrappers
// ============================================================================

/// Run power.t.test with string parameters (R's, the CLI's and the MCP spellings of `type` and
/// `alternative`; an unknown value is an error listing the accepted ones).
pub fn run_power_t_test(
    n: Option<f64>,
    delta: Option<f64>,
    sd: Option<f64>,
    sig_level: Option<f64>,
    power: Option<f64>,
    test_type: &str,
    alternative: &str,
) -> EconResult<PowerTTestResult> {
    let test_type = test_type.parse::<TTestType>()?;
    let alternative = alternative.parse::<PowerAlternative>()?;
    power_t_test(n, delta, sd, sig_level, power, test_type, alternative)
}

/// Run power.prop.test with a string `alternative`.
pub fn run_power_prop_test(
    n: Option<f64>,
    p1: f64,
    p2: f64,
    sig_level: Option<f64>,
    power: Option<f64>,
    alternative: &str,
) -> EconResult<PowerPropTestResult> {
    let alternative = alternative.parse::<PowerAlternative>()?;
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
    use p2a_validation::{Case, Check, Tol};

    // =========================================================================
    // Properties (unchanged from Tier 0)
    // =========================================================================

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
    // The noncentral distributions and the root finder
    // =========================================================================

    #[test]
    fn test_validate_noncentral_t_f_ports() {
        // R's pt(t, df, ncp) and pf(x, df1, df2, ncp), one point per branch of the series and
        // their limits (case stat_power_solve). The published series with R's constants sum the
        // same terms as R (R accumulates in long double): agreement at rounding level. pf[4] =
        // 7.9e-10 is an upper tail R itself forms as 1 - (lower tail); pf[5] has df2 = 2e8, the
        // noncentral chi-square limit.
        let case = Case::load("stat_power_solve");
        let (t, df, ncp, lo) = (
            case.vector("pt_t"),
            case.vector("pt_df"),
            case.vector("pt_ncp"),
            case.vector("pt_lower"),
        );
        let pt: Vec<f64> = (0..t.len())
            .map(|i| pnt(t[i], df[i], ncp[i], lo[i] == 1.0).value)
            .collect();
        let (x, d1, d2, nc, lf) = (
            case.vector("pf_x"),
            case.vector("pf_df1"),
            case.vector("pf_df2"),
            case.vector("pf_ncp"),
            case.vector("pf_lower"),
        );
        let pf: Vec<f64> = (0..x.len())
            .map(|i| pnf(x[i], d1[i], d2[i], nc[i], lf[i] == 1.0).value)
            .collect();
        Check::new(&case)
            .vector("pt_value", &pt, Tol::new(1e-12, 1e-14))
            .vector("pf_value", &pf, Tol::new(1e-12, 1e-14))
            .finish();
    }

    #[test]
    fn test_noncentral_limits_are_continuous_at_their_switches() {
        // The normal limit of the t (df > 4e5) and the chi-square limit of the F (df2 > 1e8)
        // agree with the series just below the switch.
        let below = pnt(2.1, 4e5, 1.79, false).value;
        let above = pnt(2.1, 4e5 * (1.0 + 1e-9), 1.79, false).value;
        assert!((below - above).abs() < 1e-5, "{below} vs {above}");
        let below = pnf(2.5, 3.0, 1e8, 4.0, false).value;
        let above = pnf(2.5, 3.0, 1e8 * (1.0 + 1e-9), 4.0, false).value;
        assert!((below - above).abs() < 1e-6, "{below} vs {above}");
        // the noncentral chi-square series against its own normal approximation far out
        let series = pnchisq(2.1e5, 1e4, 2e5, true).value;
        let k: f64 = 1e4 + 2e5;
        let h2 = 2.0 * (1e4 + 4e5) / (9.0 * k * k);
        let z = ((2.1e5 / k).cbrt() - (1.0 - h2)) / h2.sqrt();
        let approx = r_mathlib::normal_cdf(z, 0.0, 1.0, true, false);
        assert!((series - approx).abs() < 1e-3, "{series} vs {approx}");
        // and its total mass
        assert!((pnchisq(1e9, 5.0, 3.0, true).value - 1.0).abs() < 1e-12);
        assert_eq!(pnchisq(0.0, 5.0, 3.0, true).value, 0.0);
    }

    #[test]
    fn test_uniroot_extends_the_bracket() {
        // R: uniroot(function(x) x^3 - 2, c(2, 3), extendInt = "upX", tol = 1e-12): root 2^(1/3)
        // after extending the lower end (f(2) = 6 > 0); "no" with the same interval is R's sign
        // error.
        let mut f = |x: f64| -> EconResult<f64> { Ok(x * x * x - 2.0) };
        let (root, info) = uniroot(&mut f, 2.0, 3.0, ExtendInt::UpX, 1e-12, "x").unwrap();
        assert!((root - 2f64.cbrt()).abs() < 1e-11, "{root}");
        assert!(info.converged && info.extension_steps > 0 && info.lower < 2.0);
        let e = uniroot(&mut f, 2.0, 3.0, ExtendInt::No, 1e-12, "x")
            .unwrap_err()
            .to_string();
        assert!(e.contains("cannot solve for x"), "{e}");
        // an empty interval is R's "lower < upper is not fulfilled"
        assert!(uniroot(&mut f, 3.0, 3.0, ExtendInt::No, 1e-12, "x").is_err());
        // both ends move with "yes"; downX extends towards a decreasing crossing
        let (root, info) = uniroot(&mut f, 3.0, 4.0, ExtendInt::Yes, 1e-12, "x").unwrap();
        assert!((root - 2f64.cbrt()).abs() < 1e-11 && info.extension_steps > 0);
        let mut g = |x: f64| -> EconResult<f64> { Ok(2.0 - x * x * x) };
        let (root, _) = uniroot(&mut g, 2.0, 3.0, ExtendInt::DownX, 1e-12, "x").unwrap();
        assert!((root - 2f64.cbrt()).abs() < 1e-11, "{root}");
        // a function with no root: the extension stops with an error, not a loop
        let mut h = |x: f64| -> EconResult<f64> { Ok(1.0 + x * x) };
        assert!(uniroot(&mut h, 2.0, 3.0, ExtendInt::Yes, 1e-12, "x").is_err());
        assert!(uniroot(&mut h, 2.0, 3.0, ExtendInt::UpX, 1e-12, "x").is_err());
    }

    // =========================================================================
    // Power as R: stat_power_solve
    // =========================================================================

    /// Direct power: the series and WP02's quantile helpers against R's.
    const POWER_TOL: Tol = Tol::new(1e-10, 1e-14);
    /// Roots at tol 1e-12 on both sides: the helper differences move a root by < 1e-11.
    const ROOT_TOL: Tol = Tol::new(1e-9, 1e-12);
    /// A root at R's default tolerance against R's own default-tolerance root: each is within
    /// tol + 4 eps |x| of the exact root (Brent's bound), so within 2 tol of each other.
    const PKG_TOL: Tol = Tol::new(8.0 * f64::EPSILON, 2.0 * R_UNIROOT_TOL);
    /// A root at R's default tolerance against R's tight root.
    const PKG_EXACT_TOL: Tol = Tol::new(4.0 * f64::EPSILON, R_UNIROOT_TOL);
    const TYPES: [(TTestType, &str); 3] = [
        (TTestType::TwoSample, "twosample"),
        (TTestType::OneSample, "onesample"),
        (TTestType::Paired, "paired"),
    ];
    const ALTS: [(PowerAlternative, &str); 2] = [
        (PowerAlternative::TwoSided, "twosided"),
        (PowerAlternative::OneSided, "onesided"),
    ];

    fn t_spec(test_type: TTestType, alternative: PowerAlternative) -> PowerTTestSpec {
        PowerTTestSpec {
            sd: Some(1.2),
            sig_level: Some(0.05),
            test_type,
            alternative,
            tol: 1e-12,
            ..Default::default()
        }
    }

    fn two() -> PowerTTestSpec {
        t_spec(TTestType::TwoSample, PowerAlternative::TwoSided)
    }

    #[test]
    fn test_validate_power_legacy_api() {
        // The unchanged functions, at R's default tolerance and R's defaults sd = 1, sig = 0.05.
        let case = Case::load("stat_power_solve");
        let (ts, two_sided) = (TTestType::TwoSample, PowerAlternative::TwoSided);
        let t = |n, d, sd, sig, p, ty| power_t_test(n, d, sd, sig, p, ty, two_sided).unwrap();
        let n_pkg = t(None, Some(0.5), None, None, Some(0.8), ts).n;
        let sd_pkg = t(Some(20.0), Some(0.8), None, Some(0.05), Some(0.9), ts).sd;
        let prop_n = power_prop_test(None, 0.3, Some(0.5), None, Some(0.8), two_sided)
            .unwrap()
            .n;
        let anova_n = power_anova_test(3, None, 0.25, 1.0, None, Some(0.8))
            .unwrap()
            .n;
        Check::new(&case)
            .scalar(
                "t_n30_d05_power",
                t(Some(30.0), Some(0.5), None, None, None, ts).power,
                POWER_TOL,
            )
            .scalar(
                "t_default_sd_sig_power",
                t(Some(15.0), Some(1.0), None, None, None, ts).power,
                POWER_TOL,
            )
            .scalar(
                "t_onesample_n10_d1_power",
                t(
                    Some(10.0),
                    Some(1.0),
                    None,
                    None,
                    None,
                    TTestType::OneSample,
                )
                .power,
                POWER_TOL,
            )
            .scalar("t_d05_p08_n_pkg", n_pkg, PKG_TOL)
            .scalar("t_d05_p08_n", n_pkg, PKG_EXACT_TOL)
            .scalar("t_twosample_twosided_sd_pkg", sd_pkg, PKG_TOL)
            .scalar("t_twosample_twosided_sd", sd_pkg, PKG_EXACT_TOL)
            .scalar(
                "prop_n100_power",
                power_prop_test(Some(100.0), 0.3, Some(0.5), None, None, two_sided)
                    .unwrap()
                    .power,
                POWER_TOL,
            )
            .scalar("prop_p03_p05_n_pkg", prop_n, PKG_TOL)
            .scalar("prop_p03_p05_n", prop_n, PKG_EXACT_TOL)
            .scalar(
                "anova_power",
                power_anova_test(4, Some(5.0), 1.0, 3.0, None, None)
                    .unwrap()
                    .power,
                POWER_TOL,
            )
            .scalar(
                "anova_g3_n20_power",
                power_anova_test(3, Some(20.0), 0.25, 1.0, None, None)
                    .unwrap()
                    .power,
                POWER_TOL,
            )
            .scalar("anova_g3_n_pkg", anova_n, PKG_TOL)
            .scalar("anova_g3_n", anova_n, PKG_EXACT_TOL)
            .scalar(
                "t_twosample_twosided_power",
                run_power_t_test(
                    Some(20.0),
                    Some(0.8),
                    Some(1.2),
                    Some(0.05),
                    None,
                    "two.sample",
                    "two.sided",
                )
                .unwrap()
                .power,
                POWER_TOL,
            )
            .finish();
    }

    #[test]
    fn test_validate_power_t_solve_power() {
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        for (ty, ty_name) in TYPES {
            for (alt, alt_name) in ALTS {
                let r = power_t_test_with(&PowerTTestSpec {
                    n: Some(20.0),
                    delta: Some(0.8),
                    ..t_spec(ty, alt)
                })
                .unwrap();
                assert_eq!(r.solved_for, PowerTParam::Power);
                assert!(r.root.is_none() && r.warnings.is_empty());
                check.scalar(&format!("t_{ty_name}_{alt_name}_power"), r.power, POWER_TOL);
            }
        }
        let loose = power_t_test_with(&PowerTTestSpec {
            n: Some(20.0),
            delta: Some(0.8),
            ..two()
        })
        .unwrap();
        let strict = power_t_test_with(&PowerTTestSpec {
            n: Some(20.0),
            delta: Some(0.8),
            strict: true,
            ..two()
        })
        .unwrap();
        // the option is not cosmetic: the lower tail is added
        assert!(strict.power > loose.power && strict.strict);
        let neg = power_t_test_with(&PowerTTestSpec {
            n: Some(20.0),
            delta: Some(-0.8),
            ..two()
        })
        .unwrap();
        check
            .scalar("t_twosample_twosided_power_strict", strict.power, POWER_TOL)
            .scalar("t_negdelta_power", neg.power, POWER_TOL)
            .scalar("t_negdelta_delta", neg.delta, Tol::exact())
            .finish();
    }

    #[test]
    fn test_validate_power_t_solve_n() {
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        for (ty, ty_name) in TYPES {
            for (alt, alt_name) in ALTS {
                let r = power_t_test_with(&PowerTTestSpec {
                    delta: Some(0.8),
                    power: Some(0.9),
                    ..t_spec(ty, alt)
                })
                .unwrap();
                assert_eq!(r.solved_for, PowerTParam::N);
                assert!(r.root.unwrap().converged);
                check.scalar(&format!("t_{ty_name}_{alt_name}_n"), r.n, ROOT_TOL);
            }
        }
        let defaults = power_t_test_with(&PowerTTestSpec {
            delta: Some(0.5),
            power: Some(0.8),
            tol: 1e-12,
            ..Default::default()
        })
        .unwrap();
        let strict = power_t_test_with(&PowerTTestSpec {
            delta: Some(0.8),
            power: Some(0.9),
            strict: true,
            ..two()
        })
        .unwrap();
        // the root lies below the interval's lower end 2: the bracket is extended downwards
        let extend = power_t_test_with(&PowerTTestSpec {
            delta: Some(10.0),
            sd: Some(1.0),
            power: Some(0.8),
            tol: 1e-12,
            ..Default::default()
        })
        .unwrap();
        let info = extend.root.unwrap();
        assert!(info.extension_steps > 0 && info.lower < 2.0, "{info:?}");
        check
            .scalar("t_d05_p08_n", defaults.n, ROOT_TOL)
            .scalar("t_twosample_twosided_n_strict", strict.n, ROOT_TOL)
            .scalar("t_extend_n", extend.n, ROOT_TOL)
            .finish();
    }

    #[test]
    fn test_validate_power_t_solve_delta() {
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        for (ty, ty_name) in TYPES {
            for (alt, alt_name) in ALTS {
                let r = power_t_test_with(&PowerTTestSpec {
                    n: Some(20.0),
                    power: Some(0.9),
                    ..t_spec(ty, alt)
                })
                .unwrap();
                assert_eq!(r.solved_for, PowerTParam::Delta);
                check.scalar(&format!("t_{ty_name}_{alt_name}_delta"), r.delta, ROOT_TOL);
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_power_t_solve_sd() {
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        for (ty, ty_name) in TYPES {
            for (alt, alt_name) in ALTS {
                let spec = PowerTTestSpec {
                    n: Some(20.0),
                    delta: Some(0.8),
                    power: Some(0.9),
                    sd: None,
                    ..t_spec(ty, alt)
                };
                let r = power_t_test_with(&spec).unwrap();
                assert_eq!(r.solved_for, PowerTParam::Sd);
                let named = power_t_test_with(&PowerTTestSpec {
                    solve_for: Some(PowerTParam::Sd),
                    ..spec
                })
                .unwrap();
                assert_eq!(named.sd, r.sd);
                check.scalar(&format!("t_{ty_name}_{alt_name}_sd"), r.sd, ROOT_TOL);
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_power_t_solve_sig() {
        // R's own pnt warned during the one-sided searches, at intermediate values; p2a reports
        // the returned value only, which is precise.
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        for (ty, ty_name) in TYPES {
            for (alt, alt_name) in ALTS {
                let r = power_t_test_with(&PowerTTestSpec {
                    n: Some(20.0),
                    delta: Some(0.8),
                    power: Some(0.9),
                    sig_level: None,
                    ..t_spec(ty, alt)
                })
                .unwrap();
                assert_eq!(r.solved_for, PowerTParam::SigLevel);
                assert!(r.warnings.is_empty(), "{:?}", r.warnings);
                check.scalar(
                    &format!("t_{ty_name}_{alt_name}_sig"),
                    r.sig_level,
                    ROOT_TOL,
                );
            }
        }
        check.finish();
    }

    #[test]
    fn test_validate_power_prop_solve() {
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        let p = |alternative| PowerPropTestSpec {
            alternative,
            tol: 1e-12,
            ..Default::default()
        };
        for (alt, name) in ALTS {
            let base = p(alt);
            let runs = [
                (
                    "power",
                    PowerPropTestSpec {
                        n: Some(50.0),
                        p1: Some(0.5),
                        p2: Some(0.75),
                        ..base.clone()
                    },
                ),
                (
                    "n",
                    PowerPropTestSpec {
                        p1: Some(0.5),
                        p2: Some(0.75),
                        power: Some(0.9),
                        ..base.clone()
                    },
                ),
                (
                    "p1",
                    PowerPropTestSpec {
                        n: Some(50.0),
                        p2: Some(0.75),
                        power: Some(0.9),
                        ..base.clone()
                    },
                ),
                (
                    "p2",
                    PowerPropTestSpec {
                        n: Some(50.0),
                        p1: Some(0.5),
                        power: Some(0.9),
                        ..base.clone()
                    },
                ),
                (
                    "sig",
                    PowerPropTestSpec {
                        n: Some(50.0),
                        p1: Some(0.5),
                        p2: Some(0.75),
                        power: Some(0.9),
                        ..base.clone()
                    },
                ),
            ];
            for (what, spec) in runs {
                let r = power_prop_test_with(&spec).unwrap();
                assert!(r.warnings.is_empty(), "{what}: {:?}", r.warnings);
                let (value, tol) = match what {
                    "power" => (r.power, POWER_TOL),
                    "n" => (r.n, ROOT_TOL),
                    "p1" => (r.p1, ROOT_TOL),
                    "p2" => (r.p2, ROOT_TOL),
                    _ => (r.sig_level, ROOT_TOL),
                };
                check.scalar(&format!("prop_{name}_{what}"), value, tol);
            }
        }
        let strict = power_prop_test_with(&PowerPropTestSpec {
            n: Some(50.0),
            p1: Some(0.5),
            p2: Some(0.75),
            strict: true,
            ..p(PowerAlternative::TwoSided)
        })
        .unwrap();
        let n = power_prop_test_with(&PowerPropTestSpec {
            p1: Some(0.3),
            p2: Some(0.5),
            power: Some(0.8),
            ..p(PowerAlternative::TwoSided)
        })
        .unwrap();
        check
            .scalar("prop_twosided_power_strict", strict.power, POWER_TOL)
            .scalar("prop_p03_p05_n", n.n, ROOT_TOL)
            .finish();
    }

    #[test]
    fn test_validate_power_anova_solve() {
        let case = Case::load("stat_power_solve");
        let mut check = Check::new(&case);
        let given = PowerAnovaTestSpec {
            groups: Some(4.0),
            n: Some(5.0),
            between_var: Some(1.0),
            within_var: Some(3.0),
            tol: 1e-12,
            ..Default::default()
        };
        let power = power_anova_test_with(&given).unwrap();
        check.scalar("anova_power", power.power, POWER_TOL);
        assert!((power.effect_size_f - (0.75f64 / 3.0).sqrt()).abs() < 1e-15);
        for (tol, suffix, key_tol) in [(1e-12, "", ROOT_TOL), (R_UNIROOT_TOL, "_pkg", PKG_TOL)] {
            let solve = |which: PowerAnovaParam| {
                let mut spec = PowerAnovaTestSpec {
                    power: Some(0.8),
                    tol,
                    ..given.clone()
                };
                match which {
                    PowerAnovaParam::N => spec.n = None,
                    PowerAnovaParam::Groups => spec.groups = None,
                    PowerAnovaParam::BetweenVar => spec.between_var = None,
                    PowerAnovaParam::WithinVar => spec.within_var = None,
                    _ => spec.sig_level = None,
                }
                if which != PowerAnovaParam::SigLevel {
                    spec.sig_level = Some(0.05);
                }
                let r = power_anova_test_with(&spec).unwrap();
                assert_eq!(r.solved_for, which);
                r
            };
            let n = solve(PowerAnovaParam::N).n;
            let groups = solve(PowerAnovaParam::Groups).groups;
            let between = solve(PowerAnovaParam::BetweenVar).between_var;
            let within = solve(PowerAnovaParam::WithinVar).within_var;
            let sig = solve(PowerAnovaParam::SigLevel).sig_level;
            for (key, value) in [
                ("n", n),
                ("groups", groups),
                ("between", between),
                ("within", within),
                ("sig", sig),
            ] {
                check.scalar(&format!("anova_{key}{suffix}"), value, key_tol);
                if suffix == "_pkg" {
                    check.scalar(&format!("anova_{key}"), value, PKG_EXACT_TOL);
                }
            }
        }
        let g3 = PowerAnovaTestSpec {
            groups: Some(3.0),
            between_var: Some(0.25),
            within_var: Some(1.0),
            tol: 1e-12,
            ..Default::default()
        };
        let p = power_anova_test_with(&PowerAnovaTestSpec {
            n: Some(20.0),
            ..g3.clone()
        })
        .unwrap();
        let n = power_anova_test_with(&PowerAnovaTestSpec {
            power: Some(0.8),
            ..g3
        })
        .unwrap();
        check
            .scalar("anova_g3_n20_power", p.power, POWER_TOL)
            .scalar("anova_g3_n", n.n, ROOT_TOL)
            .finish();
    }

    // =========================================================================
    // The inference rule, validation, options and display
    // =========================================================================

    fn message<T: std::fmt::Debug>(r: EconResult<T>) -> String {
        r.unwrap_err().to_string()
    }

    #[test]
    fn test_power_infers_unknown_and_applies_defaults() {
        let (ts, two_sided) = (TTestType::TwoSample, PowerAlternative::TwoSided);
        // MCP's {n: 30, delta: 0.5}
        let r = power_t_test(Some(30.0), Some(0.5), None, None, None, ts, two_sided).unwrap();
        assert_eq!(
            (r.sd, r.sig_level, r.solved_for),
            (1.0, 0.05, PowerTParam::Power)
        );
        let r = power_t_test(
            Some(20.0),
            Some(0.8),
            None,
            Some(0.05),
            Some(0.9),
            ts,
            two_sided,
        )
        .unwrap();
        assert_eq!(r.solved_for, PowerTParam::Sd);
        let e = message(power_t_test(
            None,
            None,
            None,
            None,
            Some(0.8),
            ts,
            two_sided,
        ));
        assert!(e.contains("omitted: n, delta"), "{e}");
        let e = message(power_t_test(
            Some(20.0),
            Some(0.8),
            Some(1.0),
            Some(0.05),
            Some(0.9),
            ts,
            two_sided,
        ));
        assert!(e.contains("nothing is omitted"), "{e}");
        let e = message(power_t_test_with(&PowerTTestSpec {
            n: Some(20.0),
            delta: Some(0.8),
            sd: Some(1.2),
            power: Some(0.9),
            solve_for: Some(PowerTParam::Sd),
            ..Default::default()
        }));
        assert!(e.contains("solve_for = sd"), "{e}");
        let e = message(power_t_test_with(&PowerTTestSpec {
            delta: Some(0.8),
            power: Some(0.9),
            solve_for: Some(PowerTParam::Sd),
            ..Default::default()
        }));
        assert!(e.contains("n is required when solving for sd"), "{e}");
        let r = power_prop_test(Some(50.0), 0.5, None, None, Some(0.9), two_sided).unwrap();
        assert_eq!(r.solved_for, PowerPropParam::P2);
        let r = power_anova_test(4, Some(5.0), 1.0, 3.0, None, Some(0.8)).unwrap();
        assert_eq!(r.solved_for, PowerAnovaParam::SigLevel);
    }

    #[test]
    fn test_power_rejects_invalid_parameters() {
        let (ts, two_sided) = (TTestType::TwoSample, PowerAlternative::TwoSided);
        let has =
            |e: String, what: &str| assert!(e.contains(what), "{e:?} should contain {what:?}");
        has(
            message(power_anova_test(3, Some(20.0), 0.25, 0.0, None, None)),
            "within_var",
        );
        has(
            message(power_anova_test(1, Some(20.0), 0.25, 1.0, None, None)),
            "groups",
        );
        has(
            message(power_anova_test(3, Some(1.5), 0.25, 1.0, None, None)),
            "n must be at least 2",
        );
        has(
            message(power_t_test(
                Some(1.0),
                Some(0.5),
                None,
                None,
                None,
                ts,
                two_sided,
            )),
            "n must be greater than 1",
        );
        has(
            message(power_t_test(
                Some(20.0),
                Some(0.5),
                Some(0.0),
                None,
                None,
                ts,
                two_sided,
            )),
            "sd",
        );
        has(
            message(power_t_test(
                Some(20.0),
                Some(0.5),
                None,
                Some(1.5),
                None,
                ts,
                two_sided,
            )),
            "sig_level",
        );
        has(
            message(power_t_test(
                None,
                Some(0.5),
                None,
                None,
                Some(1.0),
                ts,
                two_sided,
            )),
            "power",
        );
        has(
            message(power_t_test_with(&PowerTTestSpec {
                n: Some(20.0),
                delta: Some(0.5),
                tol: 0.0,
                ..Default::default()
            })),
            "tol",
        );
        has(
            message(power_prop_test(
                Some(50.0),
                1.5,
                Some(0.5),
                None,
                None,
                two_sided,
            )),
            "p1",
        );
        has(
            message(power_prop_test(
                Some(50.0),
                1.0,
                Some(1.0),
                None,
                None,
                two_sided,
            )),
            "both",
        );
        // the target is unreachable on R's interval [2, 1e5]
        has(
            message(power_anova_test(3, None, 1e-9, 1.0, None, Some(0.8))),
            "cannot solve for n",
        );
        // only sd is absent, so it is solved: R's interval delta * [1e-7, 1e7] is empty
        has(
            message(power_t_test_with(&PowerTTestSpec {
                n: Some(20.0),
                delta: Some(-0.8),
                sig_level: Some(0.05),
                power: Some(0.9),
                alternative: PowerAlternative::OneSided,
                ..Default::default()
            })),
            "delta > 0",
        );
    }

    #[test]
    fn test_power_option_enums_parse_aliases_and_reject_unknown() {
        TTestType::choices().validate().unwrap();
        PowerAlternative::choices().validate().unwrap();
        PowerTParam::choices().validate().unwrap();
        PowerPropParam::choices().validate().unwrap();
        PowerAnovaParam::choices().validate().unwrap();
        for s in ["two.sample", "two_sample", "Two-Sample", "twosample", "two"] {
            assert_eq!(s.parse::<TTestType>().unwrap(), TTestType::TwoSample, "{s}");
        }
        assert_eq!(
            "one.sample".parse::<TTestType>().unwrap(),
            TTestType::OneSample
        );
        for s in ["one.sided", "ONE_SIDED"] {
            assert_eq!(
                s.parse::<PowerAlternative>().unwrap(),
                PowerAlternative::OneSided,
                "{s}"
            );
        }
        for s in ["sig.level", "sig_level", "alpha"] {
            assert_eq!(
                s.parse::<PowerTParam>().unwrap(),
                PowerTParam::SigLevel,
                "{s}"
            );
        }
        assert_eq!(
            "between_var".parse::<PowerAnovaParam>().unwrap(),
            PowerAnovaParam::BetweenVar
        );
        let e = "one-tailed"
            .parse::<PowerAlternative>()
            .unwrap_err()
            .to_string();
        assert!(e.contains("two.sided") && e.contains("one.sided"), "{e}");
        let ok = run_power_t_test(
            Some(20.0),
            Some(0.8),
            None,
            None,
            None,
            "one-sample",
            "two.sided",
        );
        assert!(ok.is_ok(), "{ok:?}");
        assert!(
            run_power_t_test(
                Some(20.0),
                Some(0.8),
                None,
                None,
                None,
                "welch",
                "two.sided"
            )
            .is_err()
        );
    }

    #[test]
    fn test_power_display_ends_with_notes() {
        let (two_sided, ts) = (PowerAlternative::TwoSided, TTestType::TwoSample);
        let text = format!(
            "{}",
            power_t_test(Some(20.0), Some(0.8), Some(1.2), None, None, ts, two_sided).unwrap()
        );
        for part in [
            "Two-sample t test power calculation",
            "          power = 0.5377573\n",
            "      sig.level = 0.05\n",
            "    alternative = two.sided\n",
            "NOTE: n is number in *each* group",
        ] {
            assert!(text.contains(part), "{text:?} should contain {part:?}");
        }
        let paired = power_t_test(
            Some(20.0),
            Some(0.8),
            None,
            None,
            None,
            TTestType::Paired,
            two_sided,
        )
        .unwrap();
        assert_eq!(
            paired.note.as_deref(),
            Some("n is number of *pairs*, sd is std.dev. of *differences* within pairs")
        );
        assert_eq!(format7(1e7), "1e+07");
        assert_eq!(format7(63.765763714), "63.76576");
        assert_eq!(format7(1e-5), "1e-05");
        assert_eq!(format7(20.0), "20");
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

    /// A two-sided R value computed with `strict = TRUE`.
    fn strict_t(n: f64, delta: f64, sd: f64, test_type: TTestType) -> PowerTTestResult {
        power_t_test_with(&PowerTTestSpec {
            n: Some(n),
            delta: Some(delta),
            sd: Some(sd),
            sig_level: Some(0.05),
            test_type,
            strict: true,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn test_validate_power_t_test_ten_million_per_group() {
        // R: power.t.test(n = 1e7, delta = 0.001, sd = 1, strict = TRUE)$power
        let r = strict_t(1e7, 0.001, 1.0, TTestType::TwoSample);
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
        let r = strict_t(1e7, 0.001, 1.0, TTestType::OneSample);
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
        let r = power_t_test_with(&PowerTTestSpec {
            n: Some(1e7),
            sd: Some(1.0),
            sig_level: Some(0.05),
            power: Some(0.8),
            strict: true,
            tol: 1e-14,
            ..Default::default()
        })
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

    #[test]
    fn test_validate_power_anova_ten_million_per_group_is_full_power() {
        // R: power.anova.test(groups = 3, n = n, between.var = 0.25, within.var = 1)$power == 1
        for n in [1e7, 1e12, 1e300] {
            let r = power_anova_test(3, Some(n), 0.25, 1.0, Some(0.05), None).unwrap();
            assert_eq!(r.power, 1.0, "n = {n}");
        }
    }

    #[test]
    fn test_validate_power_anova_many_groups_solve_n() {
        // R: power.anova.test(groups = 50, between.var = 0.001, within.var = 1,
        // power = 0.8)$n is 611.92663596456464 (default tol)
        let r = power_anova_test(50, None, 0.001, 1.0, Some(0.05), Some(0.8)).unwrap();
        assert!(
            (r.n - 611.926_635_964_564_6).abs() < 2.0 * R_UNIROOT_TOL,
            "{}",
            r.n
        );
    }

    #[test]
    fn test_validate_power_anova_astronomical_group_count() {
        // R: power.anova.test(groups = 2^62, n = 10, between.var = 1, within.var = 1)$power
        // is 1; solving for n fails in R as here: the power is 1 on the whole interval
        let r = power_anova_test(1usize << 62, Some(10.0), 1.0, 1.0, Some(0.05), None).unwrap();
        assert_eq!(r.power, 1.0);
        let e = message(power_anova_test(
            1usize << 62,
            None,
            1.0,
            1.0,
            Some(0.05),
            Some(0.8),
        ));
        assert!(e.contains("cannot solve for n"), "{e}");
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
        // Solving for delta keeps Tier 0's limit: 10 * sd must stay finite
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
        // R: power.t.test(n = 1e300, delta = d, sd = 1e-300, sig.level = s,
        // alternative = a, strict = TRUE)$power == 1 for d = 1 and -1 two-sided
        // and d = 1 one-sided, at s = 0.05 and at the smallest level accepted,
        // just above 2^-53 (s = 2^-53 * (1 + 2^-52)). The non-centrality,
        // sqrt(n / 2) delta / sd, is beyond f64::MAX, and the power is its
        // limit (the lower tail adds nothing there, so strict does not matter).
        // The one-sided delta = -1 case is left out: R gives 0 there (it tests
        // the upper tail), while power_t_test tests the lower tail and gives
        // 1; see test_power_t_test_one_sided_negative_delta_tests_the_lower_tail.
        let cases = [
            (1.0, PowerAlternative::TwoSided),
            (-1.0, PowerAlternative::TwoSided),
            (1.0, PowerAlternative::OneSided),
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
    fn test_power_t_test_one_sided_negative_delta_tests_the_lower_tail() {
        // power_t_test (the Tier 0 API) runs a one-sided test in the direction
        // of delta: the lower tail for delta < 0, so a negative delta has the
        // power of its absolute value (registry deviation STAT-61).
        // power_t_test_with follows R, which tests the upper tail whatever the
        // sign of delta, and gives 0 at an infinite non-centrality
        // (power.t.test(n = 1e300, delta = -1, sd = 1e-300,
        // alternative = "one.sided", strict = TRUE)$power) and 1.0e-6 at
        // n = 20, delta = -1.
        let one_sided = |n: f64, delta: f64, sd: f64, sig_level: f64| {
            power_t_test(
                Some(n),
                Some(delta),
                Some(sd),
                Some(sig_level),
                None,
                TTestType::TwoSample,
                PowerAlternative::OneSided,
            )
            .unwrap()
        };
        for sig_level in [MIN_SIG_LEVEL.next_up(), 0.05] {
            let r = one_sided(1e300, -1.0, 1e-300, sig_level);
            assert_eq!(r.power, 1.0, "sig {sig_level}");
            assert_eq!(r.delta, -1.0, "the given delta is reported");
        }
        // deviation STAT-61: the Tier 0 convention, not compared with R
        for (n, sd) in [(20.0, 1.0), (50.0, 2.5), (200.0, 3.0)] {
            let lower = one_sided(n, -1.0, sd, 0.05).power;
            let upper = one_sided(n, 1.0, sd, 0.05).power;
            assert_eq!(lower, upper, "n = {n}, sd = {sd}");
            assert!(lower > 0.05 && lower < 1.0, "n = {n}, sd = {sd}: {lower}");
        }
        let r_convention = |n: f64, sd: f64| {
            power_t_test_with(&PowerTTestSpec {
                n: Some(n),
                delta: Some(-1.0),
                sd: Some(sd),
                sig_level: Some(0.05),
                alternative: PowerAlternative::OneSided,
                ..Default::default()
            })
            .unwrap()
            .power
        };
        assert_eq!(r_convention(1e300, 1e-300), 0.0);
        let p = r_convention(20.0, 1.0);
        assert!(p > 0.0 && p < 1e-5, "{p}");
    }

    #[test]
    fn test_validate_power_t_test_non_centrality_beside_an_underflowing_standard_error() {
        // The standard error sd sqrt(2 / n) underflows to 0 at sd = 1e-174 and
        // n = 1e300, but the non-centrality sqrt(n / 2) delta / sd is 3.5:
        // dividing delta by the underflowed standard error used to give an
        // infinite non-centrality and a power of 1.
        // R: power.t.test(n = 1e300, delta = 5e-324, sd = 1e-174,
        // type = "two.sample", strict = TRUE)$power is 0.9374369283402233, and
        // with type = "one.sample" or type = "paired" 0.99856201278583712.
        let cases = [
            (TTestType::TwoSample, 0.937_436_928_340_223_3),
            (TTestType::OneSample, 0.998_562_012_785_837_1),
            (TTestType::Paired, 0.998_562_012_785_837_1),
        ];
        for (test_type, expected) in cases {
            let r = strict_t(1e300, 5e-324, 1e-174, test_type);
            assert!(
                (r.power - expected).abs() < 1e-6,
                "{test_type:?}: {}",
                r.power
            );
        }
    }

    #[test]
    fn test_power_functions_refuse_a_sig_level_at_or_below_2_pow_minus_53() {
        // At or below 2^-53 the confidence level 1 - sig_level rounds to 1 or
        // to 1 - 2^-53, and every power function refuses the level (Tier 0's
        // floor; R: power.prop.test(n = 1e300, p1 = 0.3, p2 = 0.5,
        // sig.level = 1e-300)$power is 1). Every path is checked: compute
        // power, solve for n and, for the t test, solve for delta.
        let limit = "between 1.1102230246251565e-16 (2^-53) and 1 (exclusive)";
        let two = PowerAlternative::TwoSided;
        let one = PowerAlternative::OneSided;
        // 0 and below, inside (0, 2^-53], and 2^-53 itself
        for sig in [-0.1, 0.0, 1e-300, 1e-17, MIN_SIG_LEVEL] {
            for alternative in [two, one] {
                let t = power_t_test(
                    Some(1e6),
                    Some(1.0),
                    Some(1.0),
                    Some(sig),
                    None,
                    TTestType::TwoSample,
                    alternative,
                );
                assert_refused_containing(t, "sig_level", limit);
                let t_n = power_t_test(
                    None,
                    Some(1.0),
                    Some(1.0),
                    Some(sig),
                    Some(0.8),
                    TTestType::OneSample,
                    alternative,
                );
                assert_refused_containing(t_n, "sig_level", limit);
                let t_delta = power_t_test(
                    Some(50.0),
                    None,
                    Some(1.0),
                    Some(sig),
                    Some(0.8),
                    TTestType::TwoSample,
                    alternative,
                );
                assert_refused_containing(t_delta, "sig_level", limit);
                let prop =
                    power_prop_test(Some(1e300), 0.3, Some(0.5), Some(sig), None, alternative);
                assert_refused_containing(prop, "sig_level", limit);
                let prop_n =
                    power_prop_test(None, 0.3, Some(0.5), Some(sig), Some(0.8), alternative);
                assert_refused_containing(prop_n, "sig_level", limit);
            }
            let anova = power_anova_test(3, Some(1e6), 0.25, 1.0, Some(sig), None);
            assert_refused_containing(anova, "sig_level", limit);
            let anova_n = power_anova_test(3, None, 0.25, 1.0, Some(sig), Some(0.8));
            assert_refused_containing(anova_n, "sig_level", limit);
        }
        // The message gives the value and the reason
        assert_refused_containing(
            power_anova_test(3, Some(20.0), 0.25, 1.0, Some(0.0), None),
            "sig_level",
            "got 0.0: every test uses this floor, since at or below 2^-53 the confidence \
             level 1 - sig_level rounds to 1 or to 1 - 2^-53 in double precision",
        );
        // The reason describes levels from 0 up to the floor only: a negative
        // level, a level of 1 or more, and NaN get the limits alone
        for sig in [-0.1, 1.0, 1.5, f64::NAN] {
            let refusals = [
                power_anova_test(3, Some(20.0), 0.25, 1.0, Some(sig), None)
                    .map(|_| ())
                    .unwrap_err(),
                power_t_test(
                    Some(20.0),
                    Some(1.0),
                    Some(1.0),
                    Some(sig),
                    None,
                    TTestType::TwoSample,
                    one,
                )
                .map(|_| ())
                .unwrap_err(),
            ];
            for err in refusals {
                match err {
                    EconError::InvalidSpecification { message } => {
                        assert!(
                            message.starts_with("sig_level must") && message.contains(limit),
                            "sig {sig}: {message:?}"
                        );
                        assert!(
                            !message.contains("this floor"),
                            "sig {sig}: {message:?} should not explain the floor"
                        );
                    }
                    other => panic!("sig {sig}: expected a sig_level refusal, got {other:?}"),
                }
            }
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
        assert!((anova.power - 1.0).abs() < 1e-12, "anova: {}", anova.power);
    }

    #[test]
    fn test_validate_power_t_test_solve_delta_scales_with_sd() {
        // The solved delta is proportional to sd. R: power.t.test(n = n, sd = 1,
        // power = 0.99, tol = 1e-12)$delta is 1.1256302563108442 (two-sample,
        // n = 30) and 23.214470556951962 (one-sample, n = 2).
        let solve = |n: f64, test_type, sd: f64| {
            power_t_test_with(&PowerTTestSpec {
                n: Some(n),
                sd: Some(sd),
                sig_level: Some(0.05),
                power: Some(0.99),
                test_type,
                tol: 1e-12,
                ..Default::default()
            })
            .map(|r| r.delta)
        };
        let cases = [
            (
                30.0,
                TTestType::TwoSample,
                1.125_630_256_310_844_2,
                f64::MAX / 10.0,
            ),
            // 23.2 * f64::MAX / 10 is not finite: the largest sd here is f64::MAX / 100
            (
                2.0,
                TTestType::OneSample,
                23.214_470_556_951_962,
                f64::MAX / 100.0,
            ),
        ];
        for (n, test_type, r, sd_max) in cases {
            let unit = solve(n, test_type, 1.0).unwrap();
            assert!(
                ((unit - r) / r).abs() < 1e-9,
                "{test_type:?}: {unit} vs R {r}"
            );
            let scaled = solve(n, test_type, sd_max).unwrap() / sd_max;
            assert!(
                ((scaled - unit) / unit).abs() < 1e-6,
                "{test_type:?}: delta / sd is {scaled} at sd = {sd_max:e}, {unit} at sd = 1"
            );
        }
        // a root beyond f64::MAX is an error, not an overflowed value
        let e = message(solve(2.0, TTestType::OneSample, f64::MAX / 10.0));
        assert!(e.contains("cannot solve for delta"), "{e}");
    }

    #[test]
    fn test_validate_power_prop_test_equal_proportions_at_the_smallest_f64() {
        // With p1 == p2 the power is the significance level (R:
        // power.prop.test(n = 100, p1 = 0.3, p2 = 0.3, strict = TRUE)$power is
        // 0.05). At p = 5e-324 the standard error underflows to 0, and the
        // non-centrality 0 / 0 used to fall back to a power of 1.
        for p in [0.3, 5e-324] {
            let r = power_prop_test_with(&PowerPropTestSpec {
                n: Some(100.0),
                p1: Some(p),
                p2: Some(p),
                sig_level: Some(0.05),
                strict: true,
                ..Default::default()
            })
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
            let at_limit = anova(limit);
            assert!(
                at_limit.as_ref().is_ok_and(|r| r.power == 1.0),
                "groups = {groups}: {limit:e} gives {at_limit:?}"
            );
            assert_refused(anova(limit.next_up()), "n");
        }
    }

    #[test]
    fn test_validate_power_t_test_zero_effect_at_huge_n() {
        // With delta = 0 the power is the significance level
        // (R: power.t.test(n = 1e7, delta = 0, strict = TRUE)$power == 0.05).
        for n in [1e7, 1e15, 1e300] {
            let r = strict_t(n, 0.0, 1.0, TTestType::TwoSample);
            assert!((r.power - 0.05).abs() < 1e-9, "n = {n}: {}", r.power);
        }
        // R: power.t.test(n = 1e300, delta = 0, sd = 1e-300, strict = TRUE)$power
        // == 0.05. The standard error, sd * sqrt(2 / n), would underflow to 0
        // here, and 0 / 0 must not turn into a non-centrality of NaN.
        let r = strict_t(1e300, 0.0, 1e-300, TTestType::TwoSample);
        assert!((r.power - 0.05).abs() < 1e-9, "sd = 1e-300: {}", r.power);
    }

    // =========================================================================
    // The critical values against R (Tier 0's quantile pins, moved onto the
    // shared helpers: t_quantile(p, df) and the F upper quantile)
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
                t_quantile(p, df).unwrap(),
                r,
                1e-12,
                &format!("qt({p}, {df})"),
            );
        }
        // the far tail the power functions ask for: R qt(2^-53, 1001, lower.tail = FALSE)
        assert_rel(
            -t_quantile(f64::EPSILON / 2.0, 1001.0).unwrap(),
            8.351_823_973_969_251,
            2e-10,
            "df 1001",
        );
    }

    #[test]
    fn test_validate_f_upper_quantile() {
        // R: qf(alpha, df1, df2, lower.tail = FALSE) at alpha = 1 - p (exact for p >= 0.5):
        // Tier 0's cases with a representable upper tail, and Tier 0's tolerances. The values
        // are R's qf, which power.anova.test calls, not the root of pf that Tier 0 pinned: past
        // df = 4e5 R's qf takes the chi-square limit (qf(0.95, 94, 1e6) is 1.2514005 where the
        // root of pf is 1.2514166), and p2a takes the same limit.
        let cases = [
            (2.0, 57.0, 0.95, 3.158_842_719_260_644_7, 1e-12),
            (1e3, 1e3, 0.95, 1.109_688_290_242_986_8, 1e-12),
            (1e3, 1.0, 1.0 - 1e-6, 636_301_542_061.703_1, 1e-12),
            (2.0, 1.0, 1.0 - 1e-6, 499_999_999_970.744_26, 1e-12),
            (2.0, 1e4, 0.95, 2.996_629_893_998_736_8, 1e-11),
            (1e4, 1e4, 0.95, 1.033_445_780_480_156_5, 1e-11),
            (1.0, 1e5, 0.95, 3.841_551_813_699_162_2, 1e-9),
            (2.0, 3e5, 0.95, 2.995_762_188_451_323, 1e-9),
            (94.0, 1e6, 0.95, 1.251_400_544_067_505_6, 1e-9),
            (1e4, 1e6, 0.999, 1.044_273_056_541_017_9, 1e-9),
            (2.0, 3e7, 0.95, 2.995_732_273_553_989_1, 1e-9),
            (1.0, 1e15, 0.95, 3.841_458_820_694_125_8, 1e-9),
            (1e3, 1e7, 0.999, 1.143_917_092_619_679_1, 1e-7),
            (1e8, 10.0, 0.95, 2.537_878_382_972_047_9, 1e-8),
            (3e4, 1e6, 0.95, 1.013_467_967_648_743, 1e-8),
            (1e5, 1e7, 0.999, 1.013_876_955_325_294_6, 1e-6),
            (9e5, 2e6, 0.95, 1.002_453_265_730_635_8, 1e-8),
            (1e6, 1e9, 0.5, 0.999_999_333_333_412_29, 1e-12),
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
            assert_rel(f_upper_quantile(1.0 - p, df1, df2).unwrap(), r, tol, &what);
        }
        // alpha outside (0, 1) is NaN, as R's qf
        assert!(f_upper_quantile(0.0, 3.0, 30.0).unwrap().is_nan());
        assert!(f_upper_quantile(f64::NAN, 3.0, 30.0).unwrap().is_nan());
    }

    #[test]
    fn test_validate_quantiles_upper_tail_at_2_pow_minus_52() {
        // R: qt(2^-52, df, lower.tail = FALSE) and qf(2^-52, df1, df2, lower.tail = FALSE)
        let tail = f64::EPSILON;
        let t_cases = [
            (1.0, 1_433_540_284_805_664.8),
            (3.0, 170_608.307_621_525_75),
            (1000.0, 8.264_015_607_871_633),
        ];
        for (df, r) in t_cases {
            let upper = t_quantile(1.0 - tail, df).unwrap();
            assert_rel(upper, r, 1e-12, &format!("qt(1 - 2^-52, {df})"));
            assert_eq!(upper, -t_quantile(tail, df).unwrap(), "df = {df}");
        }
        let f_cases = [
            (2.0, 57.0, 72.446_803_480_605_54),
            (4.0, 245.0, 23.370_948_266_418_868),
            (1e3, 1e3, 1.676_802_747_107_059_7),
        ];
        for (df1, df2, r) in f_cases {
            let what = format!("qf(2^-52, {df1}, {df2}, lower.tail = FALSE)");
            assert_rel(f_upper_quantile(tail, df1, df2).unwrap(), r, 1e-12, &what);
        }
    }

    #[test]
    fn test_validate_quantiles_far_tail_at_small_df() {
        // R: qt(tail, df, lower.tail = FALSE), taken at the exact tail
        let t_cases = [
            (1.0, 1e-10, 3_183_098_861.837_906_4),
            (1.5, 1e-12, 52_194_694.273_446_366),
            (1.75, 1e-14, 61_668_983.346_367_51),
            (1.05, 1e-6, 175_852.477_391_897_6),
        ];
        for (df, tail, r) in t_cases {
            let what = format!("qt({tail:e}, {df}, lower.tail = FALSE)");
            assert_rel(-t_quantile(tail, df).unwrap(), r, 1e-12, &what);
        }
        // R: qt(1 - tail, df): 1 - tail rounds, and both work with the rounded tail
        let t_upper = [
            (1.0, 1.0 - 1e-10, 3_183_098_598.467_148),
            (1.5, 1.0 - 1e-12, 52_195_464.045_249_306),
        ];
        for (df, p, r) in t_upper {
            assert_rel(
                t_quantile(p, df).unwrap(),
                r,
                1e-12,
                &format!("qt({p}, {df})"),
            );
        }
        // R: qf(1 - tail, df1, df2); 1 - (1 - tail) is exact
        let f_upper = [
            (1.0, 2.0, 1.0 - 1e-12, 1_000_022_122_208.002_9),
            (3.0, 4.0, 1.0 - 1e-13, 5_772_603.724_181_895),
        ];
        for (df1, df2, p, r) in f_upper {
            let what = format!("qf({p}, {df1}, {df2})");
            assert_rel(
                f_upper_quantile(1.0 - p, df1, df2).unwrap(),
                r,
                1e-12,
                &what,
            );
        }
        // R: qf(tail, df1, df2, lower.tail = FALSE) at the exact tails
        let f_exact = [
            (1.0, 2.0, 1e-12, 999_999_999_998.5),
            (3.0, 4.0, 1e-13, 5_773_501.136_340_68),
        ];
        for (df1, df2, tail, r) in f_exact {
            let what = format!("qf({tail:e}, {df1}, {df2}, lower.tail = FALSE)");
            assert_rel(f_upper_quantile(tail, df1, df2).unwrap(), r, 1e-12, &what);
        }
    }

    #[test]
    fn test_f_upper_quantile_is_monotone_and_defined_at_any_df() {
        // Every branch, including astronomically large degrees of freedom,
        // returns an ordered, non-NaN, non-negative quantile for every tail
        // (Inf where the quantile is beyond f64::MAX, as R's qf).
        let dfs = [1.0, 3.0, 57.0, 1e3, 1e4, 3e4, 1e6, 1e7, 1e12, 4.6e18, 1e300];
        let tails = [
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
        for &df1 in &dfs {
            for &df2 in &dfs {
                let f: Vec<f64> = tails
                    .iter()
                    .map(|&a| f_upper_quantile(a, df1, df2).unwrap())
                    .collect();
                assert!(
                    f.iter().all(|v| !v.is_nan() && *v >= 0.0),
                    "F({df1}, {df2}): {f:?}"
                );
                assert!(f.windows(2).all(|w| w[0] >= w[1]), "F({df1}, {df2}): {f:?}");
            }
        }
    }
}
