//! Friedman's SuperSmoother.
//!
//! A port of R's `stats::supsmu` (Friedman's Fortran `supsmu`/`smooth`, as in R's
//! `ppr.f`): a running-lines smoother that chooses its span at each x by cross-validation.
//!
//! # Algorithm
//!
//! 1. Running lines with uniform weights (not tricube) in windows of `2 ibw + 1` points,
//!    `ibw = int(0.5 span n + 0.5)` (at least 2), shifted at the ends or, with
//!    `periodic = true`, wrapped around [0, 1].
//! 2. With `span` "cv" (None or 0): smooths for the tweeter, midrange and woofer spans
//!    (0.05, 0.2, 0.5, R's single-precision constants), the exact leave-one-out absolute
//!    residuals of each smoothed with the midrange span, the span with the smallest smoothed
//!    residual at each point, moved toward the woofer by `bass` (Friedman's rule
//!    `(0.5 - s) max(1e-7, r / r_woofer)^(10 - bass)`), the chosen spans smoothed, clamped
//!    and used to interpolate between the three smooths, and a final tweeter smooth.
//! 3. Tied x values enter the windows one by one; their fits are averaged afterwards, and
//!    the result has one value per distinct x.
//!
//! `periodic = true` needs x in [0, 1], as in R. `span` outside [0, 1], `bass` outside
//! [0, 10] and negative weights are errors.
//!
//! # References
//!
//! - Friedman, J.H. (1984). A variable span smoother. *Laboratory for Computational
//!   Statistics, Stanford University Technical Report No. 5*.
//!   The original SuperSmoother paper.
//!
//! - Friedman, J.H. (1984). SMART User's Guide. *Laboratory for Computational
//!   Statistics, Stanford University Technical Report No. 1*.
//!
//! - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
//!   Statistical Learning* (2nd ed.), Section 6.2. Springer.
//!   https://hastie.su.domains/ElemStatLearn/
//!
//! - Cleveland, W.S. (1979). Robust locally weighted regression and smoothing
//!   scatterplots. *Journal of the American Statistical Association*, 74(368),
//!   829-836. https://doi.org/10.1080/01621459.1979.10481038
//!   Related LOWESS method for comparison.
//!
//! Observations with a missing (NaN) or infinite x, y or weight are deleted first (R's
//! `supsmu()` deletes them too, `is.finite(x + y + wt)`) and reported in
//! [`SupsmuResult::missing`].
//!
//! R equivalent: `stats::supsmu` (port of Friedman's Fortran as in R's `ppr.f`).

use serde::{Deserialize, Serialize};

use crate::data::columns::MissingSummary;
use crate::errors::EconResult;

/// Result of SuperSmoother fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupsmuResult {
    /// Sorted x values (with duplicates removed)
    pub x: Vec<f64>,
    /// Fitted y values (smoothed)
    pub y: Vec<f64>,
    /// The bass parameter used
    pub bass: f64,
    /// Whether periodic boundary conditions were used
    pub periodic: bool,
    /// Number of unique points
    pub n: usize,
    /// Observations dropped because x, y or the weight was missing or infinite
    /// (`by_column` names `x`, `y` and `wt`); absent from JSON when nothing was dropped.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

/// Configuration for SuperSmoother.
#[derive(Debug, Clone)]
pub struct SupsmuConfig {
    /// Span parameter: either a fixed fraction, or None for cross-validation ("cv")
    pub span: Option<f64>,
    /// Treat x as periodic on [0, 1]
    pub periodic: bool,
    /// Bass parameter (0-10): higher values produce smoother results
    pub bass: f64,
}

impl Default for SupsmuConfig {
    fn default() -> Self {
        SupsmuConfig {
            span: None, // Use cross-validation
            periodic: false,
            bass: 0.0,
        }
    }
}

/// The spans and constants of R's `ppr.f` (Fortran REAL `DATA` constants widened to double as
/// R widens them; f64 literals would change the last digits of every smooth): tweeter,
/// midrange and woofer spans, and `big`, `sml`, `eps`.
const SPANS: [f64; 3] = [0.05_f32 as f64, 0.2_f32 as f64, 0.5];
const BIG: f64 = 1.0e20_f32 as f64;
const SML: f64 = 1.0e-7_f32 as f64;
const EPS: f64 = 1.0e-3_f32 as f64;

/// Half-width of the running-lines window for `span` on `n` points: Fortran
/// `int(0.5 * span * n + 0.5)`, at least 2.
fn half_window(span: f64, n: usize) -> usize {
    ((0.5 * span * n as f64 + 0.5) as usize).max(2)
}

/// Apply Friedman's SuperSmoother to data.
///
/// A port of R's `stats::supsmu` (Friedman's Fortran, as in R's `ppr.f`). Running lines
/// with uniform weights in windows of `2 ibw + 1` points (shifted at the ends, or wrapped
/// around [0, 1] when `periodic`); with `span = None` (or `Some(0.0)`, R's `"cv"`) three
/// spans (0.05, 0.2, 0.5) are smoothed, their exact leave-one-out residuals are smoothed
/// with the midrange span, the best span at each point is chosen (moved toward the woofer
/// by `bass`), smoothed, used to interpolate between the three smooths, and the result is
/// smoothed with the tweeter span. Tied x values enter the windows one by one and their
/// fits are averaged afterwards.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values (must have same length as x)
/// * `wt` - Optional non-negative weights (default: uniform)
/// * `span` - Span fraction in [0, 1] (None or 0 for cross-validation)
/// * `periodic` - Treat x as periodic on [0, 1] (x must lie in [0, 1], as in R)
/// * `bass` - Smoothness control in [0, 10] (higher = smoother; 0 = none)
///
/// Observations with a NaN or infinite x, y or weight are deleted first; at least 4
/// must remain, with at least 4 distinct x values.
///
/// # Returns
/// A `SupsmuResult` with the unique sorted x values and the smooth at each.
///
/// # Example
/// ```
/// use p2a_core::regression::supsmu::supsmu;
///
/// let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
/// let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * rand::random::<f64>()).collect();
/// let result = supsmu(&x, &y, None, None, false, 0.0).unwrap();
/// ```
pub fn supsmu(
    x: &[f64],
    y: &[f64],
    wt: Option<&[f64]>,
    span: Option<f64>,
    periodic: bool,
    bass: f64,
) -> Result<SupsmuResult, String> {
    if x.len() != y.len() {
        return Err("x and y must have the same length".to_string());
    }

    let n_total = x.len();
    if let Some(w) = wt
        && w.len() != n_total
    {
        return Err("Weights must have same length as data".to_string());
    }
    let span = span.unwrap_or(0.0);
    if !(span.is_finite() && (0.0..=1.0).contains(&span)) {
        return Err(format!(
            "span must be between 0 and 1 (0 or None for cross-validation), got {span}"
        ));
    }
    if !(bass.is_finite() && (0.0..=10.0).contains(&bass)) {
        return Err(format!("bass must be between 0 and 10, got {bass}"));
    }

    // Delete the observations with a missing or infinite x, y or weight
    let finite_w = |i: usize| wt.is_none_or(|w| w[i].is_finite());
    let keep: Vec<usize> = (0..n_total)
        .filter(|&i| x[i].is_finite() && y[i].is_finite() && finite_w(i))
        .collect();
    let n = keep.len();
    let bad = |v: &[f64]| v.iter().filter(|a| !a.is_finite()).count();
    let mut by_column = Vec::new();
    for (name, k) in [("x", bad(x)), ("y", bad(y)), ("wt", wt.map_or(0, bad))] {
        if k > 0 {
            by_column.push((name.to_string(), k));
        }
    }
    let missing = MissingSummary {
        n_total,
        n_used: n,
        n_dropped: n_total - n,
        by_column,
    };
    if n < 4 {
        return Err(format!(
            "Need at least 4 complete (x, y, wt) observations for supsmu; {n} of {n_total} \
             are complete"
        ));
    }
    if let Some(w) = wt
        && let Some(&i) = keep.iter().find(|&&i| w[i] < 0.0)
    {
        return Err(format!(
            "weights must be non-negative; weight {i} is {}",
            w[i]
        ));
    }
    if periodic && keep.iter().any(|&i| !(0.0..=1.0).contains(&x[i])) {
        return Err("x must be between 0 and 1 for a periodic smooth (as R's supsmu)".to_string());
    }

    // Sort by (x, y), stably (R's order(x, y))
    let mut ord = keep;
    ord.sort_by(|&a, &b| x[a].total_cmp(&x[b]).then(y[a].total_cmp(&y[b])));
    let xs: Vec<f64> = ord.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = ord.iter().map(|&i| y[i]).collect();
    let ws: Vec<f64> = match wt {
        Some(w) => ord.iter().map(|&i| w[i]).collect(),
        None => vec![1.0; n],
    };
    let distinct = 1 + xs.windows(2).filter(|p| p[1] != p[0]).count();
    if distinct < 4 {
        return Err("Need at least 4 unique x values".to_string());
    }
    if periodic {
        let widest = if span > 0.0 { span } else { SPANS[2] };
        let window = 2 * half_window(widest, n) + 1;
        if window > n {
            return Err(format!(
                "a periodic smooth with span {widest} needs at least {window} observations; {n} given"
            ));
        }
    }

    let iper = if periodic { 2 } else { 1 };
    let smo = super_smoother(&xs, &ys, &ws, span, iper, bass).map_err(|e| e.to_string())?;

    // The first value of each run of equal x (R's !duplicated(xo))
    let mut out_x = Vec::with_capacity(distinct);
    let mut out_y = Vec::with_capacity(distinct);
    for j in 0..n {
        if j == 0 || xs[j] != xs[j - 1] {
            out_x.push(xs[j]);
            out_y.push(smo[j]);
        }
    }

    Ok(SupsmuResult {
        x: out_x,
        y: out_y,
        bass,
        periodic,
        n: distinct,
        missing,
    })
}

/// 1 / f for a positive weight sum f, else 0.
#[inline(always)]
fn recip(f: f64) -> f64 {
    if f > 0.0 { 1.0 / f } else { 0.0 }
}

/// The window of [`smooth`]: the weighted means of x and y, the weighted sums of squares
/// and cross-products about them, the weight sum and its reciprocal (0 while the sum is not
/// positive), updated with Friedman's formulas (R `ppr.f` `smooth`).
///
/// The Fortran updates a mean as (fbo m + w v) / fbw, a division on the mean's chain;
/// m += (v - m) * (w / fbw) is the same update with the division depending on the weight sum
/// only, so it is computed off the chain (T3-PERF4), and with the factor w / fbw formed first
/// a mean's chain is one subtraction, one multiplication and one addition per update
/// (T3-PERF5). The rounding error is relative to the step v - m instead of to m.
#[derive(Default)]
struct Running {
    xm: f64,
    ym: f64,
    var: f64,
    cvar: f64,
    fbw: f64,
    rbw: f64,
}

impl Running {
    /// Point (`xti`, `yti`) with weight `wt` enters the window.
    #[inline(always)]
    fn add(&mut self, xti: f64, yti: f64, wt: f64) {
        let fbo = self.fbw;
        let rbo = self.rbw;
        self.fbw += wt;
        self.rbw = recip(self.fbw);
        if self.fbw > 0.0 {
            self.xm += (xti - self.xm) * (wt * self.rbw);
            self.ym += (yti - self.ym) * (wt * self.rbw);
        }
        let tmp = if fbo > 0.0 {
            self.fbw * wt * (xti - self.xm) * rbo
        } else {
            0.0
        };
        self.var += tmp * (xti - self.xm);
        self.cvar += tmp * (yti - self.ym);
    }

    /// Point (`xto`, `yto`) with weight `wt` leaves the window.
    #[inline(always)]
    fn remove(&mut self, xto: f64, yto: f64, wt: f64) {
        let fbo = self.fbw;
        self.fbw -= wt;
        self.rbw = recip(self.fbw);
        let tmp = if self.fbw > 0.0 {
            fbo * wt * (xto - self.xm) * self.rbw
        } else {
            0.0
        };
        self.var -= tmp * (xto - self.xm);
        self.cvar -= tmp * (yto - self.ym);
        if self.fbw > 0.0 {
            self.xm += (self.xm - xto) * (wt * self.rbw);
            self.ym += (self.ym - yto) * (wt * self.rbw);
        }
    }

    /// The window's least-squares line at `xj` (slope 0 when the window's x variance is at
    /// most `vsmlsq`).
    #[inline(always)]
    fn fit(&self, xj: f64, vsmlsq: f64) -> f64 {
        let a = if self.var > vsmlsq {
            self.cvar / self.var
        } else {
            0.0
        };
        a * (xj - self.xm) + self.ym
    }
}

/// Running-lines smooth of `y` on sorted `x` with windows of `2 ibw + 1` points (R `ppr.f`
/// `smooth`). `|iper| == 2` wraps the windows around [0, 1]; `iper > 0` also writes the
/// exact leave-one-out absolute residuals into `acvr`. Tied x are averaged at the end.
// Out of line: inlined into the call sites of `super_smoother` it measured 4-6% slower.
#[inline(never)]
fn smooth(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    span: f64,
    iper: i32,
    vsmlsq: f64,
    smo: &mut [f64],
    mut acvr: Option<&mut [f64]>,
) -> EconResult<()> {
    crate::resource::checkpoint()?;
    let n = x.len() as i64;
    // 1-based accessors, as the Fortran
    let xa = |j: i64| x[(j - 1) as usize];
    let ya = |j: i64| y[(j - 1) as usize];
    let wa = |j: i64| w[(j - 1) as usize];
    let mut st = Running::default();
    let jper = iper.abs();
    let ibw = half_window(span, x.len()) as i64;
    let it = (2 * ibw + 1).min(n);
    for i in 1..=it {
        let mut j = i;
        if jper == 2 {
            j = i - ibw - 1;
        }
        let xti = if j >= 1 {
            xa(j)
        } else {
            j += n;
            xa(j) - 1.0
        };
        st.add(xti, ya(j), wa(j));
    }
    // Without residuals and wrapping, the windows of j = ibw + 2 ..= n - ibw move by one
    // point inside the data: that run takes its points from slices (no 1-based index
    // arithmetic or bounds checks per point) with the same arithmetic in the same order.
    let interior = (ibw + 2)..=(n - ibw);
    let fast = jper != 2 && acvr.is_none() && !interior.is_empty();
    let mut j = 1;
    while j <= n {
        if fast && j == *interior.start() {
            let (lo, hi, b) = ((j - 1) as usize, *interior.end() as usize, ibw as usize);
            let (outs, ins) = ((lo - b - 1)..(hi - b - 1), (lo + b)..(hi + b));
            let leaving = x[outs.clone()].iter().zip(&y[outs.clone()]).zip(&w[outs]);
            let entering = x[ins.clone()].iter().zip(&y[ins.clone()]).zip(&w[ins]);
            for (((((&xto, &yto), &wto), ((&xti, &yti), &wti)), &xj), s) in
                leaving.zip(entering).zip(&x[lo..hi]).zip(&mut smo[lo..hi])
            {
                st.remove(xto, yto, wto);
                st.add(xti, yti, wti);
                *s = st.fit(xj, vsmlsq);
            }
            j = *interior.end() + 1;
            continue;
        }
        let mut out = j - ibw - 1;
        let mut inn = j + ibw;
        if !(jper != 2 && (out < 1 || inn > n)) {
            let (xto, xti);
            if out < 1 {
                out += n;
                xto = xa(out) - 1.0;
                xti = xa(inn);
            } else if inn > n {
                inn -= n;
                xti = xa(inn) + 1.0;
                xto = xa(out);
            } else {
                xto = xa(out);
                xti = xa(inn);
            }
            // remove the point leaving the window, add the one entering it
            st.remove(xto, ya(out), wa(out));
            st.add(xti, ya(inn), wa(inn));
        }
        let ju = (j - 1) as usize;
        smo[ju] = st.fit(xa(j), vsmlsq);
        if iper > 0
            && let Some(acvr) = acvr.as_deref_mut()
        {
            let mut h = st.rbw;
            if st.var > vsmlsq {
                h += (xa(j) - st.xm).powi(2) / st.var;
            }
            acvr[ju] = 0.0;
            let a = 1.0 - wa(j) * h;
            if a > 0.0 {
                acvr[ju] = (ya(j) - smo[ju]).abs() / a;
            } else if j > 1 {
                acvr[ju] = acvr[ju - 1];
            }
        }
        j += 1;
    }
    // tied x: replace the fits by their weighted mean
    let nu = x.len();
    let mut j = 0usize;
    while j < nu {
        let j0 = j;
        let mut sy = smo[j] * w[j];
        let mut fbw = w[j];
        while j + 1 < nu && x[j + 1] <= x[j] {
            j += 1;
            sy += w[j] * smo[j];
            fbw += w[j];
        }
        if j > j0 {
            let a = if fbw > 0.0 { sy / fbw } else { 0.0 };
            smo[j0..=j].iter_mut().for_each(|v| *v = a);
        }
        j += 1;
    }
    Ok(())
}

// The three residual smooths of supsmu's cross-validation share their span, x and weights,
// so the part of Friedman's updating formulas that depends on x only (the window's weighted
// mean and variance of x and the update factors, `XWindow`) is computed once for the three
// responses (`YState`, their weighted means and covariances with x) in `smooth_many`. The
// update factors' divisions are the bulk of a smooth's cost; sharing the x part removes those
// of two of the three residual smooths.

/// A running-lines window of `2 ibw + 1` points over sorted `x` (1-based, as the Fortran);
/// `periodic` wraps it around [0, 1].
struct XWindow<'a> {
    x: &'a [f64],
    w: &'a [f64],
    n: i64,
    ibw: i64,
    periodic: bool,
    xm: f64,
    var: f64,
    fbw: f64,
    /// 1 / fbw, 0 while fbw <= 0
    rbw: f64,
}

impl<'a> XWindow<'a> {
    fn new(x: &'a [f64], w: &'a [f64], span: f64, iper: i32) -> Self {
        XWindow {
            x,
            w,
            n: x.len() as i64,
            ibw: half_window(span, x.len()) as i64,
            periodic: iper.abs() == 2,
            xm: 0.0,
            var: 0.0,
            fbw: 0.0,
            rbw: 0.0,
        }
    }

    #[inline(always)]
    fn xa(&self, j: i64) -> f64 {
        self.x[(j - 1) as usize]
    }

    #[inline(always)]
    fn wa(&self, j: i64) -> f64 {
        self.w[(j - 1) as usize]
    }

    /// Number of points of the initial window.
    fn n_init(&self) -> i64 {
        (2 * self.ibw + 1).min(self.n)
    }

    /// Point `i` (1..=n_init) of the initial window: its index and abscissa (shifted by one
    /// when wrapped).
    #[inline(always)]
    fn init_point(&self, i: i64) -> (i64, f64) {
        let mut j = i;
        if self.periodic {
            j = i - self.ibw - 1;
        }
        if j >= 1 {
            (j, self.xa(j))
        } else {
            j += self.n;
            (j, self.xa(j) - 1.0)
        }
    }

    /// The points leaving and entering the window when it moves to point `j`, with their
    /// abscissae; `None` at the ends of a non-periodic smooth, where the window stays.
    #[inline(always)]
    fn moves(&self, j: i64) -> Option<(i64, f64, i64, f64)> {
        let (n, ibw) = (self.n, self.ibw);
        let mut out = j - ibw - 1;
        let mut inn = j + ibw;
        if !self.periodic && (out < 1 || inn > n) {
            return None;
        }
        Some(if out < 1 {
            out += n;
            (out, self.xa(out) - 1.0, inn, self.xa(inn))
        } else if inn > n {
            inn -= n;
            (out, self.xa(out), inn, self.xa(inn) + 1.0)
        } else {
            (out, self.xa(out), inn, self.xa(inn))
        })
    }

    /// Add point `j` with abscissa `xti`: the new weight sum, its reciprocal and the update
    /// factor (the arithmetic of `smooth`).
    #[inline(always)]
    fn add(&mut self, j: i64, xti: f64) -> (f64, f64, f64) {
        let wt = self.wa(j);
        let fbo = self.fbw;
        let rbo = self.rbw;
        self.fbw += wt;
        self.rbw = recip(self.fbw);
        if self.fbw > 0.0 {
            self.xm += (xti - self.xm) * (wt * self.rbw);
        }
        let tmp = if fbo > 0.0 {
            self.fbw * wt * (xti - self.xm) * rbo
        } else {
            0.0
        };
        self.var += tmp * (xti - self.xm);
        (self.fbw, self.rbw, tmp)
    }

    /// Remove point `j` with abscissa `xto`: the new weight sum, its reciprocal and the
    /// update factor (the arithmetic of `smooth`).
    #[inline(always)]
    fn remove(&mut self, j: i64, xto: f64) -> (f64, f64, f64) {
        let wt = self.wa(j);
        let fbo = self.fbw;
        self.fbw -= wt;
        self.rbw = recip(self.fbw);
        let tmp = if self.fbw > 0.0 {
            fbo * wt * (xto - self.xm) * self.rbw
        } else {
            0.0
        };
        self.var -= tmp * (xto - self.xm);
        if self.fbw > 0.0 {
            self.xm += (self.xm - xto) * (wt * self.rbw);
        }
        (self.fbw, self.rbw, tmp)
    }
}

/// The response part of `N` running-lines smooths sharing one window: each response's
/// weighted mean `ym` and covariance with x `cvar`.
struct YState<'a, const N: usize> {
    ys: [&'a [f64]; N],
    ym: [f64; N],
    cvar: [f64; N],
}

impl<'a, const N: usize> YState<'a, N> {
    fn new(ys: [&'a [f64]; N]) -> Self {
        YState {
            ys,
            ym: [0.0; N],
            cvar: [0.0; N],
        }
    }

    /// Point `j` (weight `wt`) entered the window, whose weight sum is now `fbw` (reciprocal
    /// `rbw`), with update factor `tmp`.
    #[inline(always)]
    fn add(&mut self, j: i64, wt: f64, (fbw, rbw, tmp): (f64, f64, f64)) {
        let ju = (j - 1) as usize;
        for k in 0..N {
            let y = self.ys[k][ju];
            if fbw > 0.0 {
                self.ym[k] += (y - self.ym[k]) * (wt * rbw);
            }
            self.cvar[k] += tmp * (y - self.ym[k]);
        }
    }

    /// Point `j` (weight `wt`) left the window, whose weight sum is now `fbw` (reciprocal
    /// `rbw`), with update factor `tmp`.
    #[inline(always)]
    fn remove(&mut self, j: i64, wt: f64, (fbw, rbw, tmp): (f64, f64, f64)) {
        let ju = (j - 1) as usize;
        for k in 0..N {
            let y = self.ys[k][ju];
            self.cvar[k] -= tmp * (y - self.ym[k]);
            if fbw > 0.0 {
                self.ym[k] += (self.ym[k] - y) * (wt * rbw);
            }
        }
    }
}

/// Tied x: replace the fits of each run of equal x by their weighted mean (the end of R
/// `ppr.f` `smooth`).
fn average_ties(x: &[f64], w: &[f64], smo: &mut [f64]) {
    let nu = x.len();
    let mut j = 0usize;
    while j < nu {
        let j0 = j;
        let mut sy = smo[j] * w[j];
        let mut fbw = w[j];
        while j + 1 < nu && x[j + 1] <= x[j] {
            j += 1;
            sy += w[j] * smo[j];
            fbw += w[j];
        }
        if j > j0 {
            let a = if fbw > 0.0 { sy / fbw } else { 0.0 };
            smo[j0..=j].iter_mut().for_each(|v| *v = a);
        }
        j += 1;
    }
}

/// Running-lines smooths of the `N` responses `ys` on sorted `x` with one span and no
/// leave-one-out residuals (R `ppr.f` `smooth` once per response), with one x part for all of
/// them; each smooth is bit-identical to its own `smooth` call. Tied x are averaged at the end.
fn smooth_many<const N: usize>(
    x: &[f64],
    w: &[f64],
    ys: [&[f64]; N],
    span: f64,
    iper: i32,
    vsmlsq: f64,
    mut out: [&mut [f64]; N],
) -> EconResult<()> {
    crate::resource::checkpoint()?;
    let mut win = XWindow::new(x, w, span, iper);
    let mut yst = YState::new(ys);
    for i in 1..=win.n_init() {
        let (j, xti) = win.init_point(i);
        let upd = win.add(j, xti);
        yst.add(j, win.wa(j), upd);
    }
    let mut a = [0.0; N];
    for j in 1..=win.n {
        let ju = (j - 1) as usize;
        let moved = win.moves(j);
        if let Some((out_j, xto, inn_j, xti)) = moved {
            let upd_out = win.remove(out_j, xto);
            yst.remove(out_j, win.wa(out_j), upd_out);
            let upd_in = win.add(inn_j, xti);
            yst.add(inn_j, win.wa(inn_j), upd_in);
        }
        if moved.is_some() || j == 1 {
            for k in 0..N {
                a[k] = if win.var > vsmlsq {
                    yst.cvar[k] / win.var
                } else {
                    0.0
                };
            }
        }
        let xj = win.xa(j);
        for (k, o) in out.iter_mut().enumerate() {
            o[ju] = a[k] * (xj - win.xm) + yst.ym[k];
        }
    }
    for o in out.iter_mut() {
        average_ties(x, w, o);
    }
    Ok(())
}

/// Friedman's super smoother on sorted data (R `ppr.f` `supsmu`); `span > 0` is a fixed
/// span, 0 is cross-validation.
fn super_smoother(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    span: f64,
    iper: i32,
    bass: f64,
) -> EconResult<Vec<f64>> {
    use crate::resource::try_vec;
    let n = x.len();
    // scale: an interquartile-type range of x, widened until positive
    let (mut i, mut j) = (n / 4, 3 * (n / 4));
    let mut scale = x[j - 1] - x[i - 1];
    while scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        if j < n {
            j += 1;
        }
        if i > 1 {
            i -= 1;
        }
        scale = x[j - 1] - x[i - 1];
    }
    let vsmlsq = (EPS * scale).powi(2);
    let jper = iper;
    let mut smo = try_vec(n, 0.0, "supsmu smooth")?;
    if span > 0.0 {
        smooth(x, y, w, span, jper, vsmlsq, &mut smo, None)?;
        return Ok(smo);
    }
    // sc(n, 7) of the Fortran: the three smooths, their leave-one-out residuals and the
    // residual smooths
    let buf = |what| try_vec(n, 0.0, what);
    let mut fits = [
        buf("supsmu smooth")?,
        buf("supsmu smooth")?,
        buf("supsmu smooth")?,
    ];
    let mut acvr = [
        buf("supsmu residuals")?,
        buf("supsmu residuals")?,
        buf("supsmu residuals")?,
    ];
    for k in 0..3 {
        smooth(
            x,
            y,
            w,
            SPANS[k],
            jper,
            vsmlsq,
            &mut fits[k],
            Some(&mut acvr[k]),
        )?;
    }
    let mut res = [
        buf("supsmu residual smooth")?,
        buf("supsmu residual smooth")?,
        buf("supsmu residual smooth")?,
    ];
    {
        let [r0, r1, r2] = &mut res;
        smooth_many(
            x,
            w,
            [&acvr[0][..], &acvr[1][..], &acvr[2][..]],
            SPANS[1],
            -jper,
            vsmlsq,
            [&mut r0[..], &mut r1[..], &mut r2[..]],
        )?;
    }
    // per-point span: the smallest smoothed CV residual, moved toward the woofer by bass
    // the work column of the Fortran, which holds the woofer's residuals at this point
    let [_, _, work] = &mut acvr;
    for jj in 0..n {
        let mut resmin = BIG;
        for k in 0..3 {
            if res[k][jj] < resmin {
                resmin = res[k][jj];
                work[jj] = SPANS[k];
            }
        }
        if bass > 0.0 && bass <= 10.0 && resmin < res[2][jj] && resmin > 0.0 {
            work[jj] += (SPANS[2] - work[jj]) * SML.max(resmin / res[2][jj]).powf(10.0 - bass);
        }
    }
    // smooth the spans, clamp them and interpolate between the three smooths
    let [spans_smooth, ..] = &mut res;
    smooth(x, work, w, SPANS[1], -jper, vsmlsq, spans_smooth, None)?;
    for jj in 0..n {
        let s = spans_smooth[jj].clamp(SPANS[0], SPANS[2]);
        let f = s - SPANS[1];
        work[jj] = if f < 0.0 {
            let f = -f / (SPANS[1] - SPANS[0]);
            (1.0 - f) * fits[1][jj] + f * fits[0][jj]
        } else {
            let f = f / (SPANS[2] - SPANS[1]);
            (1.0 - f) * fits[1][jj] + f * fits[2][jj]
        };
    }
    smooth(x, work, w, SPANS[0], -jper, vsmlsq, &mut smo, None)?;
    Ok(smo)
}

/// Run SuperSmoother (convenience wrapper).
pub fn run_supsmu(
    x: &[f64],
    y: &[f64],
    wt: Option<&[f64]>,
    span: Option<f64>,
    periodic: bool,
    bass: f64,
) -> Result<SupsmuResult, String> {
    supsmu(x, y, wt, span, periodic, bass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// supsmu before T3-PERF4, the reference of the bit-identity test: each of the three
    /// residual smooths is its own `smooth` call.
    fn super_smoother_reference(
        x: &[f64],
        y: &[f64],
        w: &[f64],
        span: f64,
        iper: i32,
        bass: f64,
    ) -> EconResult<Vec<f64>> {
        use crate::resource::try_vec;
        let n = x.len();
        // scale: an interquartile-type range of x, widened until positive
        let (mut i, mut j) = (n / 4, 3 * (n / 4));
        let mut scale = x[j - 1] - x[i - 1];
        while scale.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            if j < n {
                j += 1;
            }
            if i > 1 {
                i -= 1;
            }
            scale = x[j - 1] - x[i - 1];
        }
        let vsmlsq = (EPS * scale).powi(2);
        let jper = iper;
        let mut smo = try_vec(n, 0.0, "supsmu smooth")?;
        if span > 0.0 {
            smooth(x, y, w, span, jper, vsmlsq, &mut smo, None)?;
            return Ok(smo);
        }
        // sc(n, 7): smooths and residual smooths of the three spans, and a work column
        let mut fits: Vec<Vec<f64>> = Vec::with_capacity(3);
        let mut res: Vec<Vec<f64>> = Vec::with_capacity(3);
        let mut work = try_vec(n, 0.0, "supsmu work")?;
        for &s in &SPANS {
            let mut f = try_vec(n, 0.0, "supsmu smooth")?;
            smooth(x, y, w, s, jper, vsmlsq, &mut f, Some(&mut work))?;
            let mut r = try_vec(n, 0.0, "supsmu residual smooth")?;
            smooth(x, &work, w, SPANS[1], -jper, vsmlsq, &mut r, None)?;
            fits.push(f);
            res.push(r);
        }
        // per-point span: the smallest smoothed CV residual, moved toward the woofer by bass
        for jj in 0..n {
            let mut resmin = BIG;
            for k in 0..3 {
                if res[k][jj] < resmin {
                    resmin = res[k][jj];
                    work[jj] = SPANS[k];
                }
            }
            if bass > 0.0 && bass <= 10.0 && resmin < res[2][jj] && resmin > 0.0 {
                work[jj] += (SPANS[2] - work[jj]) * SML.max(resmin / res[2][jj]).powf(10.0 - bass);
            }
        }
        // smooth the spans, clamp them and interpolate between the three smooths
        let mut spans_smooth = try_vec(n, 0.0, "supsmu spans")?;
        smooth(
            x,
            &work,
            w,
            SPANS[1],
            -jper,
            vsmlsq,
            &mut spans_smooth,
            None,
        )?;
        for jj in 0..n {
            let s = spans_smooth[jj].clamp(SPANS[0], SPANS[2]);
            let f = s - SPANS[1];
            work[jj] = if f < 0.0 {
                let f = -f / (SPANS[1] - SPANS[0]);
                (1.0 - f) * fits[1][jj] + f * fits[0][jj]
            } else {
                let f = f / (SPANS[2] - SPANS[1]);
                (1.0 - f) * fits[1][jj] + f * fits[2][jj]
            };
        }
        smooth(x, &work, w, SPANS[0], -jper, vsmlsq, &mut smo, None)?;
        Ok(smo)
    }

    /// T3-PERF4: the three residual smooths sharing their x updates (`smooth_many`) give
    /// bit-identical results to three `smooth` calls, on noisy data with ties and zero and
    /// unequal weights, periodic and not, three bass values and two fixed spans.
    #[test]
    fn test_supsmu_shared_residual_smooths_are_bit_identical() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut u = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for &n in &[7usize, 23, 50, 200, 1003] {
            let mut x: Vec<f64> = (0..n).map(|_| u()).collect();
            // ties: round a third of the points to two decimals
            for v in x.iter_mut().step_by(3) {
                *v = (*v * 100.0).round() / 100.0;
            }
            x.sort_by(f64::total_cmp);
            let y: Vec<f64> = x
                .iter()
                .map(|&v| (6.0 * v).sin() + 0.4 * (u() - 0.5))
                .collect();
            let w: Vec<f64> = (0..n)
                .map(|i| if i % 11 == 0 { 0.0 } else { 0.5 + u() })
                .collect();
            let ones = vec![1.0; n];
            for wt in [&ones, &w] {
                for iper in [1, 2] {
                    for (span, bass) in
                        [(0.0, 0.0), (0.0, 3.0), (0.0, 10.0), (0.3, 0.0), (0.05, 0.0)]
                    {
                        let fused = super_smoother(&x, &y, wt, span, iper, bass).unwrap();
                        let reference =
                            super_smoother_reference(&x, &y, wt, span, iper, bass).unwrap();
                        let bits = |v: &[f64]| v.iter().map(|a| a.to_bits()).collect::<Vec<_>>();
                        assert_eq!(
                            bits(&fused),
                            bits(&reference),
                            "n {n} iper {iper} span {span} bass {bass}"
                        );
                    }
                }
            }
        }
    }

    /// `smooth` before T3-PERF4, with the divisions of R's Fortran: the reference of the
    /// reciprocal-update test.
    #[allow(clippy::too_many_arguments)]
    fn smooth_division_reference(
        x: &[f64],
        y: &[f64],
        w: &[f64],
        span: f64,
        iper: i32,
        vsmlsq: f64,
        smo: &mut [f64],
        mut acvr: Option<&mut [f64]>,
    ) -> EconResult<()> {
        crate::resource::checkpoint()?;
        let n = x.len() as i64;
        // 1-based accessors, as the Fortran
        let xa = |j: i64| x[(j - 1) as usize];
        let ya = |j: i64| y[(j - 1) as usize];
        let wa = |j: i64| w[(j - 1) as usize];
        let (mut xm, mut ym, mut var, mut cvar, mut fbw) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let jper = iper.abs();
        let ibw = half_window(span, x.len()) as i64;
        let it = (2 * ibw + 1).min(n);
        for i in 1..=it {
            let mut j = i;
            if jper == 2 {
                j = i - ibw - 1;
            }
            let xti = if j >= 1 {
                xa(j)
            } else {
                j += n;
                xa(j) - 1.0
            };
            let wt = wa(j);
            let fbo = fbw;
            fbw += wt;
            if fbw > 0.0 {
                xm = (fbo * xm + wt * xti) / fbw;
                ym = (fbo * ym + wt * ya(j)) / fbw;
            }
            let tmp = if fbo > 0.0 {
                fbw * wt * (xti - xm) / fbo
            } else {
                0.0
            };
            var += tmp * (xti - xm);
            cvar += tmp * (ya(j) - ym);
        }
        for j in 1..=n {
            let mut out = j - ibw - 1;
            let mut inn = j + ibw;
            if !(jper != 2 && (out < 1 || inn > n)) {
                let (xto, xti);
                if out < 1 {
                    out += n;
                    xto = xa(out) - 1.0;
                    xti = xa(inn);
                } else if inn > n {
                    inn -= n;
                    xti = xa(inn) + 1.0;
                    xto = xa(out);
                } else {
                    xto = xa(out);
                    xti = xa(inn);
                }
                // remove the point leaving the window
                let wt = wa(out);
                let fbo = fbw;
                fbw -= wt;
                let tmp = if fbw > 0.0 {
                    fbo * wt * (xto - xm) / fbw
                } else {
                    0.0
                };
                var -= tmp * (xto - xm);
                cvar -= tmp * (ya(out) - ym);
                if fbw > 0.0 {
                    xm = (fbo * xm - wt * xto) / fbw;
                    ym = (fbo * ym - wt * ya(out)) / fbw;
                }
                // add the point entering it
                let wt = wa(inn);
                let fbo = fbw;
                fbw += wt;
                if fbw > 0.0 {
                    xm = (fbo * xm + wt * xti) / fbw;
                    ym = (fbo * ym + wt * ya(inn)) / fbw;
                }
                let tmp = if fbo > 0.0 {
                    fbw * wt * (xti - xm) / fbo
                } else {
                    0.0
                };
                var += tmp * (xti - xm);
                cvar += tmp * (ya(inn) - ym);
            }
            let a = if var > vsmlsq { cvar / var } else { 0.0 };
            let ju = (j - 1) as usize;
            smo[ju] = a * (xa(j) - xm) + ym;
            if iper > 0
                && let Some(acvr) = acvr.as_deref_mut()
            {
                let mut h = if fbw > 0.0 { 1.0 / fbw } else { 0.0 };
                if var > vsmlsq {
                    h += (xa(j) - xm).powi(2) / var;
                }
                acvr[ju] = 0.0;
                let a = 1.0 - wa(j) * h;
                if a > 0.0 {
                    acvr[ju] = (ya(j) - smo[ju]).abs() / a;
                } else if j > 1 {
                    acvr[ju] = acvr[ju - 1];
                }
            }
        }
        // tied x: replace the fits by their weighted mean
        let nu = x.len();
        let mut j = 0usize;
        while j < nu {
            let j0 = j;
            let mut sy = smo[j] * w[j];
            let mut fbw = w[j];
            while j + 1 < nu && x[j + 1] <= x[j] {
                j += 1;
                sy += w[j] * smo[j];
                fbw += w[j];
            }
            if j > j0 {
                let a = if fbw > 0.0 { sy / fbw } else { 0.0 };
                smo[j0..=j].iter_mut().for_each(|v| *v = a);
            }
            j += 1;
        }
        Ok(())
    }

    /// T3-PERF5: without residuals, `smooth` runs the windows that move inside the data from
    /// slices; the fits are bit-identical to the indexed loop (which runs when residuals are
    /// asked for, and writes them only for `iper > 0`), on noisy data with ties and zero and
    /// unequal weights, for spans whose window covers all the data too.
    #[test]
    fn test_supsmu_interior_run_matches_indexed_loop() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut u = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for &n in &[4usize, 9, 50, 200, 1003] {
            let mut x: Vec<f64> = (0..n).map(|_| u()).collect();
            for v in x.iter_mut().step_by(3) {
                *v = (*v * 50.0).round() / 50.0;
            }
            x.sort_by(f64::total_cmp);
            let y: Vec<f64> = x
                .iter()
                .map(|&v| 3.0 + (5.0 * v).cos() + 0.3 * (u() - 0.5))
                .collect();
            let w: Vec<f64> = (0..n)
                .map(|i| if i % 7 == 0 { 0.0 } else { 0.5 + u() })
                .collect();
            let ones = vec![1.0; n];
            for wt in [&ones, &w] {
                for span in [0.05, 0.1, 0.2, 0.5, 1.0] {
                    for iper in [1, -1] {
                        let (mut fast, mut indexed, mut res) =
                            (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
                        smooth(&x, &y, wt, span, iper, 1e-20, &mut fast, None).unwrap();
                        smooth(&x, &y, wt, span, iper, 1e-20, &mut indexed, Some(&mut res))
                            .unwrap();
                        for (a, b) in fast.iter().zip(&indexed) {
                            assert_eq!(a.to_bits(), b.to_bits(), "n {n} span {span} iper {iper}");
                        }
                    }
                }
            }
        }
    }

    /// T3-PERF4: `smooth` updates the window's means as `m += w (v - m) / W` with the
    /// reciprocal of the weight sum W, where the Fortran forms `(W_old m + w v) / W`; the
    /// divisions then depend only on W and leave the chains of the running means. On noisy data
    /// with ties and zero and unequal weights, periodic and not, with the leave-one-out
    /// residuals: with x in [0, 1] the smooths and residuals are within 1e-10 (relative to the
    /// response's scale) of the division form; with x shifted by 1000, where both forms lose
    /// digits to cancellation, the new form's error against the unshifted smooth is at most
    /// 1e-8 and in total no larger than 1.5 times the division form's.
    #[test]
    fn test_supsmu_reciprocal_updates_match_divisions() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut u = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let run = |f: &dyn Fn(&[f64], &[f64], &[f64], f64, i32, &mut [f64], &mut [f64]),
                   x: &[f64],
                   y: &[f64],
                   w: &[f64],
                   span: f64,
                   iper: i32| {
            let n = x.len();
            let (mut s, mut a) = (vec![0.0; n], vec![0.0; n]);
            f(x, y, w, span, iper, &mut s, &mut a);
            [s, a]
        };
        let new = |x: &[f64], y: &[f64], w: &[f64], span, iper, s: &mut [f64], a: &mut [f64]| {
            smooth(x, y, w, span, iper, 1e-20, s, Some(a)).unwrap()
        };
        let old = |x: &[f64], y: &[f64], w: &[f64], span, iper, s: &mut [f64], a: &mut [f64]| {
            smooth_division_reference(x, y, w, span, iper, 1e-20, s, Some(a)).unwrap()
        };
        let max_diff = |p: &[Vec<f64>; 2], q: &[Vec<f64>; 2]| {
            (0..2)
                .flat_map(|k| p[k].iter().zip(q[k].iter()).map(|(a, b)| (a - b).abs()))
                .fold(0.0f64, f64::max)
                / 6.0
        };
        let (mut total_new, mut total_old) = (0.0, 0.0);
        for &n in &[7usize, 23, 200, 1003] {
            let mut x: Vec<f64> = (0..n).map(|_| u()).collect();
            for v in x.iter_mut().step_by(3) {
                *v = (*v * 100.0).round() / 100.0;
            }
            x.sort_by(f64::total_cmp);
            let y: Vec<f64> = x
                .iter()
                .map(|&v| 5.0 + (6.0 * v).sin() + 0.4 * (u() - 0.5))
                .collect();
            let shifted: Vec<f64> = x.iter().map(|&v| v + 1000.0).collect();
            let w: Vec<f64> = (0..n)
                .map(|i| if i % 11 == 0 { 0.0 } else { 0.5 + u() })
                .collect();
            let ones = vec![1.0; n];
            for wt in [&ones, &w] {
                for span in [0.05, 0.2, 0.5] {
                    for iper in [1, 2, -1, -2] {
                        let exact = run(&old, &x, &y, wt, span, iper);
                        let d = max_diff(&run(&new, &x, &y, wt, span, iper), &exact);
                        assert!(d <= 1e-10, "n {n} span {span} iper {iper}: {d:e}");
                        if iper.abs() == 2 {
                            continue; // the periodic window wraps around [0, 1]
                        }
                        let e_new = max_diff(&run(&new, &shifted, &y, wt, span, iper), &exact);
                        let e_old = max_diff(&run(&old, &shifted, &y, wt, span, iper), &exact);
                        assert!(
                            e_new <= 1e-8,
                            "shifted n {n} span {span} iper {iper}: {e_new:e}"
                        );
                        total_new += e_new;
                        total_old += e_old;
                    }
                }
            }
        }
        assert!(
            total_new <= 1.5 * total_old,
            "shifted x: total error {total_new:e} against the division form's {total_old:e}"
        );
    }

    #[test]
    fn test_supsmu_basic() {
        // Simple linear data
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = supsmu(&x, &y, None, None, false, 0.0).unwrap();

        assert_eq!(result.n, 50);
        // Smoothed values should be close to original for linear data
        for i in 5..45 {
            // Avoid boundary effects
            assert_relative_eq!(result.y[i], y[i], epsilon = 1.0);
        }
    }

    #[test]
    fn test_supsmu_with_noise() {
        // Sine wave with noise
        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (2.0 * std::f64::consts::PI * xi).sin() + 0.1 * (i as f64 % 3.0 - 1.0))
            .collect();

        let result = supsmu(&x, &y, None, None, false, 5.0).unwrap();

        assert_eq!(result.n, 100);
        // Smoothed curve should have less variation than noisy input
        let smooth_var: f64 =
            result.y.iter().map(|&yi| yi * yi).sum::<f64>() / result.y.len() as f64;

        // Both should be similar magnitude for sinusoidal data
        assert!(smooth_var > 0.1); // Not completely flat
    }

    #[test]
    fn test_supsmu_with_weights() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi + 1.0).collect();
        let wt: Vec<f64> = (0..30).map(|i| if i < 15 { 1.0 } else { 2.0 }).collect();

        let result = supsmu(&x, &y, Some(&wt), None, false, 0.0).unwrap();

        assert_eq!(result.n, 30);
    }

    #[test]
    fn test_supsmu_periodic() {
        let x: Vec<f64> = (0..40).map(|i| i as f64 / 40.0).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).sin())
            .collect();

        let result = supsmu(&x, &y, None, None, true, 0.0).unwrap();

        assert!(result.periodic);
    }

    #[test]
    fn test_supsmu_bass_parameter() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        let result_no_bass = supsmu(&x, &y, None, None, false, 0.0).unwrap();
        let result_high_bass = supsmu(&x, &y, None, None, false, 10.0).unwrap();

        // High bass should produce smoother output
        assert_relative_eq!(result_no_bass.bass, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result_high_bass.bass, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_supsmu_length_mismatch() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let result = supsmu(&x, &y, None, None, false, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_supsmu_too_few_points() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let result = supsmu(&x, &y, None, None, false, 0.0);
        assert!(result.is_err());
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    fn col(ds: &crate::data::Dataset, n: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), n, crate::data::columns::NaPolicy::Fail)
            .unwrap()
            .0
    }

    /// Case `smoothers_gcv_periodic_loess`: stats::supsmu, periodic and not, cv / span 0.3 /
    /// bass 5 (REG-55, REG-31).
    #[test]
    fn test_validate_supsmu_periodic() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("smoothers_gcv_periodic_loess");
        let (x, y) = (col(&ds, "x"), col(&ds, "y"));
        let run = |span: Option<f64>, periodic: bool, bass: f64| {
            supsmu(&x, &y, None, span, periodic, bass).unwrap()
        };
        let (cv_p, cv) = (run(None, true, 0.0), run(None, false, 0.0));
        assert!(cv_p.periodic && !cv.periodic && cv_p.n == 100);
        Check::new(&case)
            .vector("supsmu_periodic_y", &cv_p.y, Tol::coef(100))
            .vector("supsmu_plain_y", &cv.y, Tol::coef(100))
            .vector(
                "supsmu_span03_periodic_y",
                &run(Some(0.3), true, 0.0).y,
                Tol::coef(100),
            )
            .vector(
                "supsmu_span03_plain_y",
                &run(Some(0.3), false, 0.0).y,
                Tol::coef(100),
            )
            .vector("supsmu_bass5_y", &run(None, false, 5.0).y, Tol::coef(100))
            .vector(
                "supsmu_bass5_periodic_y",
                &run(None, true, 5.0).y,
                Tol::coef(100),
            )
            .finish();
        // two values R distinguishes give different p2a output (convention 2)
        assert!(cv_p.y.iter().zip(&cv.y).any(|(a, b)| (a - b).abs() > 0.05));
    }

    /// Case `supsmu_sine` (REG-44: replaces test_validate_supsmu_sine_wave's correlation >
    /// 0.9 on 20 copied points).
    #[test]
    fn test_validate_supsmu_sine() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("supsmu_sine");
        let (x, y, w, xr) = (col(&ds, "x"), col(&ds, "y"), col(&ds, "w"), col(&ds, "xr"));
        let cv = supsmu(&x, &y, None, None, false, 0.0).unwrap();
        let ties = supsmu(&xr, &y, None, None, false, 0.0).unwrap();
        Check::new(&case)
            .vector("cv_x", &cv.x, Tol::exact())
            .vector("cv_y", &cv.y, Tol::coef(100))
            .vector(
                "span03_y",
                &supsmu(&x, &y, None, Some(0.3), false, 0.0).unwrap().y,
                Tol::coef(100),
            )
            .vector(
                "bass5_y",
                &supsmu(&x, &y, None, None, false, 5.0).unwrap().y,
                Tol::coef(100),
            )
            .vector(
                "weighted_y",
                &supsmu(&x, &y, Some(&w), None, false, 0.0).unwrap().y,
                Tol::coef(100),
            )
            .vector("ties_x", &ties.x, Tol::exact())
            .vector("ties_y", &ties.y, Tol::coef(100))
            .finish();
    }

    /// A periodic smooth on [0, 1) does not depend on where the circle is cut (shift by
    /// 0.25 on a dyadic grid).
    #[test]
    fn test_supsmu_periodic_is_rotation_invariant() {
        let x: Vec<f64> = (0..64).map(|i| i as f64 / 64.0).collect();
        let y: Vec<f64> = (0..64)
            .map(|i| {
                (2.0 * std::f64::consts::PI * x[i]).sin()
                    + 0.3 * (14.0 * std::f64::consts::PI * x[i]).cos()
                    + ((i * 37) % 11) as f64 / 22.0
            })
            .collect();
        let xs: Vec<f64> = x.iter().map(|v| (v + 0.25) % 1.0).collect();
        let (a, b) = (
            supsmu(&x, &y, None, None, true, 0.0).unwrap(),
            supsmu(&xs, &y, None, None, true, 0.0).unwrap(),
        );
        for i in 0..64 {
            let ia = a.x.iter().position(|v| *v == x[i]).unwrap();
            let ib = b.x.iter().position(|v| *v == xs[i]).unwrap();
            assert!(
                (a.y[ia] - b.y[ib]).abs() < 1e-10,
                "point {i}: {} vs {}",
                a.y[ia],
                b.y[ib]
            );
        }
    }

    #[test]
    fn test_supsmu_rejects_invalid_inputs() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect(); // up to 2.9: not in [0, 1]
        let y: Vec<f64> = x
            .iter()
            .map(|v| v.sin() + 0.05 * ((v * 13.0) % 1.0))
            .collect();
        let err = |r: Result<SupsmuResult, String>| r.unwrap_err();
        assert!(err(supsmu(&x, &y, None, None, true, 0.0)).contains("between 0 and 1"));
        assert!(err(supsmu(&x, &y, None, Some(1.5), false, 0.0)).contains("span"));
        assert!(err(supsmu(&x, &y, None, Some(f64::NAN), false, 0.0)).contains("span"));
        assert!(err(supsmu(&x, &y, None, None, false, -1.0)).contains("bass"));
        assert!(err(supsmu(&x, &y, None, None, false, 11.0)).contains("bass"));
        let mut w = vec![1.0; 30];
        w[3] = -1.0;
        assert!(err(supsmu(&x, &y, Some(&w), None, false, 0.0)).contains("weights"));
        let x4 = [0.1, 0.4, 0.6, 0.9];
        assert!(
            err(supsmu(&x4, &[1.0, 2.0, 0.5, 1.5], None, None, true, 0.0)).contains("periodic")
        );
    }

    /// REG-13: observations with a missing (NaN) or infinite x, y or weight are deleted
    /// first, as R's supsmu does (`is.finite(x + y + wt)`); the smooth on the NA data
    /// equals the smooth on the complete rows, and its x grid is R's.
    #[test]
    fn test_validate_supsmu_na() {
        use crate::data::columns::MissingSummary;
        use crate::errors::EconResult;
        use crate::regression::ols::na_fixtures::nan_vec;
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("reg_smoothers_na");
        let (x, y) = (nan_vec(&ds, "x"), nan_vec(&ds, "y"));
        let finite_rows = |cols: &[&[f64]]| -> Vec<usize> {
            (0..x.len())
                .filter(|&i| cols.iter().all(|c| c[i].is_finite()))
                .collect()
        };
        let keep = finite_rows(&[&x, &y]);
        let xc: Vec<f64> = keep.iter().map(|&i| x[i]).collect();
        let yc: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
        let r = supsmu(&x, &y, None, None, false, 0.0).unwrap();
        let c = supsmu(&xc, &yc, None, None, false, 0.0).unwrap();
        assert_eq!(r.n, 56);
        assert!(r.y.iter().all(|v| v.is_finite()));
        assert_eq!(
            r.missing,
            MissingSummary {
                n_total: 60,
                n_used: 56,
                n_dropped: 4,
                by_column: vec![("x".into(), 2), ("y".into(), 2)],
            }
        );
        assert_eq!((&r.x, &r.y), (&c.x, &c.y));
        assert!(c.missing.is_empty());
        // R's unique sorted x after deleting the incomplete pairs: exact, both sides read
        // the same doubles; the smooth is R's (REG-31 fixed).
        Check::new(&case)
            .vector("supsmu_x", &r.x, Tol::exact())
            .vector("supsmu_y", &r.y, Tol::coef(56))
            .scalar("n_complete", r.n as f64, Tol::exact())
            .finish();
        // Weights are deleted jointly: one NaN weight and one +Inf x on top of the NAs.
        let mut wt = vec![1.0; 60];
        wt[1] = f64::NAN;
        let mut xi = x.clone();
        xi[0] = f64::INFINITY;
        let rw = supsmu(&xi, &y, Some(&wt), None, false, 0.0).unwrap();
        assert_eq!(
            rw.missing.by_column,
            vec![("x".into(), 3), ("y".into(), 2), ("wt".into(), 1)]
        );
        assert_eq!((rw.missing.n_used, rw.missing.n_dropped), (54, 6));
        let keep = finite_rows(&[&xi, &y, &wt]);
        let x54: Vec<f64> = keep.iter().map(|&i| xi[i]).collect();
        let y54: Vec<f64> = keep.iter().map(|&i| y[i]).collect();
        let cw = supsmu(&x54, &y54, Some(&vec![1.0; 54]), None, false, 0.0).unwrap();
        assert_eq!((&rw.x, &rw.y), (&cw.x, &cw.y));
        let e = supsmu(
            &[1.0, 2.0, f64::NAN, 4.0],
            &[1.0; 4],
            None,
            None,
            false,
            0.0,
        )
        .unwrap_err();
        assert!(
            e.contains("at least 4 complete") && e.contains("3 of 4"),
            "{e}"
        );
    }
}
