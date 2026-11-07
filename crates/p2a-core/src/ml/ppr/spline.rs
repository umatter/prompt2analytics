//! The regression-spline ridge smoother of `ppr` (`sm.method = "spline"` and `"gcvspline"`).
//!
//! A penalised cubic B-spline regression on at most 15 coefficients, the small dense problem
//! behind R's `ppr.f` `spline()`:
//!
//! - knots as `ppr.f` places them (a convention that defines the basis, single-precision
//!   fraction included);
//! - `(X'WX + lambda Omega) c = X'Wy` with `Omega` the exact integral of `B_i'' B_j''`, solved by
//!   a banded Cholesky factorisation;
//! - `lambda = r 256^(3 spar - 1)`, `r = tr(X'WX) / tr(Omega)` over the inner coefficients (R's
//!   parametrisation of `smooth.spline`);
//! - `spar` in [0, 1.5] chosen by Brent's minimiser (R's bracket, tolerance and eps) on
//!   `3 + (df - edf)^2` (df matching) or `(rss / sum w) / (1 - gcvpen edf / sum w)^2` (GCV).
//!
//! The returned fit, df, spar and lambda all belong to the minimiser. R differs twice: `sgram.f`
//! integrates the quadratic term with 0.333 for 1/3, and `sbart.c` returns the smooth, leverages
//! and lambda of its last Brent evaluation, not of the spar it reports.

use super::SmoothingMethod;
use crate::errors::{EconError, EconResult};
use crate::resource;

/// Most B-spline coefficients: `ppr.f` uses `nk = min(n, 15)`.
const NK_MAX: usize = 15;
/// Band of the symmetric matrices: `band[i][d]` is entry `(i, i + d)`, `d <= 3`.
type Band = [[f64; 4]; NK_MAX];

/// What Brent's minimiser minimises.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SplineCriterion {
    /// `3 + (df - edf)^2`
    DfMatch { df: f64 },
    /// `(rss / sum w) / (1 - penalty edf / sum w)^2`
    Gcv { penalty: f64 },
}

/// The basis at each abscissa, reused between calls.
pub(crate) struct SplineWork {
    dx: Vec<f64>,
    left: Vec<u32>,
    basis: Vec<[f64; 4]>,
}

impl SplineWork {
    pub(crate) fn new(n: usize) -> EconResult<Self> {
        Ok(SplineWork {
            dx: resource::try_vec(n, 0.0, "spline smoother work")?,
            left: resource::try_vec(n, 0, "spline smoother work")?,
            basis: resource::try_vec(n, [0.0; 4], "spline smoother work")?,
        })
    }
}

/// The fit at the minimiser of the criterion.
#[derive(Debug, Clone)]
pub(crate) struct SplineFit {
    pub(crate) spar: f64,
    pub(crate) lambda: f64,
    pub(crate) ratio: f64,
    pub(crate) edf: f64,
    pub(crate) crit: f64,
}

/// `ppr.f`'s knots on `dx` in [0, 1]: `nk = min(n, 15)` coefficients, `nk + 4` knots, the inner
/// ones at the single-precision fraction `(n - 1) (i - 4) / (nk - 3)` of the sorted `dx`.
pub(crate) fn ppr_spline_knots(dx: &[f64]) -> Vec<f64> {
    let n = dx.len();
    let nk = n.min(NK_MAX);
    let mut knot = vec![dx[0]; nk + 4];
    knot[nk..nk + 4].fill(dx[n - 1]);
    for i in 5..=nk {
        let p = (((n - 1) * (i - 4)) as f32 / (nk - 3) as f32) as f64;
        let ip = p as usize;
        let f = p - ip as f64;
        knot[i - 1] = (1.0 - f) * dx[ip] + f * dx[ip + 1];
    }
    knot
}

/// The four cubic B-splines non-zero on `[t[left], t[left + 1])`, at `x` (Cox-de Boor).
#[inline]
fn bspline4(t: &[f64], left: usize, x: f64) -> [f64; 4] {
    let mut b = [1.0, 0.0, 0.0, 0.0];
    let (mut dl, mut dr) = ([0.0; 3], [0.0; 3]);
    for j in 0..3 {
        dr[j] = t[left + j + 1] - x;
        dl[j] = x - t[left - j];
        let mut saved = 0.0;
        for i in 0..=j {
            let term = b[i] / (dr[i] + dl[j - i]);
            b[i] = saved + dr[i] * term;
            saved = dl[j - i] * term;
        }
        b[j + 1] = saved;
    }
    b
}

/// Second derivatives at `x` of the four cubic B-splines of the interval `left`, with the
/// derivative recursion on the two linear B-splines there. Knots coincide only under a spline
/// that vanishes on this interval, whose term is 0.
fn bspline4_d2(t: &[f64], left: usize, x: f64) -> [f64; 4] {
    let ratio = |num: f64, den: f64| if den > 0.0 { num / den } else { 0.0 };
    let width = t[left + 1] - t[left];
    let b2 = [0.0, (t[left + 1] - x) / width, (x - t[left]) / width, 0.0];
    let mut d3 = [0.0; 5];
    for k in 0..3 {
        let i = left - 2 + k;
        d3[k + 1] = 2.0 * (ratio(b2[k], t[i + 2] - t[i]) - ratio(b2[k + 1], t[i + 3] - t[i + 1]));
    }
    let mut d2 = [0.0; 4];
    for k in 0..4 {
        let i = left - 3 + k;
        d2[k] = 3.0 * (ratio(d3[k], t[i + 3] - t[i]) - ratio(d3[k + 1], t[i + 4] - t[i + 1]));
    }
    d2
}

/// `Omega_ij = integral of B_i'' B_j''` over the knot range: B'' is linear on each interval, so
/// the integral of a product is `h (a_i a_j + (e_i a_j + e_j a_i) / 2 + e_i e_j / 3)`.
fn penalty_matrix(t: &[f64], nk: usize) -> Band {
    let mut omega = [[0.0; 4]; NK_MAX];
    for left in 3..nk {
        let h = t[left + 1] - t[left];
        if !(h > 0.0) {
            continue;
        }
        let a = bspline4_d2(t, left, t[left]);
        let c = bspline4_d2(t, left, t[left + 1]);
        for i in 0..4 {
            let ei = c[i] - a[i];
            for j in i..4 {
                let ej = c[j] - a[j];
                omega[left - 3 + i][j - i] +=
                    h * (a[i] * a[j] + (ei * a[j] + ej * a[i]) * 0.5 + ei * ej / 3.0);
            }
        }
    }
    omega
}

/// The accumulated normal equations of one smooth.
struct Problem {
    nk: usize,
    xwx: Band,
    omega: Band,
    xwy: [f64; NK_MAX],
    ratio: f64,
    sumw: f64,
}

/// A solve at one lambda.
struct Solved {
    coef: [f64; NK_MAX],
    edf: f64,
}

impl Problem {
    /// Fills `left` and `basis` for sorted `dx` and accumulates `X'WX`, `X'Wy` and `Omega`.
    fn new(
        dx: &[f64],
        y: &[f64],
        w: &[f64],
        knot: &[f64],
        left: &mut [u32],
        basis: &mut [[f64; 4]],
    ) -> EconResult<Problem> {
        let nk = knot.len() - 4;
        if nk < 6 {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "smoothing spline: the scale of the penalty needs at least 6 observations, \
                     got {nk}"
                ),
            });
        }
        let mut xwx = [[0.0; 4]; NK_MAX];
        let mut xwy = [0.0; NK_MAX];
        let mut sumw = 0.0;
        let mut l = 3;
        for i in 0..dx.len() {
            while l + 1 < nk && dx[i] >= knot[l + 1] {
                l += 1;
            }
            let b = bspline4(knot, l, dx[i]);
            left[i] = l as u32;
            basis[i] = b;
            let wi = w[i];
            sumw += wi;
            for r in 0..4 {
                let wb = wi * b[r];
                xwy[l - 3 + r] += wb * y[i];
                for s in r..4 {
                    xwx[l - 3 + r][s - r] += wb * b[s];
                }
            }
        }
        let omega = penalty_matrix(knot, nk);
        let (mut t1, mut t2) = (0.0, 0.0);
        for i in 2..nk - 3 {
            t1 += xwx[i][0];
            t2 += omega[i][0];
        }
        let ratio = t1 / t2;
        if !(ratio.is_finite() && ratio > 0.0) {
            return Err(EconError::InvalidSpecification {
                message: "smoothing spline: the abscissae have too few distinct values to scale \
                          the penalty"
                    .to_string(),
            });
        }
        Ok(Problem {
            nk,
            xwx,
            omega,
            xwy,
            ratio,
            sumw,
        })
    }

    fn lambda(&self, spar: f64) -> f64 {
        self.ratio * 256f64.powf(3.0 * spar - 1.0)
    }

    /// Banded Cholesky of `X'WX + lambda Omega`, coefficients, and the equivalent df
    /// `tr(M^-1 X'WX)`, which is the sum of the leverages, from the band of `M^-1`
    /// (Hutchinson and de Hoog 1985).
    fn solve(&self, lambda: f64) -> EconResult<Solved> {
        let nk = self.nk;
        let mut r = [[0.0; 4]; NK_MAX];
        for i in 0..nk {
            for d in 0..4.min(nk - i) {
                let j = i + d;
                let mut s = self.xwx[i][d] + lambda * self.omega[i][d];
                for k in j.saturating_sub(3)..i {
                    s -= r[k][i - k] * r[k][j - k];
                }
                if d > 0 {
                    r[i][d] = s / r[i][0];
                } else if s > 0.0 {
                    r[i][0] = s.sqrt();
                } else {
                    return Err(EconError::InvalidSpecification {
                        message: format!(
                            "smoothing spline: the penalised normal equations are not positive \
                             definite (pivot {}); the smoothing parameter is too small",
                            i + 1
                        ),
                    });
                }
            }
        }
        let mut coef = self.xwy;
        for i in 0..nk {
            let mut s = coef[i];
            for k in i.saturating_sub(3)..i {
                s -= r[k][i - k] * coef[k];
            }
            coef[i] = s / r[i][0];
        }
        for i in (0..nk).rev() {
            let mut s = coef[i];
            for d in 1..4.min(nk - i) {
                s -= r[i][d] * coef[i + d];
            }
            coef[i] = s / r[i][0];
        }
        // The band of S = M^-1: R S = R'^-1 gives, for j >= i, from the last row up,
        // R_ii S_ij = delta_ij / R_ii - sum_{k > i} R_ik S_kj.
        let mut inv = [[0.0; 4]; NK_MAX];
        for i in (0..nk).rev() {
            for d in (0..4.min(nk - i)).rev() {
                let j = i + d;
                let mut s = if d == 0 { 1.0 / r[i][0] } else { 0.0 };
                for e in 1..4.min(nk - i) {
                    let k = i + e;
                    let skj = if k <= j { inv[k][j - k] } else { inv[j][k - j] };
                    s -= r[i][e] * skj;
                }
                inv[i][d] = s / r[i][0];
            }
        }
        let mut edf = 0.0;
        for i in 0..nk {
            edf += inv[i][0] * self.xwx[i][0];
            for d in 1..4.min(nk - i) {
                edf += 2.0 * inv[i][d] * self.xwx[i][d];
            }
        }
        Ok(Solved { coef, edf })
    }

    fn criterion(
        &self,
        solved: &Solved,
        crit: SplineCriterion,
        y: &[f64],
        w: &[f64],
        left: &[u32],
        basis: &[[f64; 4]],
    ) -> f64 {
        match crit {
            SplineCriterion::DfMatch { df } => 3.0 + (df - solved.edf) * (df - solved.edf),
            SplineCriterion::Gcv { penalty } => {
                let mut rss = 0.0;
                for i in 0..y.len() {
                    let e = y[i] - fitted_at(&solved.coef, left[i] as usize, &basis[i]);
                    rss += w[i] * e * e;
                }
                let denom = 1.0 - penalty * solved.edf / self.sumw;
                (rss / self.sumw) / (denom * denom)
            }
        }
    }
}

#[inline]
fn fitted_at(coef: &[f64; NK_MAX], left: usize, b: &[f64; 4]) -> f64 {
    let c = &coef[left - 3..=left];
    b[0] * c[0] + b[1] * c[1] + b[2] * c[2] + b[3] * c[3]
}

/// Brent's minimiser on `spar` in [`LSPAR`, `USPAR`] with R's tolerances. Returns the minimiser.
fn brent_spar(mut f: impl FnMut(f64) -> EconResult<f64>) -> EconResult<f64> {
    const LSPAR: f64 = 0.0;
    const USPAR: f64 = 1.5;
    const TOL: f64 = 1e-2;
    const MAXIT: usize = 500;
    const BIG: f64 = 1e100;
    const C_GOLD: f64 = 0.381_966_011_250_105_15;
    let eps = 0.000244_f32 as f64;
    let finite = |v: f64| if v.is_finite() { v } else { 2.0 * BIG };
    let sign = |a: f64, b: f64| if b >= 0.0 { a.abs() } else { -a.abs() };

    let (mut a, mut b) = (LSPAR, USPAR);
    let mut x = a + C_GOLD * (b - a);
    let (mut v, mut w) = (x, x);
    let (mut d, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = finite(f(x)?);
    let (mut fv, mut fw) = (fx, fx);
    let mut iter = 0;
    loop {
        resource::checkpoint()?;
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + TOL / 3.0;
        let tol2 = tol1 * 2.0;
        iter += 1;
        if (x - xm).abs() <= tol2 - (b - a) * 0.5 || iter > MAXIT {
            return Ok(x);
        }
        let mut golden = e.abs() <= tol1 || fx >= BIG || fv >= BIG || fw >= BIG;
        if !golden {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let r = e;
            e = d;
            if p.abs() >= (0.5 * q * r).abs() || q == 0.0 || p <= q * (a - x) || p >= q * (b - x) {
                golden = true;
            } else {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = sign(tol1, xm - x);
                }
            }
        }
        if golden {
            e = if x >= xm { a - x } else { b - x };
            d = C_GOLD * e;
        }
        let u = x + if d.abs() >= tol1 { d } else { sign(tol1, d) };
        let fu = finite(f(u)?);
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            (v, fv) = (w, fw);
            (w, fw) = (x, fx);
            (x, fx) = (u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv) = (w, fw);
                (w, fw) = (u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
}

/// Smooths `y` on sorted `dx` in [0, 1] on `knot`, with the spar that minimises `crit`, into `smo`.
fn fit_spline(
    dx: &[f64],
    y: &[f64],
    w: &[f64],
    knot: &[f64],
    crit: SplineCriterion,
    left: &mut [u32],
    basis: &mut [[f64; 4]],
    smo: &mut [f64],
) -> EconResult<SplineFit> {
    let problem = Problem::new(dx, y, w, knot, left, basis)?;
    let spar = brent_spar(|spar| {
        let solved = problem.solve(problem.lambda(spar))?;
        Ok(problem.criterion(&solved, crit, y, w, left, basis))
    })?;
    let lambda = problem.lambda(spar);
    let solved = problem.solve(lambda)?;
    for i in 0..dx.len() {
        smo[i] = fitted_at(&solved.coef, left[i] as usize, &basis[i]);
    }
    Ok(SplineFit {
        spar,
        lambda,
        ratio: problem.ratio,
        edf: solved.edf,
        crit: problem.criterion(&solved, crit, y, w, left, basis),
    })
}

/// `ppr`'s spline smoother on sorted `x` with `x[n - 1] > x[0]`: df matching for
/// `SmoothingMethod::Spline`, GCV with `gcvpen` otherwise. Writes `smo`; returns the equivalent df.
pub(crate) fn ppr_spline(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    method: SmoothingMethod,
    df: f64,
    gcvpen: f64,
    smo: &mut [f64],
    work: &mut SplineWork,
) -> EconResult<f64> {
    let n = x.len();
    let range = x[n - 1] - x[0];
    for (d, &v) in work.dx.iter_mut().zip(x) {
        *d = (v - x[0]) / range;
    }
    let mut knot = [0.0; NK_MAX + 4];
    let nk = n.min(NK_MAX);
    knot[..nk + 4].copy_from_slice(&ppr_spline_knots(&work.dx));
    let crit = if method == SmoothingMethod::Spline {
        SplineCriterion::DfMatch { df }
    } else {
        SplineCriterion::Gcv { penalty: gcvpen }
    };
    let fit = fit_spline(
        &work.dx,
        y,
        w,
        &knot[..nk + 4],
        crit,
        &mut work.left,
        &mut work.basis,
        smo,
    )?;
    Ok(fit.edf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Case, Check, Tol};

    /// Sorted (dx, y) of `ppr_smooth`, with unit weights.
    fn smooth_data() -> (Case, Vec<f64>, Vec<f64>) {
        let (case, ds) = crate::test_support::load_case("ppr_smoothers");
        let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        let (x, y) = (col("x"), col("y"));
        let mut o: Vec<usize> = (0..x.len()).collect();
        o.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let (lo, hi) = (x[o[0]], x[o[o.len() - 1]]);
        let dx = o.iter().map(|&i| (x[i] - lo) / (hi - lo)).collect();
        let ys = o.iter().map(|&i| y[i]).collect();
        (case, dx, ys)
    }

    /// A problem on `n` points of a noisy curve.
    fn curve(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let dx: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y = dx
            .iter()
            .enumerate()
            .map(|(i, &v)| (5.0 * v).sin() + 0.1 * (((i * 7) % 5) as f64 - 2.0))
            .collect();
        let w = (0..n).map(|i| 1.0 + ((i * 3) % 4) as f64 * 0.5).collect();
        (dx, y, w)
    }

    fn problem(dx: &[f64], y: &[f64], w: &[f64]) -> (Problem, Vec<u32>, Vec<[f64; 4]>, Vec<f64>) {
        let knot = ppr_spline_knots(dx);
        let (mut left, mut basis) = (vec![0; dx.len()], vec![[0.0; 4]; dx.len()]);
        let p = Problem::new(dx, y, w, &knot, &mut left, &mut basis).unwrap();
        (p, left, basis, knot)
    }

    /// `Omega` against composite Simpson on 64 points per knot interval, on B'' itself.
    #[test]
    fn test_ppr_spline_penalty_is_exact() {
        let (dx, _, _) = curve(100);
        let knot = ppr_spline_knots(&dx);
        let nk = knot.len() - 4;
        let omega = penalty_matrix(&knot, nk);
        let mut simpson = [[0.0; 4]; NK_MAX];
        for left in 3..nk {
            let h = (knot[left + 1] - knot[left]) / 64.0;
            for s in 0..=64 {
                let weight = match s {
                    0 | 64 => 1.0,
                    s if s % 2 == 1 => 4.0,
                    _ => 2.0,
                } * h
                    / 3.0;
                let b = bspline4_d2(&knot, left, knot[left] + s as f64 * h);
                for i in 0..4 {
                    for j in i..4 {
                        simpson[left - 3 + i][j - i] += weight * b[i] * b[j];
                    }
                }
            }
        }
        for i in 0..nk {
            for d in 0..4.min(nk - i) {
                let (a, b) = (omega[i][d], simpson[i][d]);
                assert!(
                    (a - b).abs() <= 1e-12 * b.abs().max(1.0),
                    "{i},{d}: {a} {b}"
                );
            }
        }
        // B'' against second differences of the basis, inside one interval.
        let (left, x, h) = (7, 0.5 * (knot[7] + knot[8]), 1e-3 * (knot[8] - knot[7]));
        let (lo, mid, hi) = (
            bspline4(&knot, left, x - h),
            bspline4(&knot, left, x),
            bspline4(&knot, left, x + h),
        );
        let d2 = bspline4_d2(&knot, left, x);
        for i in 0..4 {
            let fd = (lo[i] - 2.0 * mid[i] + hi[i]) / (h * h);
            assert!(
                (fd - d2[i]).abs() <= 1e-4 * d2[i].abs().max(1.0),
                "{i}: {fd} {}",
                d2[i]
            );
        }
    }

    /// At a fixed lambda: the fit is the dense solve, edf is the trace of the smoother matrix
    /// built column by column, and a huge lambda leaves least squares on (1, x).
    #[test]
    fn test_ppr_spline_fit_is_consistent() {
        let (dx, y, w) = curve(60);
        let (p, left, basis, _) = problem(&dx, &y, &w);
        let nk = p.nk;
        let lambda = p.lambda(0.6);
        let solved = p.solve(lambda).unwrap();
        let mut dense = vec![vec![0.0; nk + 1]; nk];
        for i in 0..nk {
            for d in 0..4.min(nk - i) {
                let v = p.xwx[i][d] + lambda * p.omega[i][d];
                dense[i][i + d] = v;
                dense[i + d][i] = v;
            }
            dense[i][nk] = p.xwy[i];
        }
        for c in 0..nk {
            let piv = (c..nk)
                .max_by(|&a, &b| dense[a][c].abs().total_cmp(&dense[b][c].abs()))
                .unwrap();
            dense.swap(c, piv);
            for r in 0..nk {
                if r != c {
                    let f = dense[r][c] / dense[c][c];
                    for k in c..=nk {
                        let v = dense[c][k];
                        dense[r][k] -= f * v;
                    }
                }
            }
        }
        for c in 0..nk {
            let want = dense[c][nk] / dense[c][c];
            assert!((solved.coef[c] - want).abs() <= 1e-9 * want.abs().max(1.0));
        }
        // edf: solve for each unit response, read the fit back at that point.
        let mut trace = 0.0;
        for i in 0..dx.len() {
            let mut e = vec![0.0; dx.len()];
            e[i] = 1.0;
            let (q, l, b, _) = problem(&dx, &e, &w);
            let s = q.solve(lambda).unwrap();
            trace += fitted_at(&s.coef, l[i] as usize, &b[i]);
        }
        assert!(
            (solved.edf - trace).abs() <= 1e-10,
            "{} {trace}",
            solved.edf
        );
        // A large lambda: close to the penalty's null space, a weighted line.
        let big = p.solve(1e6).unwrap();
        let sw: f64 = w.iter().sum();
        let xb: f64 = dx.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
        let yb: f64 = y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / sw;
        let sxy: f64 = (0..60).map(|i| w[i] * (dx[i] - xb) * (y[i] - yb)).sum();
        let sxx: f64 = (0..60).map(|i| w[i] * (dx[i] - xb).powi(2)).sum();
        for i in 0..60 {
            let line = yb + sxy / sxx * (dx[i] - xb);
            let fit = fitted_at(&big.coef, left[i] as usize, &basis[i]);
            assert!((fit - line).abs() <= 1e-3, "{i}: {fit} {line}");
        }
    }

    /// The returned spar is within 0.02 of the argmin of the criterion on a fine grid.
    #[test]
    fn test_ppr_spline_brent_finds_the_minimum() {
        let (_, dx, y) = smooth_data();
        let n = dx.len();
        let w = vec![1.0; n];
        let (p, left, basis, knot) = problem(&dx, &y, &w);
        for crit in [
            SplineCriterion::DfMatch { df: 5.0 },
            SplineCriterion::Gcv { penalty: 1.0 },
            SplineCriterion::Gcv { penalty: 2.0 },
        ] {
            let (mut l, mut b, mut smo) = (vec![0; n], vec![[0.0; 4]; n], vec![0.0; n]);
            let fit = fit_spline(&dx, &y, &w, &knot, crit, &mut l, &mut b, &mut smo).unwrap();
            let (mut best, mut best_crit) = (0.0, f64::INFINITY);
            for k in 0..=1500 {
                let spar = k as f64 / 1000.0;
                let s = p.solve(p.lambda(spar)).unwrap();
                let c = p.criterion(&s, crit, &y, &w, &left, &basis);
                if c < best_crit {
                    (best, best_crit) = (spar, c);
                }
            }
            assert!(
                (fit.spar - best).abs() <= 0.02,
                "{crit:?}: {} {best}",
                fit.spar
            );
            // the reported df, lambda and criterion belong to the reported spar
            let s = p.solve(p.lambda(fit.spar)).unwrap();
            assert_eq!(fit.edf, s.edf);
            assert_eq!(fit.lambda, p.lambda(fit.spar));
            assert_eq!(fit.crit, p.criterion(&s, crit, &y, &w, &left, &basis));
        }
    }

    /// Fewer than 6 points cannot scale the penalty, and two distinct abscissae leave the normal
    /// equations without a usable trace.
    #[test]
    fn test_ppr_spline_refuses_what_it_cannot_scale() {
        let mut work = SplineWork::new(5).unwrap();
        let mut smo = [0.0; 5];
        let x = [0.0, 0.2, 0.4, 0.8, 1.0];
        let y = [1.0, 2.0, 1.5, 3.0, 2.5];
        let e = ppr_spline(
            &x,
            &y,
            &[1.0; 5],
            SmoothingMethod::Spline,
            3.0,
            1.0,
            &mut smo,
            &mut work,
        );
        assert!(e.unwrap_err().to_string().contains("at least 6"));
        let mut work = SplineWork::new(8).unwrap();
        let mut smo = [0.0; 8];
        let x = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 1.5, 3.0, 2.5, 2.0, 1.0, 0.5];
        let e = ppr_spline(
            &x,
            &y,
            &[1.0; 8],
            SmoothingMethod::GcvSpline,
            3.0,
            1.0,
            &mut smo,
            &mut work,
        );
        assert!(e.is_err());
    }

    /// Case `ppr_smoothers`, against `smooth.spline(all.knots = ppr knots)` with `ppr`'s
    /// `control.spar`. The knots are a convention and match to digits. Every other value differs
    /// by R's two quirks, so it is compared at a measured tolerance (10x the measured gap), and
    /// `iter` and `crit` not at all (path).
    #[test]
    fn test_validate_ppr_spline_smoother() {
        let (case, dx, y) = smooth_data();
        let n = dx.len();
        let w = vec![1.0; n];
        let knot = ppr_spline_knots(&dx);
        assert_eq!(knot.len(), 15 + 4);
        let knots13: Vec<f64> = knot[3..16].to_vec(); // c(0, inner, 1) as smooth.spline's all.knots
        let (p, left, basis, _) = problem(&dx, &y, &w);
        let (mut l, mut b, mut smo) = (vec![0; n], vec![[0.0; 4]; n], vec![0.0; n]);
        let mut check = Check::new(&case);
        check.vector("spline_knots", &knots13, Tol::coef(n));
        for (tag, crit) in [
            ("spline_df5", SplineCriterion::DfMatch { df: 5.0 }),
            ("spline_gcv", SplineCriterion::Gcv { penalty: 1.0 }),
            ("spline_gcv_pen2", SplineCriterion::Gcv { penalty: 2.0 }),
        ] {
            let f = fit_spline(&dx, &y, &w, &knot, crit, &mut l, &mut b, &mut smo).unwrap();
            // R's smooth, edf and lambda are of its last Brent evaluation; its spar is the
            // minimiser. 0.333 for 1/3 moves ratio and spar.
            check
                .scalar(&format!("{tag}_spar"), f.spar, Tol::iterative(SPAR_REL))
                .scalar(&format!("{tag}_ratio"), f.ratio, Tol::iterative(RATIO_REL))
                .vector(
                    &format!("{tag}_y"),
                    &smo,
                    Tol::new(LAST_EVAL_REL, LAST_EVAL_ABS),
                )
                .scalar(&format!("{tag}_edf"), f.edf, Tol::iterative(LAST_EVAL_REL));
            // R refitted at its own spar has no last-evaluation gap: this pair is 0.333 alone.
            let r_spar = case.scalar(&format!("{tag}_spar"));
            let s = p.solve(p.lambda(r_spar)).unwrap();
            let fit: Vec<f64> = (0..n)
                .map(|i| fitted_at(&s.coef, left[i] as usize, &basis[i]))
                .collect();
            check
                .vector(
                    &format!("{tag}_refit_y"),
                    &fit,
                    Tol::new(REFIT_REL, REFIT_ABS),
                )
                .scalar(
                    &format!("{tag}_refit_edf"),
                    s.edf,
                    Tol::iterative(REFIT_REL),
                );
        }
        check.finish();
    }

    // Measured in the test above; see validation/ml/ppr.md.
    const SPAR_REL: f64 = 5e-3;
    const RATIO_REL: f64 = 3e-2;
    const LAST_EVAL_REL: f64 = 0.15;
    const LAST_EVAL_ABS: f64 = 0.02;
    const REFIT_REL: f64 = 5e-3;
    const REFIT_ABS: f64 = 4e-3;
}
