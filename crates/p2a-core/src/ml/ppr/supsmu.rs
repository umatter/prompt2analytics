//! Friedman's super smoother on sorted abscissae, the ridge smoother of `ppr`.
//!
//! The routine R's `ppr.f` calls as `supsmu`/`smooth` and `stats::supsmu` calls too (Friedman
//! 1984): running lines at the tweeter, midrange and woofer spans, their leave-one-out
//! residuals smoothed, the best span per point moved toward the woofer by `bass`, and a final
//! smooth of the interpolated fits. Unlike `regression::supsmu` it keeps tied x as separate
//! points (their fits are averaged, not merged), works in caller-owned buffers, and has no
//! periodic windows, which `ppr` never asks for. The two are transcriptions of one Fortran
//! routine and agree to rounding; they are to be merged into one kernel.

use crate::errors::EconResult;
use crate::resource;

/// Tweeter, midrange and woofer spans: REAL literals of the Fortran's block data.
pub(crate) const SPANS: [f64; 3] = [0.05_f32 as f64, 0.2_f32 as f64, 0.5];
const BIG: f64 = 1.0e20_f32 as f64;
const SML: f64 = 1.0e-7_f32 as f64;
const EPS: f64 = 1.0e-3_f32 as f64;

/// Seven work columns (the Fortran's `sc`), reused between calls.
pub(crate) struct SupsmuWork {
    sc: [Vec<f64>; 7],
}

impl SupsmuWork {
    pub(crate) fn new(n: usize) -> EconResult<Self> {
        let col = || resource::try_vec(n, 0.0, "super smoother work");
        Ok(SupsmuWork {
            sc: [col()?, col()?, col()?, col()?, col()?, col()?, col()?],
        })
    }
}

/// Super smoother of `y` on sorted `x` into `smo`. `span = 0` is the cross-validated span;
/// `span` in (0, 1] is a fixed span, for which `bass` has no effect.
pub(crate) fn supsmu_sorted(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    span: f64,
    bass: f64,
    smo: &mut [f64],
    work: &mut SupsmuWork,
) {
    let n = x.len();
    if !x[n - 1].gt(&x[0]) {
        let (mut sy, mut sw) = (0.0, 0.0);
        for j in 0..n {
            sy += w[j] * y[j];
            sw += w[j];
        }
        let a = if sw > 0.0 { sy / sw } else { 0.0 };
        smo.fill(a);
        return;
    }
    // Interquartile range of x; it grows until it is positive, which x[n - 1] > x[0] guarantees.
    let mut i = n / 4;
    let mut j = 3 * i;
    let mut scale = x[j - 1] - x[i - 1];
    while !scale.gt(&0.0) {
        if j < n {
            j += 1;
        }
        if i > 1 {
            i -= 1;
        }
        scale = x[j - 1] - x[i - 1];
    }
    let vsmlsq = (EPS * scale) * (EPS * scale);
    if span > 0.0 {
        smooth(x, y, w, span, vsmlsq, smo, None);
        return;
    }

    let [sc1, sc2, sc3, sc4, sc5, sc6, sc7] = &mut work.sc;
    smooth(x, y, w, SPANS[0], vsmlsq, sc1, Some(sc7));
    smooth(x, sc7, w, SPANS[1], vsmlsq, sc2, None);
    smooth(x, y, w, SPANS[1], vsmlsq, sc3, Some(sc7));
    smooth(x, sc7, w, SPANS[1], vsmlsq, sc4, None);
    smooth(x, y, w, SPANS[2], vsmlsq, sc5, Some(sc7));
    smooth(x, sc7, w, SPANS[1], vsmlsq, sc6, None);
    for jj in 0..n {
        let mut resmin = BIG;
        for (res, s) in [
            (sc2[jj], SPANS[0]),
            (sc4[jj], SPANS[1]),
            (sc6[jj], SPANS[2]),
        ] {
            if res < resmin {
                resmin = res;
                sc7[jj] = s;
            }
        }
        if bass > 0.0 && bass <= 10.0 && resmin < sc6[jj] && resmin > 0.0 {
            sc7[jj] += (SPANS[2] - sc7[jj]) * SML.max(resmin / sc6[jj]).powf(10.0 - bass);
        }
    }
    smooth(x, sc7, w, SPANS[1], vsmlsq, sc2, None);
    for jj in 0..n {
        sc2[jj] = sc2[jj].clamp(SPANS[0], SPANS[2]);
        let f = sc2[jj] - SPANS[1];
        sc4[jj] = if f < 0.0 {
            let f = -f / (SPANS[1] - SPANS[0]);
            (1.0 - f) * sc3[jj] + f * sc1[jj]
        } else {
            let f = f / (SPANS[2] - SPANS[1]);
            (1.0 - f) * sc3[jj] + f * sc5[jj]
        };
    }
    smooth(x, sc4, w, SPANS[0], vsmlsq, smo, None);
}

/// Running lines of `y` on sorted `x`, in windows of `2 ibw + 1` points shifted at the ends, with
/// Friedman's updating formulas. `acvr` receives the exact leave-one-out absolute residuals.
/// Tied x get the weighted mean of their fits.
fn smooth(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    span: f64,
    vsmlsq: f64,
    smo: &mut [f64],
    mut acvr: Option<&mut [f64]>,
) {
    let n = x.len();
    let (mut xm, mut ym, mut var, mut cvar, mut fbw) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let ibw = ((0.5 * span * n as f64 + 0.5) as usize).max(2);
    for i in 0..(2 * ibw + 1).min(n) {
        let wt = w[i];
        let fbo = fbw;
        fbw += wt;
        if fbw > 0.0 {
            xm = (fbo * xm + wt * x[i]) / fbw;
            ym = (fbo * ym + wt * y[i]) / fbw;
        }
        let tmp = if fbo > 0.0 {
            fbw * wt * (x[i] - xm) / fbo
        } else {
            0.0
        };
        var += tmp * (x[i] - xm);
        cvar += tmp * (y[i] - ym);
    }
    for j in 0..n {
        if j > ibw && j + ibw < n {
            let (out, inn) = (j - ibw - 1, j + ibw);
            // the point leaving the window
            let wt = w[out];
            let fbo = fbw;
            fbw -= wt;
            let tmp = if fbw > 0.0 {
                fbo * wt * (x[out] - xm) / fbw
            } else {
                0.0
            };
            var -= tmp * (x[out] - xm);
            cvar -= tmp * (y[out] - ym);
            if fbw > 0.0 {
                xm = (fbo * xm - wt * x[out]) / fbw;
                ym = (fbo * ym - wt * y[out]) / fbw;
            }
            // the point entering it
            let wt = w[inn];
            let fbo = fbw;
            fbw += wt;
            if fbw > 0.0 {
                xm = (fbo * xm + wt * x[inn]) / fbw;
                ym = (fbo * ym + wt * y[inn]) / fbw;
            }
            let tmp = if fbo > 0.0 {
                fbw * wt * (x[inn] - xm) / fbo
            } else {
                0.0
            };
            var += tmp * (x[inn] - xm);
            cvar += tmp * (y[inn] - ym);
        }
        let a = if var > vsmlsq { cvar / var } else { 0.0 };
        smo[j] = a * (x[j] - xm) + ym;
        if let Some(acvr) = acvr.as_deref_mut() {
            let mut h = if fbw > 0.0 { 1.0 / fbw } else { 0.0 };
            if var > vsmlsq {
                h += (x[j] - xm) * (x[j] - xm) / var;
            }
            let a = 1.0 - w[j] * h;
            acvr[j] = if a > 0.0 {
                (y[j] - smo[j]).abs() / a
            } else if j > 0 {
                acvr[j - 1]
            } else {
                0.0
            };
        }
    }
    let mut j = 0;
    while j < n {
        let j0 = j;
        let mut sy = smo[j] * w[j];
        let mut sw = w[j];
        while j + 1 < n && x[j + 1] <= x[j] {
            j += 1;
            sy += w[j] * smo[j];
            sw += w[j];
        }
        if j > j0 {
            let a = if sw > 0.0 { sy / sw } else { 0.0 };
            smo[j0..=j].fill(a);
        }
        j += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::columns::{NaPolicy, numeric_column};
    use p2a_validation::{Check, Tol};

    /// Case `ppr_smoothers`: `stats::supsmu` calls the same Fortran routine as `ppr` (ML-34).
    #[test]
    fn test_validate_ppr_supsmu_smoother() {
        let (case, ds) = crate::test_support::load_case("ppr_smoothers");
        let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        let (x, y, w) = (col("x"), col("y"), col("w"));
        let mut o: Vec<usize> = (0..x.len()).collect();
        o.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let xs: Vec<f64> = o.iter().map(|&i| x[i]).collect();
        let ys: Vec<f64> = o.iter().map(|&i| y[i]).collect();
        let ws: Vec<f64> = o.iter().map(|&i| w[i]).collect();
        let n = xs.len();
        let ones = vec![1.0; n];
        let mut work = SupsmuWork::new(n).unwrap();
        let mut run = |w: &[f64], span: f64, bass: f64| {
            let mut smo = vec![0.0; n];
            supsmu_sorted(&xs, &ys, w, span, bass, &mut smo, &mut work);
            smo
        };
        let (cv, b5, s3, wt) = (
            run(&ones, 0.0, 0.0),
            run(&ones, 0.0, 5.0),
            run(&ones, 0.3, 0.0),
            run(&ws, 0.0, 0.0),
        );
        let mut c = Check::new(&case);
        c.vector("supsmu_cv", &cv, Tol::coef(n))
            .vector("supsmu_bass5", &b5, Tol::coef(n))
            .vector("supsmu_span03", &s3, Tol::coef(n))
            .vector("supsmu_weighted", &wt, Tol::coef(n));
        c.finish();
    }

    /// Tied x get one fit, and constant x the weighted mean (the branch `ppr.f` takes first).
    #[test]
    fn test_ppr_supsmu_ties_and_constant_x() {
        let x = [0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 4.0, 5.0, 6.0];
        let y = [0.1, 1.2, 0.8, 2.1, 2.9, 3.2, 3.0, 4.1, 4.9, 6.2];
        let w = [1.0, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0];
        let mut work = SupsmuWork::new(10).unwrap();
        let mut smo = [0.0; 10];
        supsmu_sorted(&x, &y, &w, 0.0, 0.0, &mut smo, &mut work);
        assert_eq!(smo[1], smo[2]);
        assert_eq!(smo[4], smo[6]);
        supsmu_sorted(&[2.0; 10], &y, &w, 0.0, 0.0, &mut smo, &mut work);
        let mean = y.iter().zip(&w).map(|(a, b)| a * b).sum::<f64>() / w.iter().sum::<f64>();
        assert!(smo.iter().all(|&s| (s - mean).abs() < 1e-14));
    }
}
