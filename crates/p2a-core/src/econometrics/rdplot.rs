//! The RD plot of `rdrobust::rdplot` 4.0.0 (Calonico, Cattaneo & Titiunik 2015) with its
//! defaults: binned means of the outcome on evenly spaced bins on each side of the cutoff,
//! and a global polynomial of order `p` (4) fitted on each side with a uniform kernel over
//! the whole range.
//!
//! The number of bins on each side mimics the variance of the outcome (`binselect = "esmv"`):
//! `J = ceiling(var(y) / V * n / log(n)^2)`, with `V` the spacings estimator
//! `(1 / range) * sum(dx * dy^2) / 2` over the side's sorted observations. When at least 20 %
//! of a side's observations share a value with another one (mass points, rdplot's
//! `masspoints = "adjust"`), `V` is instead `(1 / range) * sum(dx * sigma2(x_mid))`, with
//! `sigma2` the conditional variance from global quartic regressions of `y` and `y^2` on `x`
//! (`"esmvpr"`); a negative estimate is replaced by `var(y)`. A side whose outcome does not
//! vary gets one bin. `nbins` sets the bins by hand.
//!
//! Bins follow rdplot exactly: the breaks are `seq(min(x), c, by = range / J)` (and from `c`
//! to `max(x)` on the right), observations are assigned with
//! `findInterval(..., rightmost.closed = TRUE)`, observations at the cutoff belong to the
//! right side, and empty bins are left out. One difference: the closing break of each side
//! is exactly the cutoff or the end of the data, where R's `seq` can fall one rounding step
//! short and put the last observation into an extra bin without a midpoint. The quartic regressions are fitted on a rescaled
//! `x`, which leaves their fitted values unchanged and keeps the fit well conditioned.
//! rdplot's options `covs`, `weights`, `h`, `kernel`, `support`, `scale`, `ci` and quantile
//! spaced bins have no counterpart here.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::least_squares;
use crate::resource;

/// Highest polynomial order (rdplot `p`).
pub const MAX_RDPLOT_ORDER: usize = 8;

/// Most bins on one side, by hand or selected.
pub const MAX_RDPLOT_BINS: usize = 10_000;

/// Points per side of the drawn polynomial (rdplot's `nplot`).
const CURVE_POINTS: usize = 500;

/// Options of [`rd_plot`]. `Default` is rdplot's: cutoff 0, order 4, bins selected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RdPlotConfig {
    /// The cutoff (rdplot `c`); must lie strictly inside the range of x.
    pub cutoff: f64,
    /// Order of the global polynomial on each side (rdplot `p`).
    pub p: usize,
    /// Bins left and right of the cutoff by hand (rdplot `nbins`); `None` selects them.
    pub nbins: Option<(usize, usize)>,
}

impl Default for RdPlotConfig {
    fn default() -> Self {
        Self {
            cutoff: 0.0,
            p: 4,
            nbins: None,
        }
    }
}

/// One non-empty bin.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RdPlotBin {
    /// rdplot's bin index: -J..-1 left of the cutoff, 1..J right of it.
    pub index: i64,
    /// Lower and upper break of the bin.
    pub lower: f64,
    pub upper: f64,
    /// Middle of the bin, where its dot is drawn (`rdplot_mean_bin`).
    pub mid: f64,
    /// Mean of x and of y in the bin (`rdplot_mean_x`, `rdplot_mean_y`).
    pub mean_x: f64,
    pub mean_y: f64,
    /// Observations in the bin (`rdplot_N`).
    pub n: usize,
}

/// How the bins were chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RdPlotBinSelect {
    /// Mimicking the variance, evenly spaced, spacings estimator (`esmv`).
    MimickingVariance,
    /// Mimicking the variance, evenly spaced, polynomial-regression estimator (`esmvpr`, mass
    /// points).
    MimickingVariancePolynomial,
    /// By hand (`nbins`).
    Manual,
}

impl RdPlotBinSelect {
    /// rdplot's name of the method.
    pub fn label(self) -> &'static str {
        match self {
            Self::MimickingVariance => "mimicking variance, evenly spaced (esmv)",
            Self::MimickingVariancePolynomial => {
                "mimicking variance, evenly spaced, polynomial regression (esmvpr; mass points)"
            }
            Self::Manual => "evenly spaced, set by hand",
        }
    }
}

/// The RD plot's numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdPlotResult {
    pub cutoff: f64,
    /// Order of the polynomial.
    pub p: usize,
    /// Observations left and right of the cutoff (rdplot `N`).
    pub n: (usize, usize),
    /// Bins left and right (rdplot `J`).
    pub bins_per_side: (usize, usize),
    pub bin_select: RdPlotBinSelect,
    /// The non-empty bins, left to right.
    pub bins: Vec<RdPlotBin>,
    /// Polynomial coefficients in powers of `x - cutoff`, left and right (rdplot `coef`).
    pub coef_left: Vec<f64>,
    pub coef_right: Vec<f64>,
    /// The polynomial at 500 points on each side (rdplot `vars_poly`).
    pub curve_left: Vec<(f64, f64)>,
    pub curve_right: Vec<(f64, f64)>,
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for RdPlotResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "RD plot (rdrobust::rdplot defaults)")?;
        writeln!(f, "Cutoff: {}", self.cutoff)?;
        writeln!(f, "Observations: {} left, {} right", self.n.0, self.n.1)?;
        writeln!(
            f,
            "Bins: {} left, {} right; {}",
            self.bins_per_side.0,
            self.bins_per_side.1,
            self.bin_select.label()
        )?;
        writeln!(
            f,
            "Global polynomial of order {} on each side (uniform kernel, whole range)",
            self.p
        )?;
        let coefs = |c: &[f64]| {
            c.iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        writeln!(
            f,
            "Coefficients left (powers of x - c): {}",
            coefs(&self.coef_left)
        )?;
        writeln!(
            f,
            "Coefficients right (powers of x - c): {}",
            coefs(&self.coef_right)
        )?;
        writeln!(
            f,
            "The gap between the curves at the cutoff is descriptive; estimate the effect with a \
             local polynomial (rd_estimate)."
        )?;
        for w in &self.warnings {
            writeln!(f, "Warning: {w}")?;
        }
        Ok(())
    }
}

/// [`rd_plot`] on two columns of a dataset; rows with a missing value in either are dropped
/// and counted, as rdplot's `complete.cases`.
pub fn run_rd_plot(
    dataset: &Dataset,
    outcome: &str,
    running_var: &str,
    config: RdPlotConfig,
) -> EconResult<RdPlotResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(outcome)
        .numeric(running_var)
        .na(NaPolicy::Omit)
        .extract()?;
    let mut result = rd_plot(cols.numeric(outcome), cols.numeric(running_var), config)?;
    result.missing = cols.missing().clone();
    if let Some(note) = result.missing.note() {
        result.warnings.push(note);
    }
    Ok(result)
}

/// The RD plot of `y` against `x` (finite values, same length); see the module docs.
pub fn rd_plot(y: &[f64], x: &[f64], config: RdPlotConfig) -> EconResult<RdPlotResult> {
    let c = config.cutoff;
    let p = config.p;
    if y.len() != x.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("y and x differ in length: {} and {}", y.len(), x.len()),
        });
    }
    if !c.is_finite() {
        return Err(EconError::InvalidSpecification {
            message: "the cutoff must be a finite number".into(),
        });
    }
    if p > MAX_RDPLOT_ORDER {
        return Err(EconError::InvalidSpecification {
            message: format!("p must be at most {MAX_RDPLOT_ORDER}, got {p}"),
        });
    }
    if let Some((l, r)) = config.nbins {
        if l == 0 || r == 0 || l > MAX_RDPLOT_BINS || r > MAX_RDPLOT_BINS {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "nbins must be between 1 and {MAX_RDPLOT_BINS} on each side, got {l} and {r}"
                ),
            });
        }
    }
    if x.len() < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: x.len(),
            context: "rdplot's bin selection needs at least 20 observations".into(),
        });
    }
    let x_min = x.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if c <= x_min || c >= x_max {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the cutoff {c} must lie inside the range of the running variable ({x_min} to \
                 {x_max})"
            ),
        });
    }

    let (mut xl, mut yl, mut xr, mut yr) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (&xi, &yi) in x.iter().zip(y) {
        if xi < c {
            xl.push(xi);
            yl.push(yi);
        } else {
            xr.push(xi);
            yr.push(yi);
        }
    }
    let need = (p + 1).max(3);
    for (side, n) in [("left", xl.len()), ("right", xr.len())] {
        if n < need {
            return Err(EconError::InsufficientData {
                required: need,
                provided: n,
                context: format!("observations {side} of the cutoff for a polynomial of order {p}"),
            });
        }
    }
    let n = x.len();
    let (range_l, range_r) = (c - x_min, x_max - c);
    let mut warnings = Vec::new();

    let coef_left = side_polynomial(&xl, &yl, c, p)?;
    let coef_right = side_polynomial(&xr, &yr, c, p)?;
    let curve = |coef: &[f64], from: f64, to: f64| -> Vec<(f64, f64)> {
        (0..CURVE_POINTS)
            .map(|i| {
                let t = from + (to - from) * i as f64 / (CURVE_POINTS - 1) as f64;
                (t, horner(coef, t - c))
            })
            .collect()
    };
    let curve_left = curve(&coef_left, c - range_l, c);
    let curve_right = curve(&coef_right, c, c + range_r);

    let var_l = sample_variance(&yl);
    let var_r = sample_variance(&yr);
    let mass = |xs: &[f64]| {
        let mut sorted = xs.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        1.0 - sorted.len() as f64 / xs.len() as f64 >= 0.2
    };

    let (mut j_l, mut j_r, bin_select) = match config.nbins {
        Some((l, r)) => (l, r, RdPlotBinSelect::Manual),
        None => {
            let polynomial = mass(&xl) || mass(&xr);
            if polynomial {
                warnings.push(
                    "Mass points in the running variable: the bins use rdplot's polynomial \
                     regression variance (esmvpr)."
                        .into(),
                );
            }
            let n_f = n as f64;
            let scale = n_f / n_f.ln().powi(2);
            let pick = |xs: &[f64], ys: &[f64], var: f64, range: f64, side: &str| {
                let v = if polynomial {
                    variance_polynomial(xs, ys, var, range)
                } else {
                    variance_spacings(xs, ys, range)
                }?;
                bins_from(var / v * scale, side)
            };
            let l = pick(&xl, &yl, var_l, range_l, "left")?;
            let r = pick(&xr, &yr, var_r, range_r, "right")?;
            let select = if polynomial {
                RdPlotBinSelect::MimickingVariancePolynomial
            } else {
                RdPlotBinSelect::MimickingVariance
            };
            (l, r, select)
        }
    };
    if var_l == 0.0 {
        j_l = 1;
        warnings.push("The outcome does not vary below the cutoff: one bin there.".into());
    }
    if var_r == 0.0 {
        j_r = 1;
        warnings.push("The outcome does not vary above the cutoff: one bin there.".into());
    }

    let mut bins = side_bins(&xl, &yl, x_min, c, j_l, true)?;
    bins.extend(side_bins(&xr, &yr, c, x_max, j_r, false)?);

    Ok(RdPlotResult {
        cutoff: c,
        p,
        n: (xl.len(), xr.len()),
        bins_per_side: (j_l, j_r),
        bin_select,
        bins,
        coef_left,
        coef_right,
        curve_left,
        curve_right,
        warnings,
        missing: MissingSummary::default(),
    })
}

/// The bin count from rdplot's ceiling formula, refused when it is not a usable number.
fn bins_from(j: f64, side: &str) -> EconResult<usize> {
    let j = j.ceil();
    if !j.is_finite() || j < 1.0 || j > MAX_RDPLOT_BINS as f64 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "the variance-mimicking bin count {side} of the cutoff is {j}, outside 1 to \
                 {MAX_RDPLOT_BINS} (the outcome barely changes between neighbouring values of \
                 the running variable); set nbins by hand"
            ),
        });
    }
    Ok(j as usize)
}

/// Sample variance (n - 1); 0 for a constant side.
fn sample_variance(v: &[f64]) -> f64 {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

/// The side's observations sorted by x (ties in data order, as R's `order`).
fn sorted_pairs(xs: &[f64], ys: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, f64)> = xs.iter().copied().zip(ys.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    pairs
}

/// rdplot's `V_es_hat_dw`: `(0.5 / range) * sum(dx * dy^2)` over consecutive sorted pairs.
fn variance_spacings(xs: &[f64], ys: &[f64], range: f64) -> EconResult<f64> {
    let pairs = sorted_pairs(xs, ys);
    let sum: f64 = pairs
        .windows(2)
        .map(|w| (w[1].0 - w[0].0) * (w[1].1 - w[0].1).powi(2))
        .sum();
    Ok(0.5 / range * sum)
}

/// rdplot's `V_es_chk_dw`: `(1 / range) * sum(dx * sigma2(x_mid))`, with `sigma2` from global
/// quartic fits of y and y^2 on x; negative estimates become `var`.
fn variance_polynomial(xs: &[f64], ys: &[f64], var: f64, range: f64) -> EconResult<f64> {
    let pairs = sorted_pairs(xs, ys);
    let (lo, hi) = (pairs[0].0, pairs[pairs.len() - 1].0);
    let (mid, half) = ((lo + hi) / 2.0, ((hi - lo) / 2.0).max(f64::MIN_POSITIVE));
    let t = |v: f64| (v - mid) / half;
    let k = 4.min(pairs.len() - 1);
    let design = powers(pairs.iter().map(|q| t(q.0)), k)?;
    let y1 = Array1::from_iter(pairs.iter().map(|q| q.1));
    let y2 = y1.mapv(|v| v * v);
    // An aliased power (few distinct values) drops out of the fit, as under rdplot's
    // generalized inverse.
    let zeroed = |c: Array1<f64>| c.mapv(|v| if v.is_nan() { 0.0 } else { v });
    let g1 = zeroed(least_squares(design.view(), y1.view())?.coefficients);
    let g2 = zeroed(least_squares(design.view(), y2.view())?.coefficients);
    let mut sum = 0.0;
    for w in pairs.windows(2) {
        let at = t((w[0].0 + w[1].0) / 2.0);
        let m1 = horner(g1.as_slice().unwrap_or(&[]), at);
        let m2 = horner(g2.as_slice().unwrap_or(&[]), at);
        let s2 = m2 - m1 * m1;
        sum += (w[1].0 - w[0].0) * if s2 < 0.0 { var } else { s2 };
    }
    Ok(sum / range)
}

/// The design of powers 0..=k of the values.
fn powers(values: impl Iterator<Item = f64>, k: usize) -> EconResult<Array2<f64>> {
    let values: Vec<f64> = values.collect();
    let mut out = resource::try_zeros2(values.len(), k + 1, "polynomial design")?;
    for (i, v) in values.iter().enumerate() {
        let mut acc = 1.0;
        for j in 0..=k {
            out[[i, j]] = acc;
            acc *= v;
        }
    }
    Ok(out)
}

/// The order-p polynomial of y in powers of `x - c` over the whole side (uniform kernel).
fn side_polynomial(xs: &[f64], ys: &[f64], c: f64, p: usize) -> EconResult<Vec<f64>> {
    let design = powers(xs.iter().map(|v| v - c), p)?;
    let fit = least_squares(design.view(), Array1::from(ys.to_vec()).view())?;
    if fit.rank < p + 1 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "a polynomial of order {p} cannot be fitted on one side of the cutoff (too few \
                 distinct values of the running variable); lower p"
            ),
        });
    }
    Ok(fit.coefficients.to_vec())
}

/// `sum(coef[j] * t^j)`.
fn horner(coef: &[f64], t: f64) -> f64 {
    coef.iter().rev().fold(0.0, |acc, &b| acc * t + b)
}

/// The non-empty bins of one side, left to right: breaks `seq(from, to, by = (to - from) / j)`
/// (capped at `to`, as R's `seq`), assignment by `findInterval(rightmost.closed = TRUE)`.
fn side_bins(
    xs: &[f64],
    ys: &[f64],
    from: f64,
    to: f64,
    j: usize,
    left: bool,
) -> EconResult<Vec<RdPlotBin>> {
    let by = (to - from) / j as f64;
    // R's seq(from, to, by): n = floor((to - from) / by + 1e-10) breaks after the first.
    let last = (((to - from) / by) + 1e-10).floor() as usize;
    // The closing break is `to` itself: R's `seq` can stop one rounding step short of it,
    // which leaves the side's last observation in an extra bin with no midpoint.
    let brk = |i: usize| {
        if i >= j {
            to
        } else {
            (from + i as f64 * by).min(to)
        }
    };
    // findInterval: the number of breaks <= v; the last break closes the last interval.
    let interval = |v: f64| -> usize {
        let mut k = (((v - from) / by).floor().max(0.0) as usize).min(last);
        while k < last && brk(k + 1) <= v {
            k += 1;
        }
        while k > 0 && brk(k) > v {
            k -= 1;
        }
        let count = if brk(0) <= v { k + 1 } else { 0 };
        if count == last + 1 && v == brk(last) {
            last
        } else {
            count
        }
    };
    let mut acc: std::collections::BTreeMap<i64, (f64, f64, usize)> = Default::default();
    for (i, (&xv, &yv)) in xs.iter().zip(ys).enumerate() {
        if i % 65_536 == 0 {
            resource::checkpoint()?;
        }
        let count = interval(xv) as i64;
        let index = if left { count - j as i64 - 1 } else { count };
        let e = acc.entry(index).or_insert((0.0, 0.0, 0));
        e.0 += xv;
        e.1 += yv;
        e.2 += 1;
    }
    Ok(acc
        .into_iter()
        .map(|(index, (sx, sy, cnt))| {
            // The bin's position among the breaks (0-based lower break).
            let t = if left { index + j as i64 } else { index - 1 };
            let lower = if t >= 0 { brk(t as usize) } else { f64::NAN };
            let upper = if t >= 0 && (t as usize) < last {
                brk(t as usize + 1)
            } else {
                f64::NAN
            };
            RdPlotBin {
                index,
                lower,
                upper,
                mid: (lower + upper) / 2.0,
                mean_x: sx / cnt as f64,
                mean_y: sy / cnt as f64,
                n: cnt,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    /// Every number rdplot reports, for its defaults (esmv), mass points (esmvpr), bins by
    /// hand, p = 2 and a cutoff off zero. R's extra one-point bin past its last break is
    /// compared merged into the last bin (`merged` values; see the case file).
    #[test]
    fn test_validate_rdplot_sim() {
        let (case, ds) = load_case("rdplot_sim");
        let mut check = Check::new(&case);
        let n = 1000;
        let runs: [(&str, &str, &str, f64, usize, Option<(usize, usize)>); 5] = [
            ("default", "y", "x", 0.0, 4, None),
            ("mass", "ym", "xm", 0.0, 4, None),
            ("nbins", "y", "x", 0.0, 4, Some((10, 12))),
            ("p2", "y", "x", 0.0, 2, None),
            ("c02", "y", "x", 0.2, 4, None),
        ];
        for (prefix, y, x, cutoff, p, nbins) in runs {
            let mut config = RdPlotConfig::default();
            config.cutoff = cutoff;
            config.p = p;
            config.nbins = nbins;
            let r = run_rd_plot(&ds, y, x, config).unwrap();
            let k = |s: &str| format!("{prefix}_{s}");
            let bins = |f: fn(&RdPlotBin) -> f64| r.bins.iter().map(f).collect::<Vec<_>>();
            let (cx, cy): (Vec<_>, Vec<_>) =
                r.curve_left.iter().chain(&r.curve_right).copied().unzip();
            let pick = |v: &[f64]| [0, 249, 499, 500, 749, 999].map(|i| v[i]).to_vec();
            check
                .vector(
                    &k("j"),
                    &[r.bins_per_side.0 as f64, r.bins_per_side.1 as f64],
                    Tol::exact(),
                )
                .vector(&k("n"), &[r.n.0 as f64, r.n.1 as f64], Tol::exact())
                .vector(&k("coef_left"), &r.coef_left, Tol::coef(n))
                .vector(&k("coef_right"), &r.coef_right, Tol::coef(n))
                .vector(&k("bin_mid"), &bins(|b| b.mid), Tol::coef(n))
                .vector(&k("bin_mean_x"), &bins(|b| b.mean_x), Tol::coef(n))
                .vector(&k("bin_mean_y"), &bins(|b| b.mean_y), Tol::coef(n))
                .vector(&k("bin_n"), &bins(|b| b.n as f64), Tol::exact())
                .vector(&k("bin_lower"), &bins(|b| b.lower), Tol::coef(n))
                .vector(&k("bin_upper"), &bins(|b| b.upper), Tol::coef(n))
                .vector(&k("curve_x"), &pick(&cx), Tol::coef(n))
                .vector(&k("curve_y"), &pick(&cy), Tol::coef(n));
            // deviation rdplot-last-break: R's raw bins hold an extra one-point bin without a
            // midpoint past its last break; p2a's last bin closes at max(x), so it matches the
            // merged bins, and has the raw bin count less the bins without a midpoint.
            let raw = case.scalar(&k("raw_bins")) - case.scalar(&k("raw_na_bins"));
            assert_eq!(r.bins.len() as f64, raw, "{prefix}");
            let mass = prefix == "mass";
            assert_eq!(
                r.bin_select == RdPlotBinSelect::MimickingVariancePolynomial,
                mass,
                "{prefix}"
            );
        }
        check.finish();
    }

    fn data(n: usize, ties: bool) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n)
            .map(|i| {
                let u = ((i as f64 * 0.618_033_988_7) % 1.0) * 2.0 - 1.0;
                if ties { (u * 10.0).round() / 10.0 } else { u }
            })
            .collect();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| v + 0.5 * f64::from(v >= 0.0) + 0.3 * ((i as f64) * 1.3).sin())
            .collect();
        (y, x)
    }

    /// Every observation lands in one bin; bins cover their side and are ordered.
    #[test]
    fn bins_hold_every_observation() {
        let (y, x) = data(500, false);
        let r = rd_plot(&y, &x, RdPlotConfig::default()).unwrap();
        assert_eq!(r.bins.iter().map(|b| b.n).sum::<usize>(), 500);
        assert!(r.bins.windows(2).all(|w| w[0].index < w[1].index));
        assert!(r.bins.iter().all(|b| b.index != 0));
        for b in &r.bins {
            assert!(
                b.lower <= b.mean_x && b.mean_x <= b.upper,
                "{b:?} {:?}",
                r.bins_per_side
            );
        }
        assert_eq!(r.bin_select, RdPlotBinSelect::MimickingVariance);
        assert_eq!(r.coef_left.len(), 5);
        assert_eq!(r.curve_left.len(), CURVE_POINTS);
    }

    /// Many ties switch to the polynomial variance; manual bins are used as given.
    #[test]
    fn mass_points_and_manual_bins() {
        let (y, x) = data(500, true);
        let r = rd_plot(&y, &x, RdPlotConfig::default()).unwrap();
        assert_eq!(r.bin_select, RdPlotBinSelect::MimickingVariancePolynomial);
        let mut config = RdPlotConfig::default();
        config.nbins = Some((7, 9));
        let r = rd_plot(&y, &x, config).unwrap();
        assert_eq!(r.bins_per_side, (7, 9));
        assert!(r.bins.len() <= 16);
    }

    /// Bad cutoffs, orders and bin counts are refused with their bound.
    #[test]
    fn options_are_checked() {
        let (y, x) = data(200, false);
        let mut config = RdPlotConfig::default();
        config.cutoff = 5.0;
        assert!(rd_plot(&y, &x, config).is_err());
        let mut config = RdPlotConfig::default();
        config.p = MAX_RDPLOT_ORDER + 1;
        assert!(rd_plot(&y, &x, config).is_err());
        let mut config = RdPlotConfig::default();
        config.nbins = Some((0, 3));
        assert!(rd_plot(&y, &x, config).is_err());
        assert!(rd_plot(&y[..10], &x[..10], RdPlotConfig::default()).is_err());
    }
}
