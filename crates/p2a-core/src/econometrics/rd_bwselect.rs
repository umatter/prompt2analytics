//! Bandwidth selection for local polynomial RD as `rdrobust::rdbwselect` 4.0.0 (Calonico,
//! Cattaneo & Farrell 2020; Calonico, Cattaneo & Titiunik 2014): MSE- and CER-optimal main
//! (h) and bias (b) bandwidths, all ten `bwselect` methods, sharp and fuzzy designs.
//!
//! The plug-in bandwidths are estimators defined by their algorithm, so this module is a port
//! of rdbwselect's algorithm:
//!
//! 1. The pilot bandwidth `c_bw = min(C_c * min(sd(x), IQR(x) / 1.349) * M^(-1/5), bw_max)`,
//!    with `C_c` the kernel's rule-of-thumb constant ([`KernelType::constant()`]), the
//!    interquartile range of quantile type 2, `M` the number of distinct values of the
//!    running variable and `bw_max` the larger distance from the cutoff to the end of the data.
//! 2. On each side, three pilot stages (orders `q + 1`, `q` and `p`), each with a variance
//!    term `V` from a local fit of order `o` under `c_bw`, a bias term `B` from a fit of
//!    order `o + 1` under the previous stage's bandwidth, and a regularisation term `R`
//!    (weighted by `scaleregul`). Every stage's bandwidth is `(V / (B^2 + scaleregul R))^(1 / (2o + 3))`.
//! 3. The `rd` family (difference of the two sides' bias terms), the `sum` family (their sum)
//!    and the `two` family (each side alone); `comb1` is the smaller of rd and sum, `comb2`
//!    the per-side median of rd, sum and two.
//! 4. A CER method scales its MSE method's h by `N^(-p / ((3 + p)(3 + 2p)))`; b is unchanged.
//!
//! Bandwidths are restricted to the range of the data (rdbwselect's `bwrestrict`). When at
//! least 20 % of a side's observations share a value with another one (mass points), the
//! pilot and the first-stage bandwidths are kept at least as wide as the distance to the
//! 10th distinct value on each side (`masspoints = "adjust"`, `bwcheck = 10`). A fuzzy design
//! whose treatment does not vary on one side (one-sided compliance) is selected as a sharp
//! design on the outcome, as rdbwselect does. A singular pilot fit uses a generalized inverse,
//! as rdbwselect's `ginv` fallback, with a warning. rdbwselect's options `covs`, `cluster`,
//! `weights`, `deriv`, `stdvars`, `sharpbw`, `bwcheck` and `bwrestrict = FALSE` have no
//! counterpart here.

use ndarray::{Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};

use super::rd::{
    BandwidthMethod, KernelType, MAX_RD_ORDER, RdConfig, VceType, hc_residuals, nn_residuals,
    validate_scaleregul,
};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};

/// rdbwselect's tuning options. `Default` = rdbwselect's defaults (p = 1, q = p + 1,
/// triangular kernel, nearest-neighbour residuals with 3 matches, scaleregul = 1).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BwSelectOptions {
    /// Order of the local polynomial of the point estimate (rdbwselect `p`).
    pub p: usize,
    /// Bias-correction order; `None` = p + 1. Must exceed p.
    pub q: Option<usize>,
    /// Kernel (rdbwselect `kernel`).
    pub kernel: KernelType,
    /// Residuals of the variance terms (rdbwselect `vce`).
    pub vce: VceType,
    /// Nearest neighbours of the `nn` residuals (rdbwselect `nnmatch`).
    pub nnmatch: usize,
    /// Weight of the regularisation term R (rdbwselect `scaleregul`); 0 switches it off.
    pub scaleregul: f64,
}

impl Default for BwSelectOptions {
    fn default() -> Self {
        Self {
            p: 1,
            q: None,
            kernel: KernelType::Triangular,
            vce: VceType::Nn,
            nnmatch: 3,
            scaleregul: 1.0,
        }
    }
}

impl BwSelectOptions {
    /// The selection options of an RD configuration: p, q, kernel, vce, nnmatch, scaleregul.
    pub fn from_config(config: &RdConfig) -> Self {
        Self {
            p: config.p,
            q: config.q,
            kernel: config.kernel,
            vce: config.vce,
            nnmatch: config.nnmatch,
            scaleregul: config.scaleregul,
        }
    }
}

/// One row of rdbwselect's `bws`: h left, h right, b left, b right.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bandwidths {
    /// Main bandwidth left of the cutoff.
    pub h_left: f64,
    /// Main bandwidth right of the cutoff.
    pub h_right: f64,
    /// Bias bandwidth left of the cutoff.
    pub b_left: f64,
    /// Bias bandwidth right of the cutoff.
    pub b_right: f64,
}

/// What rdbwselect reports besides the bandwidths, plus p2a's warnings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BwSelectInfo {
    /// Observations left of the cutoff (rdbwselect `N[1]`).
    pub n_left: usize,
    /// Observations right of the cutoff (rdbwselect `N[2]`).
    pub n_right: usize,
    /// Distinct running-variable values left of the cutoff (rdbwselect `M[1]`).
    pub m_left: usize,
    /// Distinct running-variable values right of the cutoff (rdbwselect `M[2]`).
    pub m_right: usize,
    /// `1 - M / N >= 0.2` on a side.
    pub mass_points: bool,
    /// `Some(10)` when mass points were found (rdbwselect `masspoints = "adjust"`).
    pub bwcheck: Option<usize>,
    /// The pilot bandwidth `c_bw` of the variance terms.
    pub pilot_bandwidth: f64,
    /// Order of the point estimate.
    pub p: usize,
    /// Order of the bias correction.
    pub q: usize,
    /// Kernel.
    pub kernel: KernelType,
    /// Residuals of the variance terms.
    pub vce: VceType,
    /// Nearest neighbours of the `nn` residuals.
    pub nnmatch: usize,
    /// Regularisation weight.
    pub scaleregul: f64,
    /// The treatment entered the selection (false for sharp designs and one-sided compliance).
    pub fuzzy: bool,
    /// Mass points, one-sided compliance, singular pilot fits and bandwidths at the range limit.
    pub warnings: Vec<String>,
}

/// The bandwidths of one method (rdbwselect with `bwselect = method`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BwSelection {
    /// The method.
    pub method: BandwidthMethod,
    /// h and b, left and right.
    pub bandwidths: Bandwidths,
    /// Observations left of the cutoff with positive kernel weight under the selected h
    /// (rdbwselect `N_h[1]`).
    pub n_h_left: usize,
    /// The same right of the cutoff (rdbwselect `N_h[2]`).
    pub n_h_right: usize,
    /// Sample information and warnings.
    pub info: BwSelectInfo,
}

/// The bandwidths of every method (rdbwselect with `all = TRUE`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BwSelectionAll {
    /// The ten rows in rdbwselect's order ([`BandwidthMethod::ALL`]).
    pub rows: Vec<(BandwidthMethod, Bandwidths)>,
    /// Sample information and warnings.
    pub info: BwSelectInfo,
}

impl BwSelectionAll {
    /// The row of `method`.
    pub fn get(&self, method: BandwidthMethod) -> Option<Bandwidths> {
        self.rows
            .iter()
            .find(|(m, _)| *m == method)
            .map(|(_, b)| *b)
    }
}

/// `rdbwselect(y, x, c = cutoff, fuzzy = treatment, bwselect = method, ...)` on complete,
/// finite data of equal length.
///
/// ```
/// use p2a_core::econometrics::rd::BandwidthMethod;
/// use p2a_core::econometrics::rd_bwselect::{rdbwselect, BwSelectOptions};
/// let x: Vec<f64> = (0..200).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / 200.0).collect();
/// let y: Vec<f64> = x.iter().enumerate()
///     .map(|(i, &v)| 0.5 * v - 0.6 * v * v + if v >= 0.0 { 1.0 } else { 0.0 }
///          + 0.2 * (((i * 37) % 23) as f64 / 23.0 - 0.5))
///     .collect();
/// let sel = rdbwselect(&x, &y, None, 0.0, BandwidthMethod::MseRd, &BwSelectOptions::default()).unwrap();
/// assert!(sel.bandwidths.h_left > 0.0 && sel.bandwidths.b_left >= sel.bandwidths.h_left * 0.5);
/// ```
pub fn rdbwselect(
    x: &[f64],
    y: &[f64],
    treatment: Option<&[f64]>,
    cutoff: f64,
    method: BandwidthMethod,
    opts: &BwSelectOptions,
) -> EconResult<BwSelection> {
    let mut sel = Selector::new(x, y, treatment, cutoff, opts)?;
    let mut fam = Families::default();
    let bandwidths = sel.row(method, &mut fam)?;
    sel.range_warnings(method, &bandwidths);
    let (n_h_left, n_h_right) = (
        sel.count_positive(0, bandwidths.h_left),
        sel.count_positive(1, bandwidths.h_right),
    );
    Ok(BwSelection {
        method,
        bandwidths,
        n_h_left,
        n_h_right,
        info: sel.info,
    })
}

/// `rdbwselect(..., all = TRUE)`: the ten methods' bandwidths from one set of pilot fits.
pub fn rdbwselect_all(
    x: &[f64],
    y: &[f64],
    treatment: Option<&[f64]>,
    cutoff: f64,
    opts: &BwSelectOptions,
) -> EconResult<BwSelectionAll> {
    let mut sel = Selector::new(x, y, treatment, cutoff, opts)?;
    let mut fam = Families::default();
    let mut rows = Vec::with_capacity(BandwidthMethod::ALL.len());
    for method in BandwidthMethod::ALL {
        let b = sel.row(method, &mut fam)?;
        sel.range_warnings(method, &b);
        rows.push((method, b));
    }
    Ok(BwSelectionAll {
        rows,
        info: sel.info,
    })
}

// ============================================================================
// The selector
// ============================================================================

/// One side of the cutoff, sorted by the running variable (ascending).
struct SideData {
    name: &'static str,
    x: Vec<f64>,
    /// The outcome, and the treatment of a fuzzy selection, in `x` order (m x k).
    d: Array2<f64>,
    /// |c - x_min| (left) or |c - x_max| (right): rdbwselect's `range_l`, `range_r`.
    range: f64,
    /// Distance to the 10th distinct value plus 1e-8 (used only with mass points).
    bw_min: f64,
}

/// The variance part of a stage: rdbwselect's cached `V_V`, `BConst` and `s`.
#[derive(Clone)]
struct VPart {
    vv: f64,
    bconst: f64,
    s: Vec<f64>,
}

/// rdrobust_bw's output `(V, B, R, rate)`.
#[derive(Clone, Copy)]
struct Stage {
    v: f64,
    b: f64,
    r: f64,
    rate: f64,
}

/// A weighted local polynomial fit of the rows with positive kernel weight.
struct LocalFit {
    win: Window,
    inv_g: Array2<f64>,
    beta: Array2<f64>,
}

/// The pilot window of a side under `c_bw`, shared by the side's variance parts: the
/// order-`(q + 1)` basis, Gram matrix and moments (lower orders are their leading blocks),
/// and the nearest-neighbour residuals, which do not depend on the order.
struct VBase {
    /// The window at the full order (`G` (q + 2) x (q + 2), `R' W D` (q + 2) x k).
    win: Window,
    /// `vce = nn` residuals of the window.
    nn_res: Option<Array2<f64>>,
}

/// The rows of a side with positive kernel weight, the order-`o` basis and the weighted
/// Gram matrix and moments (`G`, `R' W D`).
struct Window {
    x: Vec<f64>,
    w: Vec<f64>,
    r: Array2<f64>,
    d: Array2<f64>,
    g: Array2<f64>,
    u: Array2<f64>,
}

/// (h, b) of the three families, computed on demand.
#[derive(Default)]
struct Families {
    rd: Option<(f64, f64)>,
    sum: Option<(f64, f64)>,
    /// (h_l, h_r, b_l, b_r)
    two: Option<[f64; 4]>,
}

struct Selector {
    sides: [SideData; 2],
    c: f64,
    p: usize,
    q: usize,
    opts: BwSelectOptions,
    c_bw: f64,
    bw_max: f64,
    bwcheck: bool,
    n: usize,
    info: BwSelectInfo,
    cache: [Vec<((usize, usize), VPart)>; 2],
    vbase: [Option<VBase>; 2],
}

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// R's `var(v) == 0` for the treatment of one side: every value equal.
fn constant(v: &[f64]) -> bool {
    v.iter().all(|&t| t == v[0])
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Validate the options against rdbwselect's (and p2a's) bounds.
fn validate_options(opts: &BwSelectOptions) -> EconResult<usize> {
    let p = opts.p;
    if p > MAX_RD_ORDER {
        return Err(spec(format!(
            "Polynomial order p = {p} is too large: local polynomial RD supports p from 0 to \
             {MAX_RD_ORDER} (p = 1, local linear, is the standard choice)."
        )));
    }
    let q = opts.q.unwrap_or(p + 1);
    if q > MAX_RD_ORDER + 1 {
        return Err(spec(format!(
            "Bias-correction order q = {q} is too large: q can be at most {} (the default is p + 1).",
            MAX_RD_ORDER + 1
        )));
    }
    if q <= p {
        return Err(spec(format!(
            "Bias-correction order q = {q} must be greater than the estimation order p = {p} \
             (rdrobust requires q > p; the default is p + 1)."
        )));
    }
    if opts.nnmatch == 0 {
        return Err(spec(
            "nnmatch (nearest neighbours for the variance) must be at least 1 (got 0).".to_string(),
        ));
    }
    validate_scaleregul(opts.scaleregul)?;
    Ok(q)
}

/// `(num / den)^rate`; NaN (R's `min()` would propagate it) and a zero bandwidth are errors.
fn pw(num: f64, den: f64, rate: f64, method: &str, what: &str) -> EconResult<f64> {
    let v = (num / den).powf(rate);
    if v.is_nan() {
        return Err(EconError::Computation(format!(
            "bandwidth selection ({method}) produced a non-finite value ({what}); the pilot fits \
             are degenerate"
        )));
    }
    if v <= 0.0 {
        return Err(EconError::Computation(format!(
            "bandwidth selection ({method}) produced a zero bandwidth ({what}); the pilot fits \
             are degenerate (their variance term is zero)"
        )));
    }
    Ok(v)
}

/// The middle value of three (R's `median` of three numbers).
fn median3(a: f64, b: f64, c: f64) -> f64 {
    let mut v = [a, b, c];
    v.sort_by(f64::total_cmp);
    v[1]
}

fn factorial(nu: usize) -> f64 {
    (1..=nu).map(|i| i as f64).product()
}

impl Selector {
    /// Steps 0-4 of rdbwselect: sort, sample checks, pilot bandwidth, mass points, fuzzy rule.
    fn new(
        x: &[f64],
        y: &[f64],
        treatment: Option<&[f64]>,
        c: f64,
        opts: &BwSelectOptions,
    ) -> EconResult<Selector> {
        let q = validate_options(opts)?;
        let p = opts.p;
        let n = x.len();
        if y.len() != n {
            return Err(spec(format!(
                "rdbwselect: x and y must have the same length (got {n} and {})",
                y.len()
            )));
        }
        if let Some(t) = treatment
            && t.len() != n
        {
            return Err(spec(format!(
                "rdbwselect: the treatment must have the same length as x (got {} and {n})",
                t.len()
            )));
        }
        if !c.is_finite() {
            return Err(spec(format!(
                "rdbwselect: the cutoff must be a finite number (got {c})"
            )));
        }
        let columns: [(&str, Option<&[f64]>); 3] =
            [("x", Some(x)), ("y", Some(y)), ("treatment", treatment)];
        for (name, col) in columns {
            if let Some(col) = col
                && let Some(i) = col.iter().position(|v| !v.is_finite())
            {
                return Err(spec(format!(
                    "rdbwselect: {name} must be finite (row {} is {})",
                    i + 1,
                    col[i]
                )));
            }
        }
        if n < 20 {
            return Err(EconError::InsufficientData {
                required: 20,
                provided: n,
                context: "rdbwselect needs at least 20 observations".to_string(),
            });
        }

        // Step 0: stable sort by x (R's order()); -0.0 + 0.0 = 0.0.
        let mut order: Vec<usize> = resource::try_with_capacity(n, "rdbwselect sort order")?;
        order.extend(0..n);
        let xs0: Vec<f64> = {
            let mut v = resource::try_with_capacity(n, "rdbwselect running variable")?;
            v.extend(x.iter().map(|&v| v + 0.0));
            v
        };
        order.sort_by(|&a, &b| xs0[a].total_cmp(&xs0[b]));
        let mut xs: Vec<f64> = resource::try_with_capacity(n, "rdbwselect running variable")?;
        xs.extend(order.iter().map(|&i| xs0[i]));
        drop(xs0);
        let (x_min, x_max) = (xs[0], xs[n - 1]);
        if c <= x_min || c >= x_max {
            return Err(spec(format!(
                "cutoff c = {c} must lie strictly inside the range of the running variable \
                 [{x_min}, {x_max}] (rdbwselect: c should be set within the range of x)"
            )));
        }
        let n_l = xs.partition_point(|&v| v < c);
        let n_r = n - n_l;
        if n_l < 2 || n_r < 2 {
            return Err(EconError::InsufficientData {
                required: 2,
                provided: n_l.min(n_r),
                context: "rdbwselect needs at least 2 observations on each side of the cutoff"
                    .to_string(),
            });
        }

        // Step 1: rule-of-thumb scale.
        let mean_x = mean(&xs);
        let sd =
            (xs.iter().map(|v| (v - mean_x) * (v - mean_x)).sum::<f64>() / (n - 1) as f64).sqrt();
        let iqr = quantile_sorted(&xs, 0.75, QuantileType::T2)
            - quantile_sorted(&xs, 0.25, QuantileType::T2);
        let bw_p = sd.min(iqr / 1.349);
        let range_l = (c - x_min).abs();
        let range_r = (c - x_max).abs();
        let bw_max = range_l.max(range_r);

        // Step 2: distinct values and mass points.
        let distinct = |v: &[f64]| 1 + v.windows(2).filter(|w| w[0] != w[1]).count();
        let (left_x, right_x) = xs.split_at(n_l);
        let (m_l, m_r) = (distinct(left_x), distinct(right_x));
        let mass_points =
            1.0 - m_l as f64 / n_l as f64 >= 0.2 || 1.0 - m_r as f64 / n_r as f64 >= 0.2;
        let mut warnings = Vec::new();
        if mass_points {
            warnings.push(format!(
                "Mass points detected in the running variable ({m_l} distinct values for {n_l} \
                 observations left of the cutoff, {m_r} for {n_r} right); bandwidths are kept at \
                 least as wide as the distance to the 10th distinct value on each side, as \
                 rdrobust (masspoints = \"adjust\", bwcheck = 10)"
            ));
        }
        // The 10th distinct value from the cutoff on each side.
        let nth_distinct = |vals: &mut dyn Iterator<Item = f64>, k: usize| -> f64 {
            let mut last = f64::NAN;
            let mut seen = 0;
            for v in vals {
                if seen == 0 || v != last {
                    seen += 1;
                    last = v;
                    if seen == k {
                        break;
                    }
                }
            }
            last
        };
        let bw_min_l =
            (nth_distinct(&mut left_x.iter().rev().copied(), m_l.min(10)) - c).abs() + 1e-8;
        let bw_min_r = (nth_distinct(&mut right_x.iter().copied(), m_r.min(10)) - c).abs() + 1e-8;

        // Step 3: the treatment.
        let mut fuzzy = false;
        let mut t_sorted: Option<Vec<f64>> = None;
        if let Some(t) = treatment {
            let mut ts: Vec<f64> = resource::try_with_capacity(n, "rdbwselect treatment")?;
            ts.extend(order.iter().map(|&i| t[i]));
            let (tl, tr) = ts.split_at(n_l);
            let (cl, cr) = (constant(tl), constant(tr));
            if cl && cr && (mean(tl) - mean(tr)).abs() < f64::EPSILON.sqrt() {
                return Err(spec(
                    "Fuzzy RD: the treatment has no variation and no jump at the cutoff; the fuzzy \
                     estimator is not identified"
                        .to_string(),
                ));
            }
            if cl || cr {
                let where_ = match (cl, cr) {
                    (true, true) => "either side",
                    (true, false) => "the left side",
                    _ => "the right side",
                };
                warnings.push(format!(
                    "The treatment does not vary on {where_} of the cutoff (one-sided \
                     compliance); the bandwidths are selected for the outcome alone, as rdrobust \
                     does"
                ));
            } else {
                fuzzy = true;
                t_sorted = Some(ts);
            }
        }

        // Step 4: pilot bandwidth.
        let m_total = (m_l + m_r) as f64;
        let mut c_bw = (opts.kernel.constant() * bw_p * m_total.powf(-1.0 / 5.0)).min(bw_max);
        if mass_points {
            c_bw = c_bw.max(bw_min_l).max(bw_min_r);
        }

        // The sides' data columns.
        let k = if fuzzy { 2 } else { 1 };
        let side = |lo: usize,
                    hi: usize,
                    name: &'static str,
                    range: f64,
                    bw_min: f64|
         -> EconResult<SideData> {
            let m = hi - lo;
            let mut d = resource::try_zeros2(m, k, "rdbwselect side data")?;
            for (row, idx) in order[lo..hi].iter().enumerate() {
                d[[row, 0]] = y[*idx];
            }
            if let Some(ts) = &t_sorted {
                for row in 0..m {
                    d[[row, 1]] = ts[lo + row];
                }
            }
            let mut sx = resource::try_with_capacity(m, "rdbwselect side data")?;
            sx.extend_from_slice(&xs[lo..hi]);
            Ok(SideData {
                name,
                x: sx,
                d,
                range,
                bw_min,
            })
        };
        let left = side(0, n_l, "left", range_l, bw_min_l)?;
        let right = side(n_l, n, "right", range_r, bw_min_r)?;

        let info = BwSelectInfo {
            n_left: n_l,
            n_right: n_r,
            m_left: m_l,
            m_right: m_r,
            mass_points,
            bwcheck: mass_points.then_some(10),
            pilot_bandwidth: c_bw,
            p,
            q,
            kernel: opts.kernel,
            vce: opts.vce,
            nnmatch: opts.nnmatch,
            scaleregul: opts.scaleregul,
            fuzzy,
            warnings,
        };
        Ok(Selector {
            sides: [left, right],
            c,
            p,
            q,
            opts: *opts,
            c_bw,
            bw_max,
            bwcheck: mass_points,
            n,
            info,
            cache: [Vec::new(), Vec::new()],
            vbase: [None, None],
        })
    }

    fn warn(&mut self, w: String) {
        if !self.info.warnings.contains(&w) {
            self.info.warnings.push(w);
        }
    }

    /// Rows of `side` with positive kernel weight under `h`.
    fn count_positive(&self, side: usize, h: f64) -> usize {
        let (c, kernel) = (self.c, self.opts.kernel);
        self.sides[side]
            .x
            .iter()
            .filter(|&&xi| kernel.evaluate((xi - c) / h) / h > 0.0)
            .count()
    }

    /// The rows of `side` with positive kernel weight `K((x - c)/h)/h`, their order-`o`
    /// basis, Gram matrix `G` and moments `R' W D`.
    fn window(&self, side: usize, o: usize, h: f64) -> EconResult<Window> {
        let (c, kernel) = (self.c, self.opts.kernel);
        let sd = &self.sides[side];
        let (n_side, k) = sd.d.dim();
        let mut idx: Vec<usize> = resource::try_with_capacity(n_side, "rdbwselect window")?;
        let mut w: Vec<f64> = resource::try_with_capacity(n_side, "rdbwselect kernel weights")?;
        for (i, &xi) in sd.x.iter().enumerate() {
            let wi = kernel.evaluate((xi - c) / h) / h;
            if wi > 0.0 {
                idx.push(i);
                w.push(wi);
            }
        }
        let m = idx.len();
        if m == 0 {
            return Err(EconError::InsufficientData {
                required: 1,
                provided: 0,
                context: format!(
                    "rdbwselect: no observation on the {} side of the cutoff has positive kernel \
                     weight under the pilot bandwidth {h}",
                    sd.name
                ),
            });
        }
        let cols = o + 1;
        let mut r = resource::try_zeros2(m, cols, "rdbwselect pilot design")?;
        let mut d = resource::try_zeros2(m, k, "rdbwselect pilot data")?;
        let mut xw: Vec<f64> = resource::try_with_capacity(m, "rdbwselect window")?;
        for (row, &i) in idx.iter().enumerate() {
            let dx = sd.x[i] - c;
            let mut pow = 1.0;
            for j in 0..cols {
                r[[row, j]] = pow;
                pow *= dx;
            }
            for col in 0..k {
                d[[row, col]] = sd.d[[i, col]];
            }
            xw.push(sd.x[i]);
        }
        let mut g = Array2::<f64>::zeros((cols, cols));
        let mut u = Array2::<f64>::zeros((cols, k));
        for row in 0..m {
            for j in 0..cols {
                let a = w[row] * r[[row, j]];
                for l in 0..=j {
                    g[[j, l]] += a * r[[row, l]];
                }
                for col in 0..k {
                    u[[j, col]] += a * d[[row, col]];
                }
            }
        }
        for j in 0..cols {
            for l in 0..j {
                g[[l, j]] = g[[j, l]];
            }
        }
        Ok(Window {
            x: xw,
            w,
            r,
            d,
            g,
            u,
        })
    }

    /// `invG` and `beta = invG R' W D` of an order-`o` fit from its Gram matrix and moments
    /// (rdbwselect's `qrXXinv`: Cholesky, or a generalized inverse with a warning when `G`
    /// is singular).
    fn solve(
        &mut self,
        side: usize,
        o: usize,
        h: f64,
        g: ArrayView2<f64>,
        u: ArrayView2<f64>,
    ) -> EconResult<(Array2<f64>, Array2<f64>)> {
        let what = format!(
            "rdbwselect pilot fit (order {o}, {} side, bandwidth {h:.6})",
            self.sides[side].name
        );
        let inv = sym_inverse(g, SingularPolicy::Generalized, &what)?;
        if let Some(msg) = inv.warning(&what) {
            self.warn(msg);
        }
        let beta = inv.inverse.dot(&u);
        Ok((inv.inverse, beta))
    }

    /// The weighted local polynomial of order `o` on the rows of `side` with positive kernel
    /// weight `K((x - c)/h)/h`.
    fn local_fit(&mut self, side: usize, o: usize, h: f64) -> EconResult<LocalFit> {
        let win = self.window(side, o, h)?;
        let (inv_g, beta) = self.solve(side, o, h, win.g.view(), win.u.view())?;
        Ok(LocalFit { win, inv_g, beta })
    }

    /// rdrobust_res: nearest-neighbour or HC0-HC3 residuals of a fit of order `o` (`r` its
    /// order-`o` basis, `inv_g` and `beta` its solution).
    fn residuals(
        &self,
        side: usize,
        win: &Window,
        r: ArrayView2<f64>,
        inv_g: &Array2<f64>,
        beta: &Array2<f64>,
        o: usize,
    ) -> EconResult<Array2<f64>> {
        match self.opts.vce {
            VceType::Nn => nn_residuals(&win.x, &win.d, self.opts.nnmatch),
            vce => {
                let m = win.x.len();
                if vce == VceType::Hc1 && m <= o + 1 {
                    return Err(EconError::InsufficientData {
                        required: o + 2,
                        provided: m,
                        context: format!(
                            "rdbwselect: HC1 residuals of the order-{o} pilot fit on the {} side \
                             of the cutoff need more observations than coefficients",
                            self.sides[side].name
                        ),
                    });
                }
                hc_residuals(r, &win.w, inv_g, beta, &win.d, vce)
            }
        }
    }

    /// `(invG M invG)[a, a]` with `M = sum_i (res_i . s)^2 (w_i R_i)(w_i R_i)'`
    /// (rdrobust_vce without clusters).
    fn sandwich_element(
        w: &[f64],
        r: ArrayView2<f64>,
        inv_g: &Array2<f64>,
        res: &Array2<f64>,
        s: &[f64],
        a: usize,
    ) -> f64 {
        let row = inv_g.row(a);
        let mut v = 0.0;
        for (i, wi) in w.iter().enumerate() {
            let ri: f64 = s
                .iter()
                .enumerate()
                .map(|(col, sc)| res[[i, col]] * sc)
                .sum();
            let g = wi * r.row(i).dot(&row);
            let t = ri * g;
            v += t * t;
        }
        v
    }

    /// The shared pilot window of `side` under `c_bw` (built once per side).
    fn v_base(&mut self, side: usize) -> EconResult<()> {
        if self.vbase[side].is_some() {
            return Ok(());
        }
        // The highest order of a variance part: q + 1 (the pilot stage d).
        let top = self.q + 1;
        let win = self.window(side, top, self.c_bw)?;
        let nn_res = match self.opts.vce {
            VceType::Nn => Some(nn_residuals(&win.x, &win.d, self.opts.nnmatch)?),
            _ => None,
        };
        self.vbase[side] = Some(VBase { win, nn_res });
        Ok(())
    }

    /// The variance part of a stage, cached per (side, o, nu) as rdbwselect's `vcache`. Every
    /// order shares the side's pilot window ([`VBase`]): its `G` and `R' W D` are the leading
    /// blocks of the full order's, and the `nn` residuals are computed once.
    fn v_part(&mut self, side: usize, o: usize, nu: usize) -> EconResult<VPart> {
        if let Some((_, vp)) = self.cache[side].iter().find(|(key, _)| *key == (o, nu)) {
            return Ok(vp.clone());
        }
        let h_v = self.c_bw;
        self.v_base(side)?;
        let base = self.vbase[side]
            .take()
            .ok_or_else(|| EconError::Computation("rdbwselect: pilot window missing".into()))?;
        let out = self.v_part_from(side, &base, o, nu, h_v);
        self.vbase[side] = Some(base);
        let vp = out?;
        self.cache[side].push(((o, nu), vp.clone()));
        Ok(vp)
    }

    fn v_part_from(
        &mut self,
        side: usize,
        base: &VBase,
        o: usize,
        nu: usize,
        h_v: f64,
    ) -> EconResult<VPart> {
        let cols = o + 1;
        let win = &base.win;
        let r = win.r.slice(s![.., ..cols]);
        let (inv_g, beta) = self.solve(
            side,
            o,
            h_v,
            win.g.slice(s![..cols, ..cols]),
            win.u.slice(s![..cols, ..]),
        )?;
        let s = if win.d.ncols() == 1 {
            vec![1.0]
        } else {
            let f = factorial(nu);
            let tau_y = f * beta[[nu, 0]];
            let tau_t = f * beta[[nu, 1]];
            vec![1.0 / tau_t, -(tau_y / (tau_t * tau_t))]
        };
        let hc;
        let res = match &base.nn_res {
            Some(nn) => nn,
            None => {
                hc = self.residuals(side, win, r, &inv_g, &beta, o)?;
                &hc
            }
        };
        let vv = Self::sandwich_element(&win.w, r, &inv_g, res, &s, nu);
        let pow = i32::try_from(o + 1).unwrap_or(i32::MAX);
        let mut v = vec![0.0; cols];
        for i in 0..win.x.len() {
            let t = win.w[i] * ((win.x[i] - self.c) / h_v).powi(pow);
            for (j, vj) in v.iter_mut().enumerate() {
                *vj += win.r[[i, j]] * t;
            }
        }
        let ginv_v: f64 = inv_g.row(nu).iter().zip(&v).map(|(a, b)| a * b).sum();
        let bconst = h_v.powi(i32::try_from(nu).unwrap_or(i32::MAX)) * ginv_v;
        Ok(VPart { vv, bconst, s })
    }

    /// rdrobust_bw: the variance, bias and regularisation terms of order `o`, derivative `nu`,
    /// with the bias fit of order `o_b` under `h_b`.
    fn stage(
        &mut self,
        side: usize,
        o: usize,
        nu: usize,
        o_b: usize,
        h_b: f64,
        scale: f64,
    ) -> EconResult<Stage> {
        resource::checkpoint()?;
        let vp = self.v_part(side, o, nu)?;
        let fit = self.local_fit(side, o_b, h_b)?;
        let mut bwreg = 0.0;
        if scale > 0.0 {
            let res =
                self.residuals(side, &fit.win, fit.win.r.view(), &fit.inv_g, &fit.beta, o_b)?;
            let vb = Self::sandwich_element(
                &fit.win.w,
                fit.win.r.view(),
                &fit.inv_g,
                &res,
                &vp.s,
                o + 1,
            );
            bwreg = 3.0 * vp.bconst * vp.bconst * vb;
        }
        let sb: f64 =
            vp.s.iter()
                .enumerate()
                .map(|(col, sc)| sc * fit.beta[[o + 1, col]])
                .sum();
        let k = (o + 1 - nu) as f64;
        let h_v = self.c_bw;
        Ok(Stage {
            v: (2 * nu + 1) as f64
                * h_v.powi(i32::try_from(2 * nu + 1).unwrap_or(i32::MAX))
                * vp.vv,
            b: (2.0 * k).sqrt() * vp.bconst * sb,
            r: scale * (2.0 * k) * bwreg,
            rate: 1.0 / (2 * o + 3) as f64,
        })
    }

    /// The `two` family: (h_l, h_r, b_l, b_r), each side on its own.
    fn two(&mut self, fam: &mut Families, method: &str) -> EconResult<[f64; 4]> {
        if let Some(v) = fam.two {
            return Ok(v);
        }
        let (p, q, sr) = (self.p, self.q, self.opts.scaleregul);
        let mut out = [0.0; 4];
        for side in 0..2 {
            let range = self.sides[side].range;
            let cd = self.stage(side, q + 1, q + 1, q + 2, range, 0.0)?;
            let mut d = pw(cd.v, cd.b * cd.b, cd.rate, method, "pilot bandwidth d")?.min(range);
            if self.bwcheck {
                d = d.max(self.sides[side].bw_min);
            }
            let cb = self.stage(side, q, p + 1, q + 1, d, sr)?;
            let b = pw(cb.v, cb.b * cb.b + sr * cb.r, cb.rate, method, "b")?.min(range);
            let ch = self.stage(side, p, 0, q, b, sr)?;
            let h = pw(ch.v, ch.b * ch.b + sr * ch.r, ch.rate, method, "h")?.min(range);
            out[side] = h;
            out[2 + side] = b;
        }
        fam.two = Some(out);
        Ok(out)
    }

    /// The common-bandwidth families: `rd` (difference of the bias terms) or `sum`.
    fn common(&mut self, fam: &mut Families, sum: bool, method: &str) -> EconResult<(f64, f64)> {
        if let Some(v) = if sum { fam.sum } else { fam.rd } {
            return Ok(v);
        }
        let (p, q, sr, bw_max) = (self.p, self.q, self.opts.scaleregul, self.bw_max);
        let op = |l: f64, r: f64| if sum { r + l } else { r - l };
        let (range_l, range_r) = (self.sides[0].range, self.sides[1].range);
        let cdl = self.stage(0, q + 1, q + 1, q + 2, range_l, 0.0)?;
        let cdr = self.stage(1, q + 1, q + 1, q + 2, range_r, 0.0)?;
        let bd = op(cdl.b, cdr.b);
        let mut d = pw(
            cdl.v + cdr.v,
            bd * bd,
            cdl.rate,
            method,
            "pilot bandwidth d",
        )?
        .min(bw_max);
        if self.bwcheck {
            d = d.max(self.sides[0].bw_min).max(self.sides[1].bw_min);
        }
        let cbl = self.stage(0, q, p + 1, q + 1, d, sr)?;
        let cbr = self.stage(1, q, p + 1, q + 1, d, sr)?;
        let bb = op(cbl.b, cbr.b);
        let b = pw(
            cbl.v + cbr.v,
            bb * bb + sr * (cbr.r + cbl.r),
            cbl.rate,
            method,
            "b",
        )?
        .min(bw_max);
        let chl = self.stage(0, p, 0, q, b, sr)?;
        let chr = self.stage(1, p, 0, q, b, sr)?;
        let bh = op(chl.b, chr.b);
        let h = pw(
            chl.v + chr.v,
            bh * bh + sr * (chr.r + chl.r),
            chl.rate,
            method,
            "h",
        )?
        .min(bw_max);
        if sum {
            fam.sum = Some((h, b));
        } else {
            fam.rd = Some((h, b));
        }
        Ok((h, b))
    }

    /// The bandwidths of `method`, computing only the families it needs.
    fn row(&mut self, method: BandwidthMethod, fam: &mut Families) -> EconResult<Bandwidths> {
        let name = method.name();
        let mse = match method.mse_counterpart() {
            BandwidthMethod::MseRd => {
                let (h, b) = self.common(fam, false, name)?;
                [h, h, b, b]
            }
            BandwidthMethod::MseSum => {
                let (h, b) = self.common(fam, true, name)?;
                [h, h, b, b]
            }
            BandwidthMethod::MseTwo => self.two(fam, name)?,
            BandwidthMethod::MseComb1 => {
                let (hd, bd) = self.common(fam, false, name)?;
                let (hs, bs) = self.common(fam, true, name)?;
                let (h, b) = (hd.min(hs), bd.min(bs));
                [h, h, b, b]
            }
            _ => {
                let two = self.two(fam, name)?;
                let (hs, bs) = self.common(fam, true, name)?;
                let (hd, bd) = self.common(fam, false, name)?;
                [
                    median3(hd, hs, two[0]),
                    median3(hd, hs, two[1]),
                    median3(bd, bs, two[2]),
                    median3(bd, bs, two[3]),
                ]
            }
        };
        let cer = if method.is_cer() {
            let p = self.p as f64;
            (self.n as f64).powf(-(p / ((3.0 + p) * (3.0 + 2.0 * p))))
        } else {
            1.0
        };
        Ok(Bandwidths {
            h_left: mse[0] * cer,
            h_right: mse[1] * cer,
            b_left: mse[2],
            b_right: mse[3],
        })
    }

    /// A warning for each of h and b of `method` that reached the data-range limit.
    fn range_warnings(&mut self, method: BandwidthMethod, b: &Bandwidths) {
        let limits = [
            [self.sides[0].range, self.bw_max],
            [self.sides[1].range, self.bw_max],
        ];
        let hit = |v: f64, side: usize| limits[side].contains(&v);
        for (what, l, r) in [("h", b.h_left, b.h_right), ("b", b.b_left, b.b_right)] {
            let value = if hit(l, 0) {
                Some(l)
            } else if hit(r, 1) {
                Some(r)
            } else {
                None
            };
            if let Some(v) = value {
                self.warn(format!(
                    "{}: {what} reached the range limit of the running variable ({v:.4}); \
                     rdrobust restricts bandwidths to the data range (bwrestrict)",
                    method.name()
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use crate::econometrics::rd::{BandwidthMethod, KernelType, VceType};
    use crate::errors::EconError;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    fn col(ds: &Dataset, name: &str) -> Vec<f64> {
        ds.df()
            .column(name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect()
    }
    fn o() -> BwSelectOptions {
        BwSelectOptions::default()
    }
    fn row(b: &Bandwidths) -> [f64; 4] {
        [b.h_left, b.h_right, b.b_left, b.b_right]
    }
    /// The blocks of causal_rdbw.R: (key prefix, mass-point columns xm/ym, options).
    fn sharp_blocks() -> Vec<(&'static str, bool, BwSelectOptions)> {
        vec![
            ("tri_p1_nn", false, o()),
            (
                "epa_p1_nn",
                false,
                BwSelectOptions {
                    kernel: KernelType::Epanechnikov,
                    ..o()
                },
            ),
            (
                "uni_p1_nn",
                false,
                BwSelectOptions {
                    kernel: KernelType::Uniform,
                    ..o()
                },
            ),
            ("tri_p2_nn", false, BwSelectOptions { p: 2, ..o() }),
            ("tri_p0_nn", false, BwSelectOptions { p: 0, ..o() }),
            ("tri_p1_q3", false, BwSelectOptions { q: Some(3), ..o() }),
            (
                "tri_p1_hc0",
                false,
                BwSelectOptions {
                    vce: VceType::Hc0,
                    ..o()
                },
            ),
            (
                "tri_p1_hc1",
                false,
                BwSelectOptions {
                    vce: VceType::Hc1,
                    ..o()
                },
            ),
            (
                "tri_p1_hc2",
                false,
                BwSelectOptions {
                    vce: VceType::Hc2,
                    ..o()
                },
            ),
            (
                "tri_p1_hc3",
                false,
                BwSelectOptions {
                    vce: VceType::Hc3,
                    ..o()
                },
            ),
            ("tri_p1_nn1", false, BwSelectOptions { nnmatch: 1, ..o() }),
            (
                "tri_p1_noreg",
                false,
                BwSelectOptions {
                    scaleregul: 0.0,
                    ..o()
                },
            ),
            ("mass_tri_p1_nn", true, o()),
        ]
    }

    /// Every row of `rdbwselect_all` against `<prefix>_<method>`, and every single-method
    /// result equal to its row bit for bit.
    fn check_all(
        check: &mut Check<'_>,
        prefix: &str,
        x: &[f64],
        y: &[f64],
        t: Option<&[f64]>,
        opts: &BwSelectOptions,
    ) -> BwSelectionAll {
        let n = x.len();
        let all = rdbwselect_all(x, y, t, 0.0, opts).unwrap();
        assert_eq!(all.rows.len(), 10, "{prefix}");
        for (m, bw) in &all.rows {
            check.vector(&format!("{prefix}_{}", m.name()), &row(bw), Tol::coef(n));
            let one = rdbwselect(x, y, t, 0.0, *m, opts).unwrap();
            assert_eq!(row(&one.bandwidths), row(bw), "{prefix} {m:?}");
        }
        all
    }

    #[test]
    fn test_validate_rdbwselect_all_methods() {
        let (case, ds) = load_case("rd_bwselect_sharp");
        let mut check = Check::new(&case);
        let (x, y, xm, ym) = (col(&ds, "x"), col(&ds, "y"), col(&ds, "xm"), col(&ds, "ym"));
        let counts = |i: &BwSelectInfo| {
            (
                [i.n_left as f64, i.n_right as f64],
                [i.m_left as f64, i.m_right as f64],
            )
        };
        for (prefix, mass, opts) in sharp_blocks() {
            let (xs, ys) = if mass { (&xm, &ym) } else { (&x, &y) };
            let all = check_all(&mut check, prefix, xs, ys, None, &opts);
            assert_eq!(all.info.mass_points, mass, "{prefix}");
            if prefix == "tri_p1_nn" || prefix == "mass_tri_p1_nn" {
                let key = if mass {
                    "mass_counts"
                } else {
                    "tri_p1_nn_counts"
                };
                let (nn, mm) = counts(&all.info);
                check.vector(&format!("{key}_n"), &nn, Tol::exact()).vector(
                    &format!("{key}_m"),
                    &mm,
                    Tol::exact(),
                );
            }
        }
        check.finish();

        let mass = rdbwselect(&xm, &ym, None, 0.0, BandwidthMethod::MseRd, &o()).unwrap();
        assert_eq!(mass.info.bwcheck, Some(10));
        assert!(
            mass.info
                .warnings
                .iter()
                .any(|w| w.starts_with("Mass points detected in the running variable")),
            "{:?}",
            mass.info.warnings
        );
        let noreg = BwSelectOptions {
            scaleregul: 0.0,
            ..o()
        };
        let sum = rdbwselect(&x, &y, None, 0.0, BandwidthMethod::MseSum, &noreg).unwrap();
        assert!(
            sum.info.warnings.iter().any(|w| w
                .contains("msesum: h reached the range limit of the running variable (0.9995)")),
            "{:?}",
            sum.info.warnings
        );
        let rd = rdbwselect(&x, &y, None, 0.0, BandwidthMethod::MseRd, &o()).unwrap();
        assert!(rd.info.warnings.is_empty(), "{:?}", rd.info.warnings);
        assert_eq!((rd.n_h_left, rd.n_h_right), (145, 124));
    }

    #[test]
    fn test_validate_rdbwselect_fuzzy() {
        let (case, ds) = load_case("rd_bwselect_fuzzy");
        let mut check = Check::new(&case);
        let (x, yf, d) = (col(&ds, "x"), col(&ds, "yf"), col(&ds, "d"));
        for (prefix, opts) in [
            ("fuzzy_tri_p1_nn", o()),
            (
                "fuzzy_tri_p1_hc1",
                BwSelectOptions {
                    vce: VceType::Hc1,
                    ..o()
                },
            ),
        ] {
            let all = check_all(&mut check, prefix, &x, &yf, Some(&d), &opts);
            assert!(all.info.fuzzy, "{prefix}");
        }
        // one-sided compliance: no take-up left of the cutoff
        let d1: Vec<f64> = x
            .iter()
            .zip(&d)
            .map(|(&xi, &di)| if xi >= 0.0 { di } else { 0.0 })
            .collect();
        let one = rdbwselect_all(&x, &yf, Some(&d1), 0.0, &o()).unwrap();
        let sharp = rdbwselect_all(&x, &yf, None, 0.0, &o()).unwrap();
        assert_eq!(one.rows, sharp.rows);
        assert!(!one.info.fuzzy);
        assert!(
            one.info.warnings.iter().any(|w| w
                .contains("does not vary on the left side of the cutoff (one-sided compliance)")),
            "{:?}",
            one.info.warnings
        );
        check.vector(
            "fuzzy_onesided_mserd",
            &row(&one.get(BandwidthMethod::MseRd).unwrap()),
            Tol::coef(x.len()),
        );
        check.finish();
    }

    /// `validation/datasets/generated/rd_sim.csv` as data only (no R value is compared).
    fn rd_sim() -> Dataset {
        let path = p2a_validation::repo_path("validation/datasets/generated/rd_sim.csv");
        Dataset::new(crate::data::DataLoader::load_csv(&path).unwrap())
    }

    #[test]
    fn test_rdbwselect_scale_equivariant() {
        let ds = rd_sim();
        let (x, y) = (col(&ds, "x"), col(&ds, "y"));
        let base = rdbwselect_all(&x, &y, None, 0.0, &o()).unwrap();
        let x1000: Vec<f64> = x.iter().map(|v| v * 1000.0).collect();
        let y1e6: Vec<f64> = y.iter().map(|v| v * 1e6).collect();
        let sx = rdbwselect_all(&x1000, &y, None, 0.0, &o()).unwrap();
        let sy = rdbwselect_all(&x, &y1e6, None, 0.0, &o()).unwrap();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-9 * b.abs();
        for ((m, b0), ((_, bx), (_, by))) in base.rows.iter().zip(sx.rows.iter().zip(&sy.rows)) {
            for (v0, (vx, vy)) in row(b0).iter().zip(row(bx).iter().zip(row(by).iter())) {
                assert!(close(vx / 1000.0, *v0), "{m:?} x: {} vs {v0}", vx / 1000.0);
                assert!(close(*vy, *v0), "{m:?} y: {vy} vs {v0}");
            }
        }
    }

    /// `x_i = -1 + 2 (i + 0.5)/n`, `y_i = 0.3 x_i + 1{x_i >= 0} + 0.2 (((37 i) mod 23)/23 - 0.5)`.
    fn fixture(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64)
            .collect();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                0.3 * v
                    + if v >= 0.0 { 1.0 } else { 0.0 }
                    + 0.2 * (((37 * i) % 23) as f64 / 23.0 - 0.5)
            })
            .collect();
        (x, y)
    }

    fn spec_err<T: std::fmt::Debug>(r: EconResult<T>, needle: &str) {
        match r {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains(needle), "{needle} in {message}")
            }
            other => panic!("expected InvalidSpecification containing {needle}, got {other:?}"),
        }
    }

    #[test]
    fn test_rdbwselect_rejects_degenerate_inputs() {
        let (x, y) = fixture(200);
        let mse = BandwidthMethod::MseRd;
        let sel = |x: &[f64], y: &[f64], t: Option<&[f64]>, c: f64, opts: BwSelectOptions| {
            rdbwselect(x, y, t, c, mse, &opts)
        };
        spec_err(sel(&x, &y, None, 1.0, o()), "cutoff");
        spec_err(sel(&x, &y, None, 1.0, o()), "range");
        spec_err(sel(&x, &y, None, x[0], o()), "cutoff");
        spec_err(sel(&x, &y, None, x[0], o()), "range");
        spec_err(
            sel(&x, &y, None, 0.0, BwSelectOptions { q: Some(1), ..o() }),
            "q = 1",
        );
        spec_err(
            sel(&x, &y, None, 0.0, BwSelectOptions { p: 11, ..o() }),
            "p = 11",
        );
        for v in [-1.0, f64::NAN, f64::INFINITY] {
            spec_err(
                sel(
                    &x,
                    &y,
                    None,
                    0.0,
                    BwSelectOptions {
                        scaleregul: v,
                        ..o()
                    },
                ),
                "scaleregul",
            );
        }
        spec_err(
            sel(&x, &y, None, 0.0, BwSelectOptions { nnmatch: 0, ..o() }),
            "nnmatch",
        );
        spec_err(sel(&x, &y[..199], None, 0.0, o()), "length");
        let mut xn = x.clone();
        xn[3] = f64::NAN;
        spec_err(sel(&xn, &y, None, 0.0, o()), "finite");
        let mut yi = y.clone();
        yi[5] = f64::INFINITY;
        spec_err(sel(&x, &yi, None, 0.0, o()), "finite");
        spec_err(
            sel(&x, &y, Some(&[1.0; 200]), 0.0, o()),
            "no variation and no jump",
        );
        spec_err(sel(&x, &y, Some(&[1.0; 199]), 0.0, o()), "length");
        match sel(&x[..19], &y[..19], None, 0.0, o()) {
            Err(EconError::InsufficientData {
                required: 20,
                provided: 19,
                ..
            }) => {}
            other => panic!("expected InsufficientData 20/19, got {other:?}"),
        }
    }

    /// A variance term of exactly zero is an error naming the stage, not a zero bandwidth:
    /// x on an integer grid with five rows per value (mass points), y a noiseless integer
    /// quartic of x, so every nearest-neighbour residual (own tie block) is exactly 0.
    #[test]
    fn test_rdbwselect_zero_variance_is_an_error() {
        let x: Vec<f64> = (-10..10)
            .flat_map(|v| std::iter::repeat_n(f64::from(v), 5))
            .collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| {
                if v < 0.0 {
                    v.powi(4)
                } else {
                    2.0 * v.powi(4) + 10.0
                }
            })
            .collect();
        for method in [BandwidthMethod::MseRd, BandwidthMethod::MseTwo] {
            match rdbwselect(&x, &y, None, 0.0, method, &o()) {
                Err(EconError::Computation(msg)) => assert!(
                    msg.contains(&format!("({}) produced a zero bandwidth", method.name())),
                    "{msg}"
                ),
                other => panic!("expected a zero-bandwidth error, got {other:?}"),
            }
        }
    }

    fn cancelled<R>(f: impl FnOnce() -> R) -> R {
        use crate::resource::{Budget, CancelToken, with_budget};
        let token = CancelToken::new();
        token.cancel();
        with_budget(Budget::new(None, token), f)
    }

    #[test]
    fn test_rdbwselect_respects_budget_and_allocation_ceiling() {
        let (x, y) = fixture(200);
        let run = || rdbwselect(&x, &y, None, 0.0, BandwidthMethod::MseRd, &o());
        let r = crate::resource::with_alloc_ceiling(Some(64), run);
        assert!(matches!(r, Err(EconError::ResourceLimit(_))), "{r:?}");
        let r = cancelled(run);
        assert!(matches!(r, Err(EconError::Cancelled(_))), "{r:?}");
        assert!(run().is_ok());
    }
}
