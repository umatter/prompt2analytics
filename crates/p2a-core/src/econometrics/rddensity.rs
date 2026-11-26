//! The manipulation test of `rddensity::rddensity` 3.0 (Cattaneo, Jansson & Ma 2020): is the
//! density of the running variable continuous at the cutoff? The densities just left and
//! right of the cutoff are the slopes of local polynomial fits of the empirical CDF
//! (`Y_i = (i - 1) / (N - 1)` at the sorted `X_i`), fitted on each side separately
//! (`fitselect = "unrestricted"`), and the test statistic is their difference over its
//! standard error.
//!
//! As rddensity does by default: the test uses order `q = p + 1` (p = 2), a triangular kernel,
//! the jackknife variance, and bandwidths selected by its `rdbwdensity` (`bwselect =
//! "comb"`: per side the median of the side's own, the difference and the sum bandwidths),
//! regularised to keep at least 20 + p + 1 observations and distinct values on each side.
//! With repeated values (mass points) each tie gets the CDF value of its last member, and
//! the jackknife treats a tie as one unit. The binomial tests count the observations within
//! ten windows on each side, the first holding the 20 closest observations of the side with
//! fewer, as rddensity's `bino = TRUE`.
//!
//! The moment integrals of the kernel (`Sgenerate`, `Cgenerate`, `Ggenerate`) are polynomials
//! and are evaluated exactly. rddensity's restricted model (`fitselect = "restricted"`), its
//! `all = TRUE` output and its binomial-window options have no counterpart here.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::options::Choices;
use crate::resource;
use crate::stats::binomtest::binom_test;
use crate::stats::ttest::Alternative;
use crate::traits::estimator::{normal_p_value_two_sided, normal_pdf};

/// Highest order of the density fit (rddensity `p`; the test uses `p + 1`).
pub const MAX_RDDENSITY_ORDER: usize = 7;

/// Kernel of the local polynomial fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DensityKernel {
    Triangular,
    Epanechnikov,
    Uniform,
}

/// Variance estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DensityVce {
    Jackknife,
    Plugin,
}

/// Bandwidth choice when none is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DensityBwSelect {
    /// Per side, the median of the side's own, the difference and the sum bandwidths.
    Comb,
    /// Each side its own bandwidth.
    Each,
    /// One bandwidth, optimal for the difference of the densities.
    Diff,
    /// One bandwidth, optimal for their sum.
    Sum,
}

crate::option_enum!(
    DensityKernel,
    Choices {
        option: "kernel",
        values: &[
            (&["triangular", "tri"], DensityKernel::Triangular),
            (&["epanechnikov", "epa"], DensityKernel::Epanechnikov),
            (&["uniform", "uni"], DensityKernel::Uniform),
        ],
    }
);
crate::option_enum!(
    DensityVce,
    Choices {
        option: "vce",
        values: &[
            (&["jackknife"], DensityVce::Jackknife),
            (&["plugin"], DensityVce::Plugin),
        ],
    }
);
crate::option_enum!(
    DensityBwSelect,
    Choices {
        option: "bwselect",
        values: &[
            (&["comb"], DensityBwSelect::Comb),
            (&["each"], DensityBwSelect::Each),
            (&["diff"], DensityBwSelect::Diff),
            (&["sum"], DensityBwSelect::Sum),
        ],
    }
);

/// Options of [`rd_density`]; `Default` is rddensity's.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RdDensityConfig {
    /// The cutoff (rddensity `c`); strictly inside the range of the data.
    pub cutoff: f64,
    /// Order of the density fit for the bandwidths (1 to 7); the test uses `p + 1`.
    pub p: usize,
    pub kernel: DensityKernel,
    pub vce: DensityVce,
    pub bwselect: DensityBwSelect,
    /// Bandwidths left and right by hand (rddensity `h`); `None` selects them.
    pub h: Option<(f64, f64)>,
    /// Treat repeated values as mass points (rddensity `massPoints`).
    pub mass_points: bool,
}

impl Default for RdDensityConfig {
    fn default() -> Self {
        Self {
            cutoff: 0.0,
            p: 2,
            kernel: DensityKernel::Triangular,
            vce: DensityVce::Jackknife,
            bwselect: DensityBwSelect::Comb,
            h: None,
            mass_points: true,
        }
    }
}

/// The bandwidths `rdbwdensity` selects (its column `bw`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RdDensityBandwidths {
    pub left: f64,
    pub right: f64,
    pub diff: f64,
    pub sum: f64,
}

/// One binomial test window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RdDensityWindow {
    /// Half-widths of the window left and right of the cutoff.
    pub left: f64,
    pub right: f64,
    /// Observations within them.
    pub n_left: usize,
    pub n_right: usize,
    /// Exact two-sided binomial p-value of `n_left` out of `n_left + n_right` at 1/2.
    pub p_value: f64,
}

/// The density test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdDensityResult {
    pub cutoff: f64,
    pub p: usize,
    /// Order of the test's fit (`p + 1`).
    pub q: usize,
    pub kernel: DensityKernel,
    pub vce: DensityVce,
    pub bwselect: DensityBwSelect,
    /// The bandwidths used, left and right.
    pub h: (f64, f64),
    /// The selected bandwidths; `None` when `h` was given.
    pub bandwidths: Option<RdDensityBandwidths>,
    /// Observations left and right of the cutoff, and within the bandwidths.
    pub n: (usize, usize),
    pub n_eff: (usize, usize),
    /// Densities left and right of the cutoff, and right minus left.
    pub f_left: f64,
    pub f_right: f64,
    pub diff: f64,
    /// Their standard errors (NaN when the variance estimate is negative).
    pub se_left: f64,
    pub se_right: f64,
    pub se_diff: f64,
    /// `diff / se_diff` and its two-sided normal p-value.
    pub t: f64,
    pub p_value: f64,
    /// Whether repeated values were treated as mass points.
    pub mass_points: bool,
    pub binomial: Vec<RdDensityWindow>,
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl std::fmt::Display for RdDensityResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Manipulation test of the running variable's density (Cattaneo, Jansson & Ma 2020; rddensity)"
        )?;
        writeln!(f, "Cutoff: {}", self.cutoff)?;
        writeln!(
            f,
            "Observations: {} left, {} right; within the bandwidth: {} left, {} right",
            self.n.0, self.n.1, self.n_eff.0, self.n_eff.1
        )?;
        let source = if self.bandwidths.is_some() {
            format!("selected, bwselect = {}", self.bwselect)
        } else {
            "given".into()
        };
        writeln!(
            f,
            "Bandwidth: {:.4} left, {:.4} right ({source})",
            self.h.0, self.h.1
        )?;
        writeln!(
            f,
            "Order {} (bandwidths from order {}), {} kernel, {} variance",
            self.q, self.p, self.kernel, self.vce
        )?;
        if self.mass_points {
            writeln!(f, "Repeated values treated as mass points")?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "Density left of the cutoff:  {:.6} (SE {:.6})",
            self.f_left, self.se_left
        )?;
        writeln!(
            f,
            "Density right of the cutoff: {:.6} (SE {:.6})",
            self.f_right, self.se_right
        )?;
        writeln!(
            f,
            "Difference (right - left):   {:.6} (SE {:.6})",
            self.diff, self.se_diff
        )?;
        writeln!(f, "T = {:.4}, p = {:.4}", self.t, self.p_value)?;
        let verdict = if self.p_value < 0.05 {
            "The density jumps at the cutoff (p < 0.05): units may sort around it, which threatens the RD design."
        } else {
            "No evidence that the density jumps at the cutoff (p >= 0.05); this does not prove there is no sorting."
        };
        writeln!(f, "{verdict}")?;
        if !self.binomial.is_empty() {
            writeln!(f)?;
            writeln!(
                f,
                "Binomial tests (observations within each window around the cutoff):"
            )?;
            writeln!(
                f,
                "  Window left   Window right   N left   N right   p-value"
            )?;
            for w in &self.binomial {
                writeln!(
                    f,
                    "  {:>11.4}   {:>12.4}   {:>6}   {:>7}   {:.4}",
                    w.left, w.right, w.n_left, w.n_right, w.p_value
                )?;
            }
        }
        for w in &self.warnings {
            writeln!(f, "Warning: {w}")?;
        }
        Ok(())
    }
}

/// [`rd_density`] on a column of a dataset; missing values are dropped and counted.
pub fn run_rd_density(
    dataset: &Dataset,
    running_var: &str,
    config: RdDensityConfig,
) -> EconResult<RdDensityResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(running_var)
        .na(NaPolicy::Omit)
        .extract()?;
    let mut result = rd_density(cols.numeric(running_var), config)?;
    result.missing = cols.missing().clone();
    if let Some(note) = result.missing.note() {
        result.warnings.push(note);
    }
    Ok(result)
}

/// The density test on finite values `x`; see the module docs.
pub fn rd_density(x: &[f64], config: RdDensityConfig) -> EconResult<RdDensityResult> {
    let c = config.cutoff;
    let p = config.p;
    if !c.is_finite() {
        return Err(spec("the cutoff must be a finite number".into()));
    }
    if !(1..=MAX_RDDENSITY_ORDER).contains(&p) {
        return Err(spec(format!(
            "p must be between 1 and {MAX_RDDENSITY_ORDER}, got {p}"
        )));
    }
    if let Some((hl, hr)) = config.h {
        if !(hl.is_finite() && hr.is_finite() && hl > 0.0 && hr > 0.0) {
            return Err(spec(format!(
                "bandwidths must be positive and finite, got {hl} and {hr}"
            )));
        }
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(spec("the running variable has non-finite values".into()));
    }
    let mut xs = x.to_vec();
    xs.sort_by(f64::total_cmp);
    let n = xs.len();
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "observations of the running variable".into(),
        });
    }
    let (x_min, x_max) = (xs[0], xs[n - 1]);
    if c <= x_min || c >= x_max {
        return Err(spec(format!(
            "the cutoff {c} must lie inside the range of the running variable ({x_min} to {x_max})"
        )));
    }
    for v in &mut xs {
        *v -= c;
    }
    let nl = xs.iter().filter(|v| **v < 0.0).count();
    let nr = n - nl;
    let repeated = xs.windows(2).any(|w| w[0] == w[1]);
    let mass = repeated && config.mass_points;
    let mut warnings = Vec::new();

    // The empirical CDF at the sorted values; a tie takes the value of its last member.
    let mut y: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    if mass {
        let mut start = 0;
        for i in 0..n {
            if i + 1 == n || xs[i + 1] != xs[i] {
                for v in &mut y[start..=i] {
                    *v = i as f64 / (n - 1) as f64;
                }
                start = i + 1;
            }
        }
    }
    let unique: Vec<f64> = {
        let mut u = xs.clone();
        u.dedup();
        u
    };

    let moments = Moments::new(config.kernel);
    let (h, bandwidths) = match config.h {
        Some(h) => (h, None),
        None => {
            let bw = select_bandwidths(&xs, &y, &unique, nl, nr, p, &config, mass, &moments)?;
            let median3 = |a: f64, b: f64, c: f64| {
                let mut v = [a, b, c];
                v.sort_by(f64::total_cmp);
                v[1]
            };
            let h = match config.bwselect {
                DensityBwSelect::Each => (bw.left, bw.right),
                DensityBwSelect::Diff => (bw.diff, bw.diff),
                DensityBwSelect::Sum => (bw.sum, bw.sum),
                DensityBwSelect::Comb => (
                    median3(bw.left, bw.diff, bw.sum),
                    median3(bw.right, bw.diff, bw.sum),
                ),
            };
            (h, Some(bw))
        }
    };

    let (hl, hr) = h;
    let inside: Vec<usize> = (0..n).filter(|&i| xs[i] >= -hl && xs[i] <= hr).collect();
    let xh: Vec<f64> = inside.iter().map(|&i| xs[i]).collect();
    let yh: Vec<f64> = inside.iter().map(|&i| y[i]).collect();
    let nlh = xh.iter().filter(|v| **v < 0.0).count();
    let nrh = xh.len() - nlh;
    let q = p + 1;
    let fit = fit_density(&yh, &xh, nlh, n, hl, hr, q, 1, &config, mass, &moments)?;
    let Some(fit) = fit else {
        return Err(spec(format!(
            "the density cannot be estimated: too few distinct values within the bandwidth \
             ({nlh} left, {nrh} right) for a fit of order {q}"
        )));
    };
    let [f_left, f_right, diff, _] = fit.hat;
    let [v_left, v_right, v_diff, _] = fit.var;
    let t = diff / v_diff.sqrt();
    let p_value = normal_p_value_two_sided(t.abs());
    if !t.is_finite() {
        warnings.push(
            "The variance estimate of the difference is negative or undefined, so no test \
             statistic is reported."
                .into(),
        );
    }

    let binomial = binomial_windows(&xs, nl, nr, hl, hr)?;

    Ok(RdDensityResult {
        cutoff: c,
        p,
        q,
        kernel: config.kernel,
        vce: config.vce,
        bwselect: config.bwselect,
        h,
        bandwidths,
        n: (nl, nr),
        n_eff: (nlh, nrh),
        f_left,
        f_right,
        diff,
        se_left: v_left.sqrt(),
        se_right: v_right.sqrt(),
        se_diff: v_diff.sqrt(),
        t,
        p_value,
        mass_points: mass,
        binomial,
        warnings,
        missing: MissingSummary::default(),
    })
}

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// The k-th smallest (1-based, capped at the length) of a sorted slice.
fn kth(sorted: &[f64], k: usize) -> f64 {
    sorted[k.min(sorted.len()).max(1) - 1]
}

/// Distances to the cutoff of the left values, ascending, and the right values (ascending).
fn sides(sorted_centered: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let left: Vec<f64> = sorted_centered
        .iter()
        .rev()
        .filter(|v| **v < 0.0)
        .map(|v| -v)
        .collect();
    let right: Vec<f64> = sorted_centered
        .iter()
        .copied()
        .filter(|v| *v >= 0.0)
        .collect();
    (left, right)
}

/// rdbwdensity: pilot bandwidths from a normal reference, the variance from the order-p fit
/// and the bias from the order-(p + 2) fit, regularised.
#[allow(clippy::too_many_arguments)]
fn select_bandwidths(
    xs: &[f64],
    y: &[f64],
    unique: &[f64],
    nl: usize,
    nr: usize,
    p: usize,
    config: &RdDensityConfig,
    mass: bool,
    moments: &Moments,
) -> EconResult<RdDensityBandwidths> {
    let n = xs.len();
    let nf = n as f64;
    let mean = xs.iter().sum::<f64>() / nf;
    let sd = (xs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0)).sqrt();
    let z = mean / sd;
    let fhat_b = 1.0 / (hermite(z, p + 2).powi(2) * normal_pdf(z));
    let fhat_c = 1.0 / (hermite(z, p).powi(2) * normal_pdf(z));
    const CB: [f64; 7] = [
        25884.4444444942,
        3430865.45512362,
        845007948.042626,
        330631733667.038,
        187774809656037.0,
        145729502641999264.0,
        1.4601350297445e+20,
    ];
    const CC: [f64; 7] = [
        4.80000000000002,
        548.571428571555,
        100800.000000204,
        29558225.4581006,
        12896196859.6126,
        7890871468221.61,
        6467911284037581.0,
    ];
    let pf = p as f64;
    let mut bn =
        ((2.0 * pf + 1.0) / 4.0 * fhat_b * CB[p - 1] / nf).powf(1.0 / (2.0 * pf + 5.0)) * sd;
    let mut cn = (1.0 / (2.0 * pf) * fhat_c * CC[p - 1] / nf).powf(1.0 / (2.0 * pf + 1.0)) * sd;

    let (left, right) = sides(xs);
    let (uleft, uright) = sides(unique);
    let farthest = unique.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    bn = bn.min(farthest);
    cn = cn.min(farthest);
    let (kb, kc) = (20 + p + 3, 20 + p + 1);
    bn = bn.max(kth(&left, kb)).max(kth(&right, kb));
    cn = cn.max(kth(&left, kc)).max(kth(&right, kc));
    bn = bn.max(kth(&uleft, kb)).max(kth(&uright, kb));
    cn = cn.max(kth(&uleft, kc)).max(kth(&uright, kc));

    let window = |h: f64| -> (Vec<f64>, Vec<f64>, usize) {
        let idx: Vec<usize> = (0..n).filter(|&i| xs[i].abs() <= h).collect();
        let xw: Vec<f64> = idx.iter().map(|&i| xs[i]).collect();
        let yw: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
        let nlw = xw.iter().filter(|v| **v < 0.0).count();
        (xw, yw, nlw)
    };
    let (xb, yb, nlb) = window(bn);
    let (xc, yc, nlc) = window(cn);
    let fit_b = fit_density(
        &yb,
        &xb,
        nlb,
        n,
        bn,
        bn,
        p + 2,
        p + 1,
        config,
        mass,
        moments,
    )?;
    let fit_c = fit_density(&yc, &xc, nlc, n, cn, cn, p, 1, config, mass, moments)?;

    let mut h = [0.0_f64; 4];
    if let (Some(fb), Some(fc)) = (fit_b, fit_c) {
        let s = moments.s(p);
        let cvec = moments.c(p + 1, p);
        let sinv = inverse(&s)?;
        let t = sinv.row(1).dot(&cvec);
        let sign = if p.is_multiple_of(2) { 1.0 } else { -1.0 };
        let bias_l = fb.s[0] * t * sign;
        let bias_r = fb.s[1] * t;
        let bias = [bias_l, bias_r, bias_r - bias_l, bias_r + bias_l];
        for i in 0..4 {
            let variance = nf * cn * fc.var[i];
            let hi =
                (1.0 / (2.0 * pf) * variance / bias[i].powi(2) / nf).powf(1.0 / (2.0 * pf + 1.0));
            h[i] = if variance < 0.0 || !hi.is_finite() {
                0.0
            } else {
                hi
            };
        }
    }
    let _ = (nl, nr);
    let (lo, hi) = (unique[0].abs(), unique[unique.len() - 1]);
    h[0] = h[0].min(lo);
    h[1] = h[1].min(hi);
    h[2] = h[2].min(lo.max(hi));
    h[3] = h[3].min(lo.max(hi));
    let k = 20 + p + 1;
    for (l, r) in [
        (kth(&left, k), kth(&right, k)),
        (kth(&uleft, k), kth(&uright, k)),
    ] {
        h[0] = h[0].max(l);
        h[1] = h[1].max(r);
        h[2] = h[2].max(l).max(r);
        h[3] = h[3].max(l).max(r);
    }
    Ok(RdDensityBandwidths {
        left: h[0],
        right: h[1],
        diff: h[2],
        sum: h[3],
    })
}

/// rddensity's binomial tests: ten windows, the first reaching the 20th closest observation
/// of the side where it is farther, the others evenly spaced up to the bandwidth (or
/// multiples of the first window when ten of them fit inside it).
fn binomial_windows(
    xs: &[f64],
    nl: usize,
    nr: usize,
    hl: f64,
    hr: f64,
) -> EconResult<Vec<RdDensityWindow>> {
    const COUNT: usize = 10;
    let (left, right) = sides(xs);
    let first = kth(&left, 20.min(nl)).max(kth(&right, 20.min(nr)));
    let mut wl = vec![first];
    let mut wr = vec![first];
    if !(first >= hl || first >= hr) {
        for k in 1..COUNT {
            let kf = k as f64;
            wl.push(if first * COUNT as f64 > hl {
                first + kf * (hl - first) / (COUNT - 1) as f64
            } else {
                first + kf * first
            });
            wr.push(if first * COUNT as f64 > hr {
                first + kf * (hr - first) / (COUNT - 1) as f64
            } else {
                first + kf * first
            });
        }
    }
    let within = |sorted: &[f64], w: f64| sorted.partition_point(|v| *v <= w);
    wl.iter()
        .zip(&wr)
        .map(|(&l, &r)| {
            let (n_left, n_right) = (within(&left, l), within(&right, r));
            let total = (n_left + n_right) as u64;
            let p_value = if total == 0 {
                1.0
            } else {
                binom_test(n_left as u64, total, 0.5, Alternative::TwoSided, 0.95)?.p_value
            };
            Ok(RdDensityWindow {
                left: l,
                right: r,
                n_left,
                n_right,
                p_value,
            })
        })
        .collect()
}

/// rddensity's Hermite polynomials (probabilists'), `He_p(x)`.
fn hermite(x: f64, p: usize) -> f64 {
    let (mut prev, mut cur) = (1.0, x);
    if p == 0 {
        return prev;
    }
    for k in 1..p {
        let next = x * cur - k as f64 * prev;
        prev = cur;
        cur = next;
    }
    cur
}

/// A fit's estimates: rows left, right, difference, sum.
#[derive(Debug, Clone, Copy)]
struct DensityFit {
    /// The densities (`hat`).
    hat: [f64; 4],
    /// Their variances under the chosen estimator; NaN when negative.
    var: [f64; 4],
    /// The order-s derivative coefficients (`s` column), left and right.
    s: [f64; 2],
}

/// rddensity_fV: the local polynomial fit of order `p` of the CDF values `y` at the centered,
/// sorted values `x` (the first `nlh` left of the cutoff); `None` when the fit is singular.
#[allow(clippy::too_many_arguments)]
fn fit_density(
    y: &[f64],
    x: &[f64],
    nlh: usize,
    n: usize,
    hl: f64,
    hr: f64,
    p: usize,
    s: usize,
    config: &RdDensityConfig,
    mass: bool,
    moments: &Moments,
) -> EconResult<Option<DensityFit>> {
    resource::checkpoint()?;
    let nh = x.len();
    let k = 2 * p + 2;
    let mut xpw = resource::try_zeros2(nh, k, "density design")?;
    let mut a = Array2::<f64>::zeros((k, k));
    let mut r = Array1::<f64>::zeros(k);
    let mut hp = vec![0.0; k];
    for m in 0..=p {
        hp[2 * m] = hl.powi(m as i32);
        hp[2 * m + 1] = hr.powi(m as i32);
    }
    let mut row = vec![0.0; k];
    for i in 0..nh {
        let left = i < nlh;
        let h = if left { hl } else { hr };
        let u = x[i] / h;
        let w = match config.kernel {
            DensityKernel::Uniform => 1.0 / (2.0 * h),
            DensityKernel::Triangular => (1.0 - u.abs()) / h,
            DensityKernel::Epanechnikov => 0.75 * (1.0 - u * u) / h,
        };
        row.iter_mut().for_each(|v| *v = 0.0);
        let mut acc = 1.0;
        for m in 0..=p {
            row[2 * m + usize::from(!left)] = acc;
            acc *= u;
        }
        for j in 0..k {
            if row[j] == 0.0 {
                continue;
            }
            xpw[[i, j]] = row[j] * w;
            r[j] += xpw[[i, j]] * y[i];
            for l in 0..k {
                a[[j, l]] += xpw[[i, j]] * row[l];
            }
        }
    }
    let Ok(sinv) = sym_inverse(a.view(), SingularPolicy::Error, "density fit") else {
        return Ok(None);
    };
    let sinv = sinv.inverse;
    let beta = sinv.dot(&r);
    let b: Vec<f64> = (0..k).map(|j| beta[j] / hp[j]).collect();
    let hat = [b[2], b[3], b[3] - b[2], b[3] + b[2]];
    let svals = [b[2 * s], b[2 * s + 1]];

    let nan_if_negative = |v: f64| if v < 0.0 { f64::NAN } else { v };
    let var = match config.vce {
        DensityVce::Jackknife => {
            // L[i] = sum over rows after i of XpW / (N - 1); a tie uses the value at its first
            // member.
            let mut l = resource::try_zeros2(nh, k, "jackknife scores")?;
            let mut run = vec![0.0; k];
            for i in (0..nh).rev() {
                for j in 0..k {
                    l[[i, j]] = run[j] / (n - 1) as f64;
                    run[j] += xpw[[i, j]];
                }
            }
            if mass {
                let mut first = 0;
                for i in 1..nh {
                    if x[i] != x[i - 1] {
                        first = i;
                    } else {
                        for j in 0..k {
                            l[[i, j]] = l[[first, j]];
                        }
                    }
                }
            }
            let m = l.t().dot(&l);
            let core = sinv.dot(&m).dot(&sinv);
            let v = |a: usize, b: usize| core[[a, b]] / (hp[a] * hp[b]);
            [
                v(2, 2),
                v(3, 3),
                v(2, 2) + v(3, 3) - 2.0 * v(2, 3),
                v(2, 2) + v(3, 3) + 2.0 * v(2, 3),
            ]
        }
        DensityVce::Plugin => {
            let s_mat = moments.s(p);
            let sinv_s = inverse(&s_mat)?;
            let vmat = sinv_s.dot(&moments.g(p)).dot(&sinv_s);
            let nf = n as f64;
            let vl = hat[0] * vmat[[1, 1]] / (nf * hl);
            let vr = hat[1] * vmat[[1, 1]] / (nf * hr);
            [vl, vr, vl + vr, vl + vr]
        }
    };
    Ok(Some(DensityFit {
        hat,
        var: var.map(nan_if_negative),
        s: svals,
    }))
}

/// Inverse of a small symmetric positive-definite moment matrix.
fn inverse(m: &Array2<f64>) -> EconResult<Array2<f64>> {
    Ok(sym_inverse(m.view(), SingularPolicy::Error, "kernel moment matrix")?.inverse)
}

/// The kernel's moments on [0, 1]: rddensity's `Sgenerate`, `Cgenerate` and `Ggenerate` with
/// `low = 0, up = 1`, exact.
struct Moments {
    /// The kernel as a polynomial on [0, 1].
    kernel: Vec<f64>,
}

impl Moments {
    fn new(kernel: DensityKernel) -> Self {
        let kernel = match kernel {
            DensityKernel::Uniform => vec![0.5],
            DensityKernel::Triangular => vec![1.0, -1.0],
            DensityKernel::Epanechnikov => vec![0.75, 0.0, -0.75],
        };
        Self { kernel }
    }

    /// `x^k K(x)` as a polynomial.
    fn times_power(&self, k: usize) -> Vec<f64> {
        let mut out = vec![0.0; k];
        out.extend_from_slice(&self.kernel);
        out
    }

    /// `∫_0^1 x^k K(x) dx`.
    fn m(&self, k: usize) -> f64 {
        integral01(&self.times_power(k))
    }

    /// `S[a][b] = m(a + b)`, `a, b = 0..=p`.
    fn s(&self, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((p + 1, p + 1), |(a, b)| self.m(a + b))
    }

    /// `C[a] = m(a + k)`, `a = 0..=p`.
    fn c(&self, k: usize, p: usize) -> Array1<f64> {
        Array1::from_shape_fn(p + 1, |a| self.m(a + k))
    }

    /// `G[a][b] = ∫_0^1 y^b K(y) ∫_0^y x^(a+1) K(x) dx dy + ∫_0^1 y^(b+1) K(y) ∫_y^1 x^a K(x) dx dy`.
    fn g(&self, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((p + 1, p + 1), |(a, b)| {
            let p1 = antiderivative(&self.times_power(a + 1)); // P1(0) = 0
            let t1 = integral01(&mul(&self.times_power(b), &p1));
            let p2 = antiderivative(&self.times_power(a));
            let p2_at_1: f64 = p2.iter().sum();
            let mut inner = p2.iter().map(|v| -v).collect::<Vec<_>>();
            inner[0] += p2_at_1;
            let t2 = integral01(&mul(&self.times_power(b + 1), &inner));
            t1 + t2
        })
    }
}

fn mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// The antiderivative vanishing at 0.
fn antiderivative(a: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0];
    out.extend(a.iter().enumerate().map(|(i, v)| v / (i + 1) as f64));
    out
}

fn integral01(a: &[f64]) -> f64 {
    a.iter().enumerate().map(|(i, v)| v / (i + 1) as f64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};

    /// Every number rddensity reports (bandwidths, observations, densities, standard errors,
    /// T, p, the binomial windows) and rdbwdensity's bandwidths, for the defaults, mass points,
    /// a sorted running variable, the other kernels, the plug-in variance, each bwselect, p = 1,
    /// bandwidths given and a cutoff off zero.
    #[test]
    fn test_validate_rddensity_sim() {
        let (case, ds) = load_case("rddensity_sim");
        let mut check = Check::new(&case);
        let n = 1500;
        type Tweak = fn(&mut RdDensityConfig);
        let runs: [(&str, &str, Tweak); 12] = [
            ("default", "x", |_| {}),
            ("mass", "xm", |_| {}),
            ("sorted", "s", |_| {}),
            ("uniform", "x", |c| c.kernel = DensityKernel::Uniform),
            ("epa", "x", |c| c.kernel = DensityKernel::Epanechnikov),
            ("plugin", "x", |c| c.vce = DensityVce::Plugin),
            ("each", "x", |c| c.bwselect = DensityBwSelect::Each),
            ("diff", "x", |c| c.bwselect = DensityBwSelect::Diff),
            ("sum", "x", |c| c.bwselect = DensityBwSelect::Sum),
            ("p1", "x", |c| c.p = 1),
            ("h", "x", |c| c.h = Some((0.3, 0.4))),
            ("c02", "x", |c| c.cutoff = 0.2),
        ];
        for (prefix, column, tweak) in runs {
            let mut config = RdDensityConfig::default();
            tweak(&mut config);
            let r = run_rd_density(&ds, column, config).unwrap();
            let k = |s: &str| format!("{prefix}_{s}");
            let windows =
                |f: fn(&RdDensityWindow) -> f64| r.binomial.iter().map(f).collect::<Vec<_>>();
            check
                .vector(&k("h"), &[r.h.0, r.h.1], Tol::coef(n))
                .vector(&k("n"), &[r.n.0 as f64, r.n.1 as f64], Tol::exact())
                .vector(
                    &k("n_eff"),
                    &[r.n_eff.0 as f64, r.n_eff.1 as f64],
                    Tol::exact(),
                )
                .vector(&k("hat"), &[r.f_left, r.f_right, r.diff], Tol::coef(n))
                .vector(&k("se"), &[r.se_left, r.se_right, r.se_diff], Tol::se(n))
                .scalar(&k("t"), r.t, Tol::stat(n))
                .scalar(&k("p"), r.p_value, Tol::p_value(n))
                .vector(&k("bino_left"), &windows(|w| w.left), Tol::coef(n))
                .vector(&k("bino_right"), &windows(|w| w.right), Tol::coef(n))
                .vector(
                    &k("bino_n_left"),
                    &windows(|w| w.n_left as f64),
                    Tol::exact(),
                )
                .vector(
                    &k("bino_n_right"),
                    &windows(|w| w.n_right as f64),
                    Tol::exact(),
                )
                .vector(&k("bino_p"), &windows(|w| w.p_value), Tol::p_value(n));
            match r.bandwidths {
                Some(b) => {
                    check.vector(&k("bw"), &[b.left, b.right, b.diff, b.sum], Tol::coef(n));
                }
                None => assert_eq!(prefix, "h"),
            }
            assert_eq!(r.mass_points, prefix == "mass", "{prefix}");
        }
        check.finish();
    }

    fn uniform_draws(n: usize, jump: bool) -> Vec<f64> {
        // A deterministic low-discrepancy sample on [-1, 1]; with `jump` the right half is
        // three times as dense.
        let mut out = Vec::new();
        for i in 0..n {
            let u = ((i as f64 + 0.5) * 0.754_877_666_2) % 1.0;
            let v = 2.0 * u - 1.0;
            out.push(v);
            if jump && v > 0.0 {
                out.push(v * 0.999);
                out.push(v * 0.998);
            }
        }
        out
    }

    /// The kernel moments match closed forms (triangular S on [0,1]: 1/(k+1) - 1/(k+2)).
    #[test]
    fn moments_are_exact() {
        let m = Moments::new(DensityKernel::Triangular);
        assert!((m.m(0) - 0.5).abs() < 1e-15);
        assert!((m.m(2) - (1.0 / 3.0 - 0.25)).abs() < 1e-15);
        let u = Moments::new(DensityKernel::Uniform);
        // G for p = 0, uniform: ∫∫_{x<y} x/4 + ∫∫_{x>y} y/4 = 1/24 + 1/24.
        assert!((u.g(0)[[0, 0]] - 1.0 / 12.0).abs() < 1e-15);
        assert_eq!(hermite(2.0, 3), 8.0 - 6.0);
    }

    /// A continuous density is not rejected; a tripled density right of the cutoff is.
    #[test]
    fn detects_a_jump_and_not_a_smooth_density() {
        let smooth = rd_density(&uniform_draws(2000, false), RdDensityConfig::default()).unwrap();
        assert!(smooth.p_value > 0.05, "{smooth}");
        let jump = rd_density(&uniform_draws(2000, true), RdDensityConfig::default()).unwrap();
        assert!(jump.p_value < 0.01 && jump.diff > 0.0, "{jump}");
        assert_eq!(jump.binomial.len(), 10);
    }

    /// Bad cutoffs, orders and bandwidths are refused.
    #[test]
    fn options_are_checked() {
        let x = uniform_draws(500, false);
        let mut c = RdDensityConfig::default();
        c.cutoff = 3.0;
        assert!(rd_density(&x, c).is_err());
        let mut c = RdDensityConfig::default();
        c.p = 8;
        assert!(rd_density(&x, c).is_err());
        let mut c = RdDensityConfig::default();
        c.h = Some((0.0, 0.3));
        assert!(rd_density(&x, c).is_err());
    }
}
