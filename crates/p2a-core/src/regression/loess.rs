//! Local Polynomial Regression (LOESS/LOWESS).
//!
//! Pure Rust implementation of locally estimated scatterplot smoothing following
//! Cleveland's original algorithms.
//!
//! # Method Overview
//!
//! LOESS (LOcally Estimated Scatterplot Smoothing) fits simple models to localized
//! subsets of the data to build up a function that describes the deterministic part
//! of the variation in the data, point by point.
//!
//! For each point x0, the algorithm:
//! 1. Selects the q = floor(span * n) nearest neighbours by distance (R's rule,
//!    `min(n, floor(n span + 1e-5))`; all n for span >= 1); every point at the q-th
//!    distance is included, so a tie group is never split
//! 2. Computes tricubic weights w(u) = (1 - |u|^3)^3 on u = (x - x0) / radius, where the
//!    radius is the q-th distance; for span > 1 it is multiplied by sqrt(span) (R's code
//!    scales the squared radius by span; its documentation says span^(1/p))
//! 3. Fits a weighted polynomial of the given degree in u, with lm.fit's relative rank
//!    rule, so the fit does not depend on the scale of x; a degree that drops is reported
//! 4. Returns the fitted value at x0
//!
//! The local fits are exact at every point, as R's `loess(surface = "direct")`; R's default
//! kd-tree interpolation differs from them by up to about 1e-3 relative.
//!
//! **Zero-width neighbourhoods.** When at least q observations share the value x0, the
//! neighbourhood has zero width. The fit there is the weighted mean of the tied responses
//! (R `lowess`'s rule) and a warning names the smallest span that avoids it; R's `loess`
//! returns 0 there and warns "zero-width neighborhood. make span bigger". A constant x is
//! an error.
//!
//! `enp` is the trace of the hat matrix (R's `trace.hat`), not R's "Equivalent Number of
//! Parameters"; the residual SE and the robust iterations are not yet R's (REG-60).
//!
//! # Robust Fitting
//!
//! When `robust=true`, the algorithm applies iterative reweighting using the bisquare
//! function to downweight outliers:
//! - w(u) = (1 - u^2)^2 for |u| < 1, else 0
//!
//! # Performance
//!
//! - Pre-sorting by x with binary search; each neighbourhood is O(q)
//! - Local fits solve the (degree + 1)-square moment system by an LDL' factorisation with
//!   lm.fit's relative pivot rule, with no allocation per point
//!
//! # References
//!
//! - Cleveland, W. S. (1979). "Robust locally weighted regression and smoothing
//!   scatterplots." Journal of the American Statistical Association, 74(368), 829-836.
//! - Cleveland, W. S., & Devlin, S. J. (1988). "Locally weighted regression: an
//!   approach to regression analysis by local fitting." Journal of the American
//!   Statistical Association, 83(403), 596-610.
//! - Cleveland, W. S., Grosse, E., & Shyu, W. M. (1992). "Local regression models."
//!   Chapter 8 of Statistical Models in S, eds J.M. Chambers and T.J. Hastie.
//! - R implementation: `stats::loess()` - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/loess.html

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::LM_QR_TOL;
use crate::traits::convergence::{ResultNotes, notes_text};

/// Configuration for LOESS fitting.
#[derive(Debug, Clone)]
pub struct LoessConfig {
    /// Smoothing parameter (proportion of points in each local fit).
    /// Default: 0.75. Range: (0, 1] for neighborhood proportion, >1 uses all points.
    pub span: f64,
    /// Polynomial degree for local fitting (1 = linear, 2 = quadratic).
    /// Default: 2.
    pub degree: usize,
    /// Use robust fitting with iterative reweighting.
    /// Default: false (gaussian family). Set true for "symmetric" family.
    pub robust: bool,
    /// Number of robustifying iterations (only used if robust=true).
    /// Default: 4.
    pub robust_iterations: usize,
    /// Convergence tolerance for robust iterations.
    /// Default: 1e-4.
    pub tolerance: f64,
}

impl Default for LoessConfig {
    fn default() -> Self {
        Self {
            span: 0.75,
            degree: 2,
            robust: false,
            robust_iterations: 4,
            tolerance: 1e-4,
        }
    }
}

/// LOESS model that can be used for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoessModel {
    /// Original x values
    pub x: Vec<f64>,
    /// Original y values
    pub y: Vec<f64>,
    /// Smoothing parameter
    pub span: f64,
    /// Polynomial degree
    pub degree: usize,
    /// Whether robust fitting was used
    pub robust: bool,
    /// Robustness weights (if robust=true)
    pub robust_weights: Option<Vec<f64>>,
}

/// Result from LOESS fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoessResult {
    /// The fitted LOESS model (can be used for prediction)
    #[serde(skip)]
    pub model: Option<LoessModel>,
    /// Fitted values at original x points
    pub fitted: Vec<f64>,
    /// Residuals (y - fitted)
    pub residuals: Vec<f64>,
    /// Smoothing parameter used
    pub span: f64,
    /// Polynomial degree used
    pub degree: usize,
    /// Whether robust fitting was used
    pub robust: bool,
    /// Number of observations
    pub n_obs: usize,
    /// Effective number of parameters (trace of hat matrix)
    pub enp: f64,
    /// Residual sum of squares
    pub rss: f64,
    /// Residual standard error
    pub residual_se: f64,
    /// R-squared (1 - RSS/TSS)
    pub r_squared: f64,
    /// Number of robust iterations performed
    pub robust_iterations: usize,
    /// Rows dropped because x or y was missing ([`run_loess`]); absent from JSON when
    /// nothing was dropped.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Notes on the fit (zero-width neighbourhoods, reduced local degrees, robustness
    /// fallbacks); absent from JSON when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

impl ResultNotes for LoessResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for LoessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LOESS Results")?;
        writeln!(f, "==============================================")?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "Span: {:.4}", self.span)?;
        writeln!(f, "Degree: {}", self.degree)?;
        writeln!(
            f,
            "Family: {}",
            if self.robust { "symmetric" } else { "gaussian" }
        )?;
        if self.robust {
            writeln!(f, "Robust Iterations: {}", self.robust_iterations)?;
        }
        writeln!(f)?;
        writeln!(f, "Equivalent Number of Parameters: {:.4}", self.enp)?;
        writeln!(f, "Residual Standard Error: {:.6}", self.residual_se)?;
        writeln!(f, "Residual Sum of Squares: {:.6}", self.rss)?;
        writeln!(f, "R-squared: {:.6}", self.r_squared)?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Tricubic weight function: w(u) = (1 - |u|^3)^3 for |u| < 1, else 0.
///
/// This is the default weight function for LOESS, giving more weight to nearby
/// points and smoothly decreasing to zero at the boundary of the neighborhood.
#[inline]
fn tricubic_weight(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u < 1.0 {
        let t = 1.0 - abs_u.powi(3);
        t.powi(3)
    } else {
        0.0
    }
}

/// Bisquare weight function: w(u) = (1 - u^2)^2 for |u| < 1, else 0.
///
/// Used for robust fitting (Tukey's biweight function).
#[inline]
fn bisquare_weight(u: f64) -> f64 {
    let abs_u = u.abs();
    if abs_u < 1.0 {
        let t = 1.0 - abs_u.powi(2);
        t.powi(2)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Engine: sorted data, R's neighbourhoods, moment-based local fits
// ---------------------------------------------------------------------------

/// Data sorted by x, with R's neighbourhood size and radius rule.
struct SortedData {
    /// x values in sorted order
    x_sorted: Vec<f64>,
    /// y values reordered to match x_sorted
    y_sorted: Vec<f64>,
    /// Mapping from sorted index back to original index
    orig_idx: Vec<usize>,
    /// Neighbourhood size: R `lowesd`'s `min(n, floor(n span + 1e-5))`, n for span >= 1.
    q: usize,
    /// Radius multiplier: `sqrt(span)` for span > 1, else 1.
    radius_mult: f64,
}

/// A neighbourhood `[left, right)` in sorted order and its radius (0: zero width).
#[derive(Debug, Clone, Copy)]
struct Nbhd {
    left: usize,
    right: usize,
    radius: f64,
}

/// One local fit: the fitted value at x0, `(X'WX)^{-1}[0, 0]` on the identified columns,
/// the effective degree, and what happened.
#[derive(Debug, Clone, Copy)]
struct LocalFit {
    fit: f64,
    c00: f64,
    degree: usize,
    zero_width: bool,
    robust_fallback: bool,
}

/// `q` for span `span` and `n` observations (R's `lowesd`).
fn neighbourhood_size(n: usize, span: f64) -> usize {
    if span < 1.0 {
        ((n as f64 * span + 1e-5).floor() as usize).clamp(1, n)
    } else {
        n
    }
}

impl SortedData {
    /// Sort the data and fix q and the radius rule. Refuses non-finite and constant x.
    fn new(x: &[f64], y: &[f64], span: f64) -> EconResult<Self> {
        let n = x.len();
        for (name, v) in [("x", x), ("y", y)] {
            if let Some(i) = v.iter().position(|t| !t.is_finite()) {
                return Err(EconError::InvalidSpecification {
                    message: format!("x and y must be finite; row {i} of {name} is {}", v[i]),
                });
            }
        }
        let mut order: Vec<usize> = crate::resource::try_with_capacity(n, "LOESS order")?;
        order.extend(0..n);
        order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let x_sorted: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let y_sorted: Vec<f64> = order.iter().map(|&i| y[i]).collect();
        if n > 0 && x_sorted[0] == x_sorted[n - 1] {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "x is constant ({n} values equal {}); a local regression on x needs at least two distinct x values",
                    x_sorted[0]
                ),
            });
        }
        // R multiplies the squared radius by span (ehg127 works on squared distances), so the
        // radius grows by sqrt(span), not span^(1/p) as documented: verified against
        // loess(surface = "direct") for spans 1.2-3.
        let radius_mult = if span > 1.0 { span.sqrt() } else { 1.0 };
        Ok(SortedData {
            x_sorted,
            y_sorted,
            orig_idx: order,
            q: neighbourhood_size(n, span),
            radius_mult,
        })
    }

    /// The q nearest points of x0 by distance (ties to the left), extended by every point
    /// at the q-th distance, so that a tie group is never split. O(q).
    fn neighbourhood(&self, x0: f64) -> Nbhd {
        let xs = &self.x_sorted;
        let n = xs.len();
        let pos = xs.partition_point(|&v| v < x0);
        let (mut l, mut r) = (pos, pos);
        let mut d_q = 0.0f64;
        for _ in 0..self.q {
            let dl = if l > 0 {
                (x0 - xs[l - 1]).abs()
            } else {
                f64::INFINITY
            };
            let dr = if r < n {
                (x0 - xs[r]).abs()
            } else {
                f64::INFINITY
            };
            if dl <= dr {
                l -= 1;
                d_q = dl;
            } else {
                r += 1;
                d_q = dr;
            }
        }
        while l > 0 && (x0 - xs[l - 1]).abs() <= d_q {
            l -= 1;
        }
        while r < n && (x0 - xs[r]).abs() <= d_q {
            r += 1;
        }
        Nbhd {
            left: l,
            right: r,
            radius: d_q * self.radius_mult,
        }
    }
}

/// Squared relative pivot tolerance: dqrdc2's column rule (`LM_QR_TOL`) on the sqrt(w)
/// weighted columns, squared because the moment matrix holds squared norms.
const MOMENT_PIVOT_TOL: f64 = LM_QR_TOL * LM_QR_TOL;

/// Local polynomial fit at x0 on the rows of `[left, right)` with positive weight.
///
/// The estimator is weighted least squares on `[1, u, u^2][..=degree]`, `u = (x - x0) /
/// radius` (|u| <= 1), solved through its moment matrix `M = [sum w u^(i+j)]` by an LDL'
/// factorisation in the column order 1, u, u^2. Column j is aliased, and dropped with every
/// higher power, when its pivot is at most `LM_QR_TOL^2 M_jj`: lm.fit's relative rank rule on
/// the weighted columns. No absolute cutoff and no allocation. A zero-width neighbourhood
/// (radius 0, every row at x0) gets weight 1 on each row and degree 0 (R `lowess`'s rule).
/// When every robustness weight in the neighbourhood is 0 they are ignored there.
///
/// Within the data (u on both sides of 0, or one side of a boundary point) M is well
/// conditioned and the solve agrees with the pivoted QR to about 1e-12. Far outside the data
/// (prediction by extrapolation) all u lie near one value and the moment matrix squares the
/// columns' condition, so the extrapolated value carries a relative error of order
/// cond^2 * 1e-16.
fn local_fit(
    sd: &SortedData,
    x0: f64,
    nb: &Nbhd,
    degree: usize,
    robust: Option<&[f64]>,
) -> EconResult<LocalFit> {
    let zero_width = nb.radius == 0.0;
    let degree = if zero_width { 0 } else { degree };
    let inv_r = if zero_width { 0.0 } else { 1.0 / nb.radius };
    let accumulate = |rw: Option<&[f64]>| {
        let mut s = [0.0f64; 5];
        let mut t = [0.0f64; 3];
        let mut m = 0usize;
        for i in nb.left..nb.right {
            let d = sd.x_sorted[i] - x0;
            let mut w = if zero_width {
                1.0
            } else {
                tricubic_weight(d.abs() * inv_r)
            };
            if let Some(rw) = rw {
                w *= rw[sd.orig_idx[i]];
            }
            if w > 0.0 {
                m += 1;
                let u = d * inv_r;
                let yi = sd.y_sorted[i];
                let mut p = w;
                for j in 0..=2 * degree {
                    s[j] += p;
                    if j <= degree {
                        t[j] += p * yi;
                    }
                    p *= u;
                }
            }
        }
        (s, t, m)
    };
    let (mut s, mut t, mut m) = accumulate(robust);
    let mut robust_fallback = false;
    if m == 0 && robust.is_some() {
        (s, t, m) = accumulate(None);
        robust_fallback = true;
    }
    if m == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "no observation has positive weight in the neighbourhood of x = {x0} (q = {}); increase span",
                sd.q
            ),
        });
    }
    let p = degree.min(m - 1) + 1;
    // LDL' of M (p <= 3) with the relative pivot rule.
    let mut l = [[0.0f64; 3]; 3];
    let mut dd = [0.0f64; 3];
    let mut kept = 0;
    for j in 0..p {
        let mjj = s[2 * j];
        let mut dj = mjj;
        for k in 0..j {
            dj -= l[j][k] * l[j][k] * dd[k];
        }
        if !(dj > MOMENT_PIVOT_TOL * mjj) {
            break;
        }
        dd[j] = dj;
        for i in (j + 1)..p {
            let mut v = s[i + j];
            for k in 0..j {
                v -= l[i][k] * l[j][k] * dd[k];
            }
            l[i][j] = v / dj;
        }
        kept = j + 1;
    }
    // kept >= 1: M_00 = sum w > 0.
    let solve0 = |b: &[f64; 3]| {
        let mut z = [0.0f64; 3];
        for i in 0..kept {
            let mut v = b[i];
            for k in 0..i {
                v -= l[i][k] * z[k];
            }
            z[i] = v;
        }
        for i in 0..kept {
            z[i] /= dd[i];
        }
        for i in (0..kept).rev() {
            let mut v = z[i];
            for k in (i + 1)..kept {
                v -= l[k][i] * z[k];
            }
            z[i] = v;
        }
        z[0]
    };
    let fit = solve0(&t);
    let c00 = solve0(&[1.0, 0.0, 0.0]);
    Ok(LocalFit {
        fit,
        c00,
        degree: kept - 1,
        zero_width,
        robust_fallback,
    })
}

/// What one pass over the data met, for the warnings.
#[derive(Debug, Default, Clone)]
struct FitStats {
    zero_width_points: usize,
    /// Distinct x values with a zero-width neighbourhood and their tie counts.
    zero_width_x: Vec<(f64, usize)>,
    degree_dropped: usize,
    robust_fallback: usize,
}

impl FitStats {
    fn merge(&mut self, other: FitStats) {
        if other.zero_width_points > self.zero_width_points {
            self.zero_width_points = other.zero_width_points;
            self.zero_width_x = other.zero_width_x;
        }
        self.degree_dropped = self.degree_dropped.max(other.degree_dropped);
        self.robust_fallback = self.robust_fallback.max(other.robust_fallback);
    }

    fn warnings(&self, n: usize, q: usize, span: f64) -> Vec<String> {
        let mut out = Vec::new();
        if self.zero_width_points > 0 {
            let t_max = self.zero_width_x.iter().map(|&(_, t)| t).max().unwrap_or(0);
            let mut xs: Vec<String> = self
                .zero_width_x
                .iter()
                .take(3)
                .map(|(v, _)| v.to_string())
                .collect();
            if self.zero_width_x.len() > 3 {
                xs.push("...".into());
            }
            let share = if self.zero_width_x.len() > 1 {
                format!("up to {t_max} observations share one x value")
            } else {
                format!("{t_max} observations share one x value")
            };
            out.push(format!(
                "zero-width neighbourhood at {} of {n} points (x = {}): {share}, at least the q = {q} \
                 neighbours of span {span}; the fit there is the weighted mean of the tied responses \
                 (R's lowess rule; R's loess returns 0 there and warns \"zero-width neighborhood. make \
                 span bigger\"); span >= {} avoids this",
                self.zero_width_points,
                xs.join(", "),
                (t_max + 1) as f64 / n as f64
            ));
        }
        if self.degree_dropped > 0 {
            out.push(format!(
                "local polynomial degree reduced at {} of {n} points: fewer distinct x values with \
                 positive weight than degree + 1",
                self.degree_dropped
            ));
        }
        if self.robust_fallback > 0 {
            out.push(format!(
                "robustness weights were all zero in {} neighbourhoods; the unweighted local fit was \
                 used there",
                self.robust_fallback
            ));
        }
        out
    }
}

/// Fit all data points: (fitted values in the original order, trace of the hat matrix, stats).
fn fit_all_points(
    sd: &SortedData,
    degree: usize,
    robust_weights: Option<&[f64]>,
    compute_hat: bool,
) -> EconResult<(Vec<f64>, f64, FitStats)> {
    let n = sd.x_sorted.len();
    let mut fitted = crate::resource::try_vec(n, 0.0, "LOESS fitted values")?;
    let mut hat_trace = 0.0;
    let mut stats = FitStats::default();
    for si in 0..n {
        if si % 256 == 0 {
            crate::resource::checkpoint()?;
        }
        let x0 = sd.x_sorted[si];
        let nb = sd.neighbourhood(x0);
        let lf = local_fit(sd, x0, &nb, degree, robust_weights)?;
        fitted[sd.orig_idx[si]] = lf.fit;
        if compute_hat {
            // the point's own tricube weight is 1; its robustness weight is 1 here
            hat_trace += lf.c00;
        }
        if lf.zero_width {
            stats.zero_width_points += 1;
            if stats.zero_width_x.last().is_none_or(|&(v, _)| v != x0) {
                stats.zero_width_x.push((x0, nb.right - nb.left));
            }
        } else if lf.degree < degree {
            stats.degree_dropped += 1;
        }
        if lf.robust_fallback {
            stats.robust_fallback += 1;
        }
    }
    Ok((fitted, hat_trace, stats))
}

/// Fit at a single evaluation point (used for prediction at new x values).
fn fit_single_point(
    sd: &SortedData,
    x0: f64,
    degree: usize,
    robust_weights: Option<&[f64]>,
) -> EconResult<f64> {
    let nb = sd.neighbourhood(x0);
    Ok(local_fit(sd, x0, &nb, degree, robust_weights)?.fit)
}

/// Compute robust weights using bisquare function.
///
/// Scale residuals by 6 * MAD (median absolute deviation) and apply bisquare.
fn compute_robust_weights(residuals: &[f64]) -> Vec<f64> {
    let n = residuals.len();
    if n == 0 {
        return vec![];
    }

    // Compute median of absolute residuals
    let mut abs_resid: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    abs_resid.sort_by(|a, b| a.total_cmp(b));
    let median_idx = n / 2;
    let mad = if n % 2 == 0 {
        (abs_resid[median_idx - 1] + abs_resid[median_idx]) / 2.0
    } else {
        abs_resid[median_idx]
    };

    // Scale factor: 6 * MAD (following R's loess implementation)
    let scale = 6.0 * mad;

    if scale < 1e-10 {
        // All residuals are essentially zero
        return vec![1.0; n];
    }

    // Compute bisquare weights
    residuals
        .iter()
        .map(|&r| bisquare_weight(r / scale))
        .collect()
}

/// Fit LOESS model to data.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `config` - LOESS configuration
///
/// # Returns
/// `LoessResult` containing fitted values, residuals, and model diagnostics.
///
/// # Example
/// ```ignore
/// use p2a_core::regression::loess::{loess, LoessConfig};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let y = vec![2.1, 4.2, 5.8, 8.1, 10.0, 11.9, 14.2, 15.8, 18.1, 20.0];
/// let result = loess(&x, &y, LoessConfig::default())?;
/// println!("{}", result);
/// ```
pub fn loess(x: &[f64], y: &[f64], config: LoessConfig) -> EconResult<LoessResult> {
    let n = x.len();

    // Validate inputs
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "LOESS requires at least 3 observations".to_string(),
        });
    }
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }
    if !(config.span.is_finite() && config.span > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!("span must be a finite positive number, got {}", config.span),
        });
    }
    if config.degree > 2 {
        return Err(EconError::InvalidSpecification {
            message: format!("degree must be 0, 1, or 2, got {}", config.degree),
        });
    }
    if config.robust && config.robust_iterations == 0 {
        return Err(EconError::InvalidSpecification {
            message: "robust_iterations must be at least 1 when robust = true".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0) {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "tolerance must be a finite positive number, got {}",
                config.tolerance
            ),
        });
    }

    // Sorted data with R's q and radius rule (refuses non-finite and constant x)
    let sd = SortedData::new(x, y, config.span)?;

    // Minimum observations needed for local regression
    let min_obs = config.degree + 1;
    if sd.q < min_obs {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "span too small: need at least {} observations per neighborhood for degree {}, but span gives {}",
                min_obs, config.degree, sd.q
            ),
        });
    }

    // Initial fit (also computes hat matrix trace for ENP)
    let (mut fitted, enp, mut stats) = fit_all_points(&sd, config.degree, None, true)?;

    // Compute residuals
    let mut residuals: Vec<f64> = y
        .iter()
        .zip(fitted.iter())
        .map(|(yi, fi)| yi - fi)
        .collect();

    // Robust fitting iterations
    let mut robust_weights: Option<Vec<f64>> = None;
    let mut iterations = 0;

    if config.robust {
        for iter in 0..config.robust_iterations {
            // Compute robust weights from residuals
            let new_robust_weights = compute_robust_weights(&residuals);

            // Refit with robust weights (no need for hat trace in robust iterations)
            let (new_fitted, _, pass) =
                fit_all_points(&sd, config.degree, Some(&new_robust_weights), false)?;
            stats.merge(pass);

            // Check convergence
            let max_change: f64 = fitted
                .iter()
                .zip(new_fitted.iter())
                .map(|(f1, f2)| (f1 - f2).abs())
                .fold(0.0, f64::max);

            fitted = new_fitted;
            residuals = y
                .iter()
                .zip(fitted.iter())
                .map(|(yi, fi)| yi - fi)
                .collect();
            robust_weights = Some(new_robust_weights);
            iterations = iter + 1;

            if max_change < config.tolerance {
                break;
            }
        }
    }
    let warnings = stats.warnings(n, sd.q, config.span);

    // Compute diagnostics
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let df = (n as f64 - enp).max(1.0);
    let residual_se = (rss / df).sqrt();
    let r_squared = if tss > 1e-10 { 1.0 - rss / tss } else { 0.0 };

    // Build model for prediction
    let model = LoessModel {
        x: x.to_vec(),
        y: y.to_vec(),
        span: config.span,
        degree: config.degree,
        robust: config.robust,
        robust_weights: robust_weights.clone(),
    };

    Ok(LoessResult {
        model: Some(model),
        fitted,
        residuals,
        span: config.span,
        degree: config.degree,
        robust: config.robust,
        n_obs: n,
        enp,
        rss,
        residual_se,
        r_squared,
        robust_iterations: iterations,
        missing: MissingSummary::none(n),
        warnings,
    })
}

/// Predict new values using a fitted LOESS model.
///
/// # Arguments
/// * `model` - Fitted LOESS model
/// * `new_x` - New x values for prediction
///
/// # Returns
/// Vector of predicted y values.
pub fn loess_predict(model: &LoessModel, new_x: &[f64]) -> EconResult<Vec<f64>> {
    // Build sorted data structure for the model's training data
    let sd = SortedData::new(&model.x, &model.y, model.span)?;
    if let Some(i) = new_x.iter().position(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: format!("new x values must be finite; value {i} is {}", new_x[i]),
        });
    }

    let mut predictions = crate::resource::try_with_capacity(new_x.len(), "LOESS predictions")?;

    for (i, &x0) in new_x.iter().enumerate() {
        if i % 256 == 0 {
            crate::resource::checkpoint()?;
        }
        let pred = fit_single_point(&sd, x0, model.degree, model.robust_weights.as_deref())?;
        predictions.push(pred);
    }

    Ok(predictions)
}

/// Fit LOESS model from a dataset.
///
/// # Arguments
/// * `dataset` - Dataset containing the data
/// * `y_col` - Name of the response column
/// * `x_col` - Name of the predictor column
/// * `span` - Smoothing parameter (default: 0.75)
/// * `degree` - Polynomial degree (default: 2)
/// * `robust` - Use robust fitting (default: false)
///
/// Rows with a missing x or y are dropped first, as R's `loess` does (`na.omit`), and
/// reported in `LoessResult::missing` (no complete row is `InsufficientData`); integer
/// columns are numbers.
///
/// # Returns
/// `LoessResult` containing fitted values, residuals, and model diagnostics.
pub fn run_loess(
    dataset: &Dataset,
    y_col: &str,
    x_col: &str,
    span: f64,
    degree: usize,
    robust: bool,
) -> EconResult<LoessResult> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(x_col)
        .numeric(y_col)
        .na(NaPolicy::Omit)
        .min_rows(1, &format!("LOESS of '{y_col}' on '{x_col}'"))
        .extract()?;

    let config = LoessConfig {
        span,
        degree,
        robust,
        ..Default::default()
    };

    let mut result = loess(cols.numeric(x_col), cols.numeric(y_col), config)?;
    result.missing = cols.missing().clone();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_tricubic_weight() {
        // At u = 0, weight should be 1
        assert!((tricubic_weight(0.0) - 1.0).abs() < 1e-10);

        // At u = 1, weight should be 0
        assert!(tricubic_weight(1.0).abs() < 1e-10);

        // At u > 1, weight should be 0
        assert!(tricubic_weight(1.5).abs() < 1e-10);

        // At u = 0.5, weight should be (1 - 0.125)^3 = 0.875^3 ~ 0.6699
        let expected = (1.0 - 0.5_f64.powi(3)).powi(3);
        assert!((tricubic_weight(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_bisquare_weight() {
        // At u = 0, weight should be 1
        assert!((bisquare_weight(0.0) - 1.0).abs() < 1e-10);

        // At u = 1, weight should be 0
        assert!(bisquare_weight(1.0).abs() < 1e-10);

        // At u > 1, weight should be 0
        assert!(bisquare_weight(1.5).abs() < 1e-10);

        // At u = 0.5, weight should be (1 - 0.25)^2 = 0.5625
        let expected = (1.0 - 0.5_f64.powi(2)).powi(2);
        assert!((bisquare_weight(0.5) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_loess_linear_data() {
        // Linear data should be fit well
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + 1.0 + 0.1 * (xi * 10.0).sin())
            .collect();

        let result = loess(&x, &y, LoessConfig::default()).unwrap();

        // Check that fitted values are close to actual
        assert!(result.r_squared > 0.95);
        assert_eq!(result.n_obs, 20);
        assert_eq!(result.fitted.len(), 20);
        assert_eq!(result.residuals.len(), 20);
    }

    #[test]
    fn test_loess_with_different_spans() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin() + 0.1 * xi).collect();

        // Small span = less smooth
        let result_small = loess(
            &x,
            &y,
            LoessConfig {
                span: 0.3,
                ..Default::default()
            },
        )
        .unwrap();

        // Large span = more smooth
        let result_large = loess(
            &x,
            &y,
            LoessConfig {
                span: 0.9,
                ..Default::default()
            },
        )
        .unwrap();

        // Both should complete without error
        assert!(result_small.enp > 0.0);
        assert!(result_large.enp > 0.0);

        // Smaller span should have more effective parameters (more wiggly)
        // This isn't always strictly true but generally holds
    }

    #[test]
    fn test_loess_degree_1_vs_2() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi / 100.0 + 0.5 * xi).collect();

        let result_linear = loess(
            &x,
            &y,
            LoessConfig {
                degree: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let result_quad = loess(
            &x,
            &y,
            LoessConfig {
                degree: 2,
                ..Default::default()
            },
        )
        .unwrap();

        // Both should work
        assert!(result_linear.r_squared > 0.9);
        assert!(result_quad.r_squared > 0.9);
    }

    #[test]
    fn test_loess_robust() {
        let x: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let mut y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        // Add outliers
        y[10] = 100.0; // Far outlier
        y[20] = -50.0; // Another outlier

        let _result_normal = loess(
            &x,
            &y,
            LoessConfig {
                robust: false,
                ..Default::default()
            },
        )
        .unwrap();
        let result_robust = loess(
            &x,
            &y,
            LoessConfig {
                robust: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Robust should have lower RSS (outliers downweighted)
        // In this case, robust should better fit the underlying linear trend
        assert!(result_robust.robust_iterations > 0);
    }

    #[test]
    fn test_loess_predict() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let result = loess(&x, &y, LoessConfig::default()).unwrap();
        let model = result.model.unwrap();

        // Predict at new points
        let new_x = vec![5.5, 10.5, 15.5];
        let predictions = loess_predict(&model, &new_x).unwrap();

        assert_eq!(predictions.len(), 3);

        // Predictions should be close to 2*x + 1 for linear data
        for (i, &xi) in new_x.iter().enumerate() {
            let expected = 2.0 * xi + 1.0;
            assert!(
                (predictions[i] - expected).abs() < 1.0,
                "Prediction at {} = {}, expected ~{}",
                xi,
                predictions[i],
                expected
            );
        }
    }

    #[test]
    fn test_loess_errors() {
        // Too few observations
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0];
        assert!(loess(&x, &y, LoessConfig::default()).is_err());

        // Mismatched lengths
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(loess(&x, &y, LoessConfig::default()).is_err());

        // Invalid span
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(
            loess(
                &x,
                &y,
                LoessConfig {
                    span: 0.0,
                    ..Default::default()
                }
            )
            .is_err()
        );
        assert!(
            loess(
                &x,
                &y,
                LoessConfig {
                    span: -0.5,
                    ..Default::default()
                }
            )
            .is_err()
        );

        // Invalid degree
        assert!(
            loess(
                &x,
                &y,
                LoessConfig {
                    degree: 3,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }

    fn col(ds: &Dataset, n: &str) -> Vec<f64> {
        crate::data::columns::numeric_column(ds.df(), n, NaPolicy::Fail)
            .unwrap()
            .0
    }

    fn fit(x: &[f64], y: &[f64], span: f64, degree: usize) -> LoessResult {
        let mut c = LoessConfig::default();
        c.span = span;
        c.degree = degree;
        loess(x, y, c).unwrap()
    }

    /// REG-15: 60 of the 100 x values are 0; span 0.5 uses q = 50 neighbours, so the tie
    /// group exceeds q. Data only: the R pin is `test_validate_loess_mass_point`.
    #[test]
    fn test_loess_mass_point_no_panic() {
        let ds = crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/smoothers_t3.csv",
        );
        let (x, y) = (col(&ds, "xm"), col(&ds, "ym"));
        for degree in 0..=2 {
            let r = fit(&x, &y, 0.5, degree);
            assert!(r.fitted.iter().all(|f| f.is_finite()), "degree {degree}");
            let pred = loess_predict(r.model.as_ref().unwrap(), &[0.0, 0.25, 0.9]).unwrap();
            assert!(pred.iter().all(|f| f.is_finite()));
        }
        // a Likert-style mass point: 16 of 40 at 3.0, span 0.3 (q = 12)
        let xl: Vec<f64> = (0..40)
            .map(|i| if i < 16 { 3.0 } else { 10.0 + i as f64 })
            .collect();
        let yl: Vec<f64> = xl
            .iter()
            .enumerate()
            .map(|(i, v)| 0.5 * v + 0.1 * ((i * 7) % 5) as f64)
            .collect();
        assert!(fit(&xl, &yl, 0.3, 1).fitted.iter().all(|f| f.is_finite()));
        // a constant predictor is an error naming x, not a panic
        let yc: Vec<f64> = (0..23).map(|i| 1.0 + 0.3 * ((i * 5) % 7) as f64).collect();
        for span in [0.75, 1.0] {
            let mut c = LoessConfig::default();
            c.span = span;
            match loess(&[2.0; 23], &yc, c).unwrap_err() {
                EconError::InvalidSpecification { message } => {
                    assert!(message.contains("x is constant"), "{message}")
                }
                e => panic!("{e:?}"),
            }
        }
    }

    /// Case `smoothers_gcv_periodic_loess`: R loess(surface = "direct") where the
    /// neighbourhood has positive width; at the 60 tied zeros R returns 0 ("zero-width
    /// neighborhood"), and the pin is lowess's rule (weight 1 on the ties: their mean,
    /// `lowess_masspoint_y0` = 0.9731694359533333), decision P2.
    #[test]
    fn test_validate_loess_mass_point() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("smoothers_gcv_periodic_loess");
        let (x, y) = (col(&ds, "xm"), col(&ds, "ym"));
        let r = fit(&x, &y, 0.5, 1);
        Check::new(&case)
            .vector("loess_masspoint_expected", &r.fitted, Tol::coef(100))
            .scalar("lowess_masspoint_y0", r.fitted[0], Tol::coef(100))
            .finish();
        let w: Vec<&String> = r
            .warnings
            .iter()
            .filter(|w| w.contains("zero-width"))
            .collect();
        assert_eq!(w.len(), 1, "{:?}", r.warnings);
        assert!(
            w[0].contains("60 of 100") && w[0].contains("q = 50") && w[0].contains("0.61"),
            "{}",
            w[0]
        );
        assert!(
            r.to_string().contains("zero-width"),
            "Display ends with the notes"
        );
    }

    /// Case `loess_x10_direct` (REG-44 site; REG-45: span * n = 7.5 -> q = 7; P3: span > 1).
    #[test]
    fn test_validate_loess_x10_direct() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("loess_x10_direct");
        let (x, y) = (col(&ds, "x"), col(&ds, "y"));
        let n = case.scalar("n") as usize;
        let (r2, r1, r0, r12) = (
            fit(&x, &y, 0.75, 2),
            fit(&x, &y, 0.75, 1),
            fit(&x, &y, 0.5, 0),
            fit(&x, &y, 1.2, 2),
        );
        let pred = loess_predict(r2.model.as_ref().unwrap(), &case.vector("predict_x")).unwrap();
        Check::new(&case)
            .vector("fit_s075_d2", &r2.fitted, Tol::coef(n))
            .scalar("trace_s075_d2", r2.enp, Tol::stat(n))
            .vector("fit_s075_d1", &r1.fitted, Tol::coef(n))
            .scalar("trace_s075_d1", r1.enp, Tol::stat(n))
            .vector("fit_s05_d0", &r0.fitted, Tol::coef(n))
            .vector("fit_s12_d2", &r12.fitted, Tol::coef(n))
            .vector("predict_s075_d2", &pred, Tol::coef(n))
            .scalar("n", r2.n_obs as f64, Tol::exact())
            .finish();
        // deviation REG-60: enp_r_s075_d2 is R's "Equivalent Number of Parameters"
        // (one.delta + 2 tr(L) - n); p2a's `enp` is tr(L), R's trace.hat. Residual SE and
        // robust iterations are not compared either.
        assert!(
            r2.warnings.is_empty() && r12.warnings.is_empty(),
            "{:?}",
            r2.warnings
        );
    }

    /// Case `loess_skewed_direct` (REG-14: neighbourhoods mostly on one side).
    #[test]
    fn test_validate_loess_skewed_direct() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("loess_skewed_direct");
        let (x, y) = (col(&ds, "x"), col(&ds, "y"));
        let (a, b) = (fit(&x, &y, 0.75, 2), fit(&x, &y, 0.3, 1));
        Check::new(&case)
            .vector("fit_s075_d2", &a.fitted, Tol::coef(100))
            .scalar("trace_s075_d2", a.enp, Tol::stat(100))
            .vector("fit_s03_d1", &b.fitted, Tol::coef(100))
            .scalar("trace_s03_d1", b.enp, Tol::stat(100))
            .finish();
    }

    /// REG-46: local polynomial fitting is affine-invariant in x; x * 1e-6 and x * 1e6
    /// give the same fit, as in R.
    #[test]
    fn test_validate_loess_scale_invariance() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("loess_x10_direct");
        let (x, y) = (col(&ds, "x"), col(&ds, "y"));
        let base = fit(&x, &y, 0.75, 2);
        for s in [1e-6, 1e6] {
            let xs: Vec<f64> = x.iter().map(|v| v * s).collect();
            let r = fit(&xs, &y, 0.75, 2);
            for (a, b) in r.fitted.iter().zip(&base.fitted) {
                assert!((a - b).abs() <= 1e-9 * b.abs(), "x*{s}: {a} vs {b}");
            }
            assert!(
                (r.enp - base.enp).abs() <= 1e-9 * base.enp,
                "x*{s}: enp {} vs {}",
                r.enp,
                base.enp
            );
            assert!(r.warnings.is_empty(), "x*{s}: {:?}", r.warnings);
        }
        Check::new(&case)
            .vector("fit_s075_d2", &base.fitted, Tol::coef(10))
            .finish();
    }

    #[test]
    fn test_loess_rejects_invalid_controls() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.7).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.1 * ((i * 3) % 4) as f64)
            .collect();
        let bad = |c: LoessConfig, x: &[f64], what: &str| match loess(x, &y, c).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains(what), "{what}: {message}")
            }
            e => panic!("{what}: {e:?}"),
        };
        for span in [f64::NAN, 0.0, -0.5] {
            let mut c = LoessConfig::default();
            c.span = span;
            bad(c, &x, "span");
        }
        let mut c = LoessConfig::default();
        c.degree = 3;
        bad(c, &x, "degree");
        let mut c = LoessConfig::default();
        c.robust = true;
        c.robust_iterations = 0;
        bad(c, &x, "robust_iterations");
        for tol in [0.0, f64::NAN] {
            let mut c = LoessConfig::default();
            c.tolerance = tol;
            bad(c, &x, "tolerance");
        }
        let mut xn = x.clone();
        xn[4] = f64::NAN;
        bad(LoessConfig::default(), &xn, "finite");
    }

    /// WP11-4: the moment solver of `local_fit` agrees with least squares (pivoted QR,
    /// lm.fit's rank rule) on sqrt(w) [1, u, u^2] at every point: fit, h_ii and degree.
    #[test]
    fn test_loess_local_solver_matches_qr() {
        use crate::linalg::solve::least_squares;
        // uneven x with a run of ties (two distinct values in some windows) and noise
        let mut x: Vec<f64> = (0..200)
            .map(|i| {
                let t = i as f64 / 199.0;
                (t * t * 7.0 + ((i * 37) % 13) as f64 * 0.01).max(0.0)
            })
            .collect();
        for v in x.iter_mut().take(30) {
            *v = (*v * 4.0).round() / 4.0;
        }
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.2 * (((i * 53) % 17) as f64 / 17.0 - 0.5))
            .collect();
        let mut checked_drop = false;
        for degree in 0..=2 {
            for span in [0.1, 0.5, 1.2] {
                let sd = SortedData::new(&x, &y, span).unwrap();
                let mut evals: Vec<f64> = sd.x_sorted.clone();
                // points between the data (not extrapolation: there all neighbours lie on one side
                // and the moment matrix squares the condition of the columns)
                evals.extend([0.13, 3.3, 6.9]);
                for &x0 in &evals {
                    let nb = sd.neighbourhood(x0);
                    let lf = local_fit(&sd, x0, &nb, degree, None).unwrap();
                    // the reference: QR on the positive-weight rows
                    let rows: Vec<usize> = (nb.left..nb.right)
                        .filter(|&i| {
                            nb.radius == 0.0
                                || tricubic_weight((sd.x_sorted[i] - x0).abs() / nb.radius) > 0.0
                        })
                        .collect();
                    let d = if nb.radius == 0.0 {
                        0
                    } else {
                        degree.min(rows.len() - 1)
                    };
                    let mut xm = Array2::<f64>::zeros((rows.len(), d + 1));
                    let mut yv = Array1::<f64>::zeros(rows.len());
                    for (r, &i) in rows.iter().enumerate() {
                        let w: f64 = if nb.radius == 0.0 {
                            1.0
                        } else {
                            tricubic_weight((sd.x_sorted[i] - x0).abs() / nb.radius)
                        };
                        let u = if nb.radius == 0.0 {
                            0.0
                        } else {
                            (sd.x_sorted[i] - x0) / nb.radius
                        };
                        for j in 0..=d {
                            xm[[r, j]] = w.sqrt() * u.powi(j as i32);
                        }
                        yv[r] = w.sqrt() * sd.y_sorted[i];
                    }
                    let ls = least_squares(xm.view(), yv.view()).unwrap();
                    assert_eq!(
                        lf.degree,
                        ls.rank - 1,
                        "degree {degree} span {span} x0 {x0}"
                    );
                    checked_drop |= lf.degree < d;
                    let b0 = ls.coefficients[0];
                    let c00 = ls.unscaled_cov[[0, 0]];
                    assert!(
                        (lf.fit - b0).abs() <= 1e-10 * b0.abs().max(1e-3),
                        "fit {} vs {b0} (degree {degree} span {span} x0 {x0})",
                        lf.fit
                    );
                    assert!(
                        (lf.c00 - c00).abs() <= 1e-10 * c00.abs(),
                        "c00 {} vs {c00}",
                        lf.c00
                    );
                }
            }
        }
        assert!(
            checked_drop,
            "the data must exercise a reduced local degree"
        );
    }

    #[test]
    fn test_loess_degree_0() {
        // Degree 0 = local constant (Nadaraya-Watson)
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 2.0 * xi + 1.0 + 0.1 * (xi * 10.0).sin())
            .collect();

        let result = loess(
            &x,
            &y,
            LoessConfig {
                degree: 0,
                span: 0.5,
                ..Default::default()
            },
        )
        .unwrap();

        // Should produce reasonable fit
        assert!(result.r_squared > 0.8);
        assert_eq!(result.degree, 0);
        assert_eq!(result.fitted.len(), 20);
    }

    #[test]
    fn test_loess_sorted_vs_unsorted_input() {
        // Verify that the optimized (sorted) path gives the same results
        // as the legacy (unsorted) path, even when input is scrambled.
        let x_sorted: Vec<f64> = (0..15).map(|i| i as f64).collect();
        let y_sorted: Vec<f64> = x_sorted.iter().map(|&xi| xi.sin() + xi * 0.5).collect();

        // Scramble the order
        let perm = vec![7, 3, 12, 0, 9, 5, 14, 1, 11, 6, 2, 10, 4, 13, 8];
        let x_scrambled: Vec<f64> = perm.iter().map(|&i| x_sorted[i]).collect();
        let y_scrambled: Vec<f64> = perm.iter().map(|&i| y_sorted[i]).collect();

        let config = LoessConfig {
            span: 0.5,
            degree: 1,
            ..Default::default()
        };

        let result_sorted = loess(&x_sorted, &y_sorted, config.clone()).unwrap();
        let result_scrambled = loess(&x_scrambled, &y_scrambled, config).unwrap();

        // Compare fitted values (need to account for different ordering)
        for i in 0..15 {
            let orig_idx = perm[i];
            let diff = (result_scrambled.fitted[i] - result_sorted.fitted[orig_idx]).abs();
            assert!(
                diff < 1e-10,
                "Mismatch at scrambled[{}] (orig {}): {} vs {}, diff={}",
                i,
                orig_idx,
                result_scrambled.fitted[i],
                result_sorted.fitted[orig_idx],
                diff
            );
        }
    }

    /// REG-25: run_loess reads x and y with R's na.omit (loess drops incomplete rows) and
    /// accepts Int64 columns. R-pinned: the number of rows used is R's `n_complete`; the
    /// fit itself is not (REG-14), so it is compared with the fit on the complete rows.
    #[test]
    fn test_validate_na_loess_columns() {
        use crate::regression::ols::na_fixtures::{as_float, complete_rows, mtcars};
        use p2a_validation::{Check, Tol};
        let mt = mtcars();
        // x = hp is Int64
        let a = run_loess(&mt, "mpg", "hp", 0.75, 2, false).unwrap();
        let b = run_loess(&as_float(&mt, "hp"), "mpg", "hp", 0.75, 2, false).unwrap();
        assert_eq!((a.n_obs, &a.fitted, a.rss), (32, &b.fitted, b.rss));
        let (case, sm) = crate::test_support::load_case("reg_smoothers_na");
        let r = run_loess(&sm, "y", "x", 0.75, 2, false).unwrap();
        Check::new(&case)
            .scalar("n_complete", r.n_obs as f64, Tol::exact())
            .finish();
        let c = run_loess(&complete_rows(&sm, &["x", "y"]), "y", "x", 0.75, 2, false).unwrap();
        assert_eq!(
            (r.n_obs, &r.fitted, &r.residuals),
            (56, &c.fitted, &c.residuals)
        );
        assert_eq!(
            r.missing,
            MissingSummary {
                n_total: 60,
                n_used: 56,
                n_dropped: 4,
                by_column: vec![("x".into(), 2), ("y".into(), 2)],
            }
        );
        assert!(r.to_string().ends_with(
            "Missing values: 4 of 60 rows dropped because of missing values (x: 2, y: 2)\n"
        ));
        assert!(c.missing.is_empty() && !c.to_string().contains("Missing values"));
        assert!(serde_json::to_value(&c).unwrap().get("missing").is_none());
        assert!(matches!(
            run_loess(&mt, "mpg", "model", 0.75, 2, false),
            Err(EconError::NonNumericColumn { .. })
        ));
        assert!(matches!(
            run_loess(&mt, "mpg", "nope", 0.75, 2, false),
            Err(EconError::ColumnNotFound { .. })
        ));
        // No complete row: the error names the method and the columns.
        use crate::regression::ols::na_fixtures::{insufficient_context, with_all_null};
        let none = with_all_null(&sm, "no_x", polars::prelude::DataType::Float64);
        assert_eq!(
            insufficient_context(run_loess(&none, "y", "no_x", 0.75, 2, false)),
            "LOESS of 'y' on 'no_x': 0 of 60 rows have no missing value"
        );
    }
}
