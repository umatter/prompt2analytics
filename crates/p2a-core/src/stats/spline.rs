//! Spline interpolation functions.
//!
//! Implements R's `spline()` and `approx()` functions for interpolation.
//!
//! # References
//!
//! - R Core Team. `stats::spline()` and `stats::approx()` functions.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/splinefun.html>
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/approxfun.html>
//! - Forsythe, G. E., Malcolm, M. A., & Moler, C. B. (1977).
//!   "Computer Methods for Mathematical Computations". Prentice-Hall.

use crate::errors::{EconError, EconResult};
use crate::resource;
use serde::{Deserialize, Serialize};

/// Method for spline interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum SplineMethod {
    /// Natural spline with zero second derivatives at endpoints.
    #[default]
    Natural,
    /// FMM spline (Forsythe, Malcolm, Moler) - exact cubic through 4 points at each end.
    Fmm,
    /// Periodic spline for cyclical data.
    Periodic,
    /// Monotone Hermite spline with Fritsch-Carlson slopes (R's `splinefun(method = "monoH.FC")`),
    /// linear outside the knots.
    MonotoneFC,
    /// Hyman's monotone spline (R's `spline(method = "hyman")`): the FMM spline's slopes filtered
    /// to keep monotone data monotone. Non-monotone y is an error.
    Hyman,
}

crate::option_enum!(
    SplineMethod,
    crate::options::Choices {
        option: "spline method",
        values: &[
            (&["natural"], SplineMethod::Natural),
            (&["fmm"], SplineMethod::Fmm),
            (&["periodic"], SplineMethod::Periodic),
            (
                &["monoH.FC", "monotonefc", "monotone"],
                SplineMethod::MonotoneFC
            ),
            (&["hyman"], SplineMethod::Hyman),
        ],
    }
);

/// Method for linear/constant interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ApproxMethod {
    /// Linear interpolation between points.
    #[default]
    Linear,
    /// Constant (step function) interpolation.
    Constant,
}

crate::option_enum!(
    ApproxMethod,
    crate::options::Choices {
        option: "approx method",
        values: &[
            (&["linear"], ApproxMethod::Linear),
            (&["constant", "step"], ApproxMethod::Constant),
        ],
    }
);

/// Rule for handling values outside the interpolation range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum ApproxRule {
    /// Return NA for points outside range.
    #[default]
    Na,
    /// Return the nearest endpoint value.
    Nearest,
}

crate::option_enum!(
    ApproxRule,
    crate::options::Choices {
        option: "approx rule",
        values: &[
            (&["na", "1"], ApproxRule::Na),
            (&["nearest", "boundary", "const", "2"], ApproxRule::Nearest),
        ],
    }
);

/// Result of spline interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SplineResult {
    /// Interpolated x values.
    pub x: Vec<f64>,
    /// Interpolated y values.
    pub y: Vec<f64>,
    /// Method used for interpolation.
    pub method: SplineMethod,
    /// Number of output points.
    pub n: usize,
    /// R's warnings: "collapsing to unique 'x' values" when tied x were averaged, and for a
    /// periodic spline whose first and last y differ.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Result of approx interpolation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproxResult {
    /// Interpolated x values.
    pub x: Vec<f64>,
    /// Interpolated y values (may contain NaN for out-of-range points).
    pub y: Vec<f64>,
    /// Method used for interpolation.
    pub method: ApproxMethod,
    /// Number of output points.
    pub n: usize,
    /// R's warning "collapsing to unique 'x' values" when tied x were averaged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// R's warning when tied x values are averaged (`regularize.values(ties = mean)`).
const TIES_WARNING: &str = "collapsing to unique 'x' values";

/// R's warning when a periodic spline's first and last y differ.
const PERIODIC_WARNING: &str = "spline: first and last y values differ - using y[1] for both";

/// Perform cubic spline interpolation.
///
/// Interpolates the given data points using cubic splines. The result is
/// a smooth curve passing through all data points.
///
/// # Arguments
///
/// * `x` - Input x values (at least 4 distinct values; sorted internally). Tied x values are
///   collapsed to the mean of their y values with R's warning "collapsing to unique 'x' values".
/// * `y` - Input y values (same length as x)
/// * `xout` - Optional x values at which to interpolate. If None, uses n equally spaced points.
/// * `n` - Number of interpolation points if xout is None (R's default: 3 times the number of
///   distinct x values)
/// * `method` - Spline method to use
///
/// # Returns
///
/// `SplineResult` containing interpolated x and y values.
///
/// # Mathematical Background
///
/// Cubic splines are piecewise cubic polynomials that:
/// 1. Pass through all data points
/// 2. Have continuous first and second derivatives
/// 3. Satisfy boundary conditions specified by the method
///
/// As R's `spline`: the natural spline has `S''(x_0) = S''(x_n) = 0` and continues linearly
/// outside the knots; the FMM spline matches, on each end interval, the third derivative of the
/// cubic through the four end points, and continues its end cubics; the periodic spline wraps
/// `x` into the knots' range (and uses `y[1]` for both ends, with R's warning, when they
/// differ); "hyman" filters the FMM slopes to keep monotone data monotone (non-monotone y is an
/// error) and continues its end cubics; monoH.FC (Fritsch-Carlson, `splinefun(method =
/// "monoH.FC")`) continues linearly.
///
/// # References
///
/// - R function `stats::spline()`
/// - Forsythe, Malcolm, Moler (1977). "Computer Methods for Mathematical Computations".
/// - Hyman, J. M. (1983). Accurate monotonicity preserving cubic interpolation. *SIAM J. Sci.
///   Stat. Comput.* 4, 645-654.
/// - Fritsch, F. N. and Carlson, R. E. (1980). Monotone piecewise cubic interpolation. *SIAM J.
///   Numer. Anal.* 17, 238-246.
pub fn spline(
    x: &[f64],
    y: &[f64],
    xout: Option<&[f64]>,
    n: Option<usize>,
    method: SplineMethod,
) -> EconResult<SplineResult> {
    let mut warnings = Vec::new();
    let fitted = fit_spline(x, y, method, &mut warnings)?;

    // Determine output x values (R's default n = 3 * length(unique x))
    let x_out = output_points(&fitted.x, xout, n, 3 * fitted.x.len())?;

    // Evaluate spline at output points
    let mut y_out: Vec<f64> = resource::try_with_capacity(x_out.len(), "interpolation points")?;
    for (i, &xi) in x_out.iter().enumerate() {
        if i % 65_536 == 0 {
            resource::checkpoint()?;
        }
        y_out.push(fitted.eval(xi));
    }

    Ok(SplineResult {
        n: x_out.len(),
        x: x_out,
        y: y_out,
        method,
        warnings,
    })
}

/// Perform linear or constant interpolation.
///
/// Interpolates the given data points using either linear interpolation
/// or a step function (constant interpolation).
///
/// # Arguments
///
/// * `x` - Input x values (sorted internally). Tied x values are collapsed to the mean of
///   their y values with R's warning "collapsing to unique 'x' values".
/// * `y` - Input y values (same length as x)
/// * `xout` - Optional x values at which to interpolate. If None, uses n equally spaced points.
/// * `n` - Number of interpolation points if xout is None (default 50)
/// * `method` - Interpolation method (Linear or Constant)
/// * `rule` - How to handle values outside the x range
/// * `f` - For the constant method, R's `f`: between knots the value is
///   `(1 - f) y_left + f y_right`, so `f = 0` is right-continuous (the left knot's value on each
///   interval) and `f = 1` left-continuous. At a knot the value is that knot's own y.
///
/// # Returns
///
/// `ApproxResult` containing interpolated x and y values.
///
/// # References
///
/// - R function `stats::approx()`
pub fn approx(
    x: &[f64],
    y: &[f64],
    xout: Option<&[f64]>,
    n: Option<usize>,
    method: ApproxMethod,
    rule: ApproxRule,
    f: f64,
) -> EconResult<ApproxResult> {
    let (xs, ys, tied) = prepare_approx_knots(x, y)?;

    // Determine output x values (R's default n = 50)
    let x_out = output_points(&xs, xout, n, 50)?;

    // Interpolate at output points
    let mut y_out: Vec<f64> = resource::try_with_capacity(x_out.len(), "interpolation points")?;
    for (i, &xi) in x_out.iter().enumerate() {
        if i % 65_536 == 0 {
            resource::checkpoint()?;
        }
        y_out.push(interpolate_approx(&xs, &ys, xi, method, rule, f));
    }

    Ok(ApproxResult {
        n: x_out.len(),
        x: x_out,
        y: y_out,
        method,
        warnings: if tied {
            vec![TIES_WARNING.to_string()]
        } else {
            Vec::new()
        },
    })
}

/// Create an interpolating function from data points.
///
/// This is a convenience wrapper that returns a closure for evaluating
/// the interpolation at arbitrary points.
///
/// The knots are validated as in [`spline`]: at least 4 distinct finite x values (ties are
/// averaged); its warnings are not returned.
pub fn splinefun(x: &[f64], y: &[f64], method: SplineMethod) -> EconResult<impl Fn(f64) -> f64> {
    let fitted = fit_spline(x, y, method, &mut Vec::new())?;
    Ok(move |xi: f64| fitted.eval(xi))
}

/// Create an approx interpolating function from data points.
///
/// The knots are validated as in [`approx`]: at least 2 distinct x values with finite x and y.
pub fn approxfun(
    x: &[f64],
    y: &[f64],
    method: ApproxMethod,
    rule: ApproxRule,
    f: f64,
) -> EconResult<impl Fn(f64) -> f64> {
    let (xs, ys, _) = prepare_approx_knots(x, y)?;

    Ok(move |xi: f64| interpolate_approx(&xs, &ys, xi, method, rule, f))
}

// ========== Internal helper functions ==========

/// Sort `(x, y)` pairs by x (`total_cmp`) and collapse each run of exactly equal x into one knot
/// with the mean of its y values, as R's `regularize.values(ties = mean)`. Returns the knots and
/// whether any tie was collapsed.
fn sort_and_collapse(mut pairs: Vec<(f64, f64)>) -> (Vec<f64>, Vec<f64>, bool) {
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut xs = Vec::with_capacity(pairs.len());
    let mut ys = Vec::with_capacity(pairs.len());
    let mut tied = false;
    let mut i = 0;
    while i < pairs.len() {
        let x0 = pairs[i].0;
        let mut sum = 0.0;
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == x0 {
            sum += pairs[j].1;
            j += 1;
        }
        tied |= j - i > 1;
        xs.push(x0);
        ys.push(if j - i > 1 { sum / (j - i) as f64 } else { sum });
        i = j;
    }
    (xs, ys, tied)
}

/// Validate and sort cubic-spline knots: equal lengths, finite values and at least 4 distinct x
/// values after tied x are averaged (R's `regularize.values(ties = mean)`). Returns the sorted
/// knots and whether a tie was collapsed.
fn prepare_spline_knots(x: &[f64], y: &[f64]) -> EconResult<(Vec<f64>, Vec<f64>, bool)> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    // Check for non-finite values
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "Input contains non-finite values (NaN or Inf)".to_string(),
        });
    }

    let (xs, ys, tied) = sort_and_collapse(x.iter().copied().zip(y.iter().copied()).collect());
    if xs.len() < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: xs.len(),
            context: "cubic spline interpolation (distinct x values)".to_string(),
        });
    }
    Ok((xs, ys, tied))
}

/// Keep the points with finite x and y, sort them by x, average tied x and require at least 2
/// distinct x values. Returns the knots and whether a tie was collapsed.
fn prepare_approx_knots(x: &[f64], y: &[f64]) -> EconResult<(Vec<f64>, Vec<f64>, bool)> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    if x.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: x.len(),
            context: "interpolation".to_string(),
        });
    }

    let pairs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .filter(|(xi, yi)| xi.is_finite() && yi.is_finite())
        .map(|(&xi, &yi)| (xi, yi))
        .collect();
    let (xs, ys, tied) = sort_and_collapse(pairs);

    if xs.len() < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: xs.len(),
            context: "interpolation (distinct x values, after removing NA values)".to_string(),
        });
    }
    Ok((xs, ys, tied))
}

/// The output points: a copy of `xout`, or `n` (or `default_n`) equally spaced points over the
/// knots' range. `n = 0` is no point and `n = 1` the lower end, as `seq.int(.., length.out = n)`.
/// The buffer is allocated fallibly.
fn output_points(
    xs: &[f64],
    xout: Option<&[f64]>,
    n: Option<usize>,
    default_n: usize,
) -> EconResult<Vec<f64>> {
    match xout {
        Some(xo) => {
            let mut out = resource::try_with_capacity(xo.len(), "interpolation points")?;
            out.extend_from_slice(xo);
            Ok(out)
        }
        None => {
            let n_pts = n.unwrap_or(default_n);
            let x_min = xs[0];
            let x_max = xs[xs.len() - 1];
            let mut out = resource::try_with_capacity(n_pts, "interpolation points")?;
            if n_pts == 1 {
                out.push(x_min);
            } else {
                out.extend(
                    (0..n_pts).map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1) as f64),
                );
            }
            Ok(out)
        }
    }
}

/// How a spline is represented between the knots.
enum SplineShape {
    /// A cubic spline given by its second derivatives at the knots (natural, FMM, periodic).
    SecondDerivs(Vec<f64>),
    /// A cubic Hermite spline given by its slopes at the knots (hyman, monoH.FC).
    Slopes(Vec<f64>),
}

/// How a spline continues outside the knots (R's `spline_eval` and `splinefunH0`).
#[derive(Clone, Copy, PartialEq)]
enum Extrapolation {
    /// Linear, with the end value and end slope (natural, monoH.FC).
    Linear,
    /// The end intervals' cubics continued (FMM, hyman).
    Cubic,
    /// `x` wrapped into `[x_0, x_{n-1})` (periodic).
    Periodic,
}

/// A fitted interpolating spline: sorted distinct knots, their y values and the shape.
struct FittedSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    shape: SplineShape,
    extrapolation: Extrapolation,
}

/// Prepare the knots (pushing R's warnings) and fit a spline of the given method.
fn fit_spline(
    x: &[f64],
    y: &[f64],
    method: SplineMethod,
    warnings: &mut Vec<String>,
) -> EconResult<FittedSpline> {
    let (xs, mut ys, tied) = prepare_spline_knots(x, y)?;
    if tied {
        warnings.push(TIES_WARNING.to_string());
    }
    let (shape, extrapolation) = match method {
        SplineMethod::Natural => (
            SplineShape::SecondDerivs(compute_natural_spline(&xs, &ys)?),
            Extrapolation::Linear,
        ),
        SplineMethod::Fmm => (
            SplineShape::SecondDerivs(compute_fmm_spline(&xs, &ys)?),
            Extrapolation::Cubic,
        ),
        SplineMethod::Periodic => {
            let last = ys.len() - 1;
            if ys[0] != ys[last] {
                ys[last] = ys[0];
                warnings.push(PERIODIC_WARNING.to_string());
            }
            (
                SplineShape::SecondDerivs(compute_periodic_spline(&xs, &ys)?),
                Extrapolation::Periodic,
            )
        }
        SplineMethod::Hyman => (
            SplineShape::Slopes(compute_hyman_slopes(&xs, &ys)?),
            Extrapolation::Cubic,
        ),
        SplineMethod::MonotoneFC => (
            SplineShape::Slopes(compute_monotone_fc_slopes(&xs, &ys)),
            Extrapolation::Linear,
        ),
    };
    Ok(FittedSpline {
        x: xs,
        y: ys,
        shape,
        extrapolation,
    })
}

impl FittedSpline {
    /// The slopes at the first and last knot.
    fn end_slopes(&self) -> (f64, f64) {
        let (x, y) = (&self.x, &self.y);
        let n = x.len();
        match &self.shape {
            SplineShape::Slopes(b) => (b[0], b[n - 1]),
            SplineShape::SecondDerivs(m) => {
                let h0 = x[1] - x[0];
                let hl = x[n - 1] - x[n - 2];
                (
                    (y[1] - y[0]) / h0 - h0 * (2.0 * m[0] + m[1]) / 6.0,
                    (y[n - 1] - y[n - 2]) / hl + hl * (m[n - 2] + 2.0 * m[n - 1]) / 6.0,
                )
            }
        }
    }

    /// The spline's value at `xi` (NaN at a NaN point).
    fn eval(&self, xi: f64) -> f64 {
        let (x, y) = (&self.x, &self.y);
        let n = x.len();
        let mut xi = xi;
        match self.extrapolation {
            Extrapolation::Periodic => {
                let period = x[n - 1] - x[0];
                let mut w = (xi - x[0]) % period;
                if w < 0.0 {
                    w += period;
                }
                xi = w + x[0];
            }
            Extrapolation::Linear if xi < x[0] || xi > x[n - 1] => {
                let (left, right) = self.end_slopes();
                return if xi < x[0] {
                    y[0] + left * (xi - x[0])
                } else {
                    y[n - 1] + right * (xi - x[n - 1])
                };
            }
            _ => {}
        }
        let i = find_interval(x, xi);
        let h = x[i + 1] - x[i];
        match &self.shape {
            SplineShape::SecondDerivs(m) => {
                // S(x) = a y_i + b y_{i+1} + ((a^3 - a) M_i + (b^3 - b) M_{i+1}) h^2 / 6
                let a = (x[i + 1] - xi) / h;
                let b = (xi - x[i]) / h;
                a * y[i]
                    + b * y[i + 1]
                    + ((a.powi(3) - a) * m[i] + (b.powi(3) - b) * m[i + 1]) * h * h / 6.0
            }
            SplineShape::Slopes(s) => {
                let t = (xi - x[i]) / h;
                let t2 = t * t;
                let t3 = t2 * t;
                (2.0 * t3 - 3.0 * t2 + 1.0) * y[i]
                    + (t3 - 2.0 * t2 + t) * h * s[i]
                    + (3.0 * t2 - 2.0 * t3) * y[i + 1]
                    + (t3 - t2) * h * s[i + 1]
            }
        }
    }
}

/// The interval widths `h_i = x_{i+1} - x_i`.
fn widths(x: &[f64]) -> Vec<f64> {
    x.windows(2).map(|w| w[1] - w[0]).collect()
}

/// The right-hand side `6 [(y_{i+1} - y_i)/h_i - (y_i - y_{i-1})/h_{i-1}]` of the interior rows.
fn interior_rhs(y: &[f64], h: &[f64], i: usize) -> f64 {
    6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
}

/// Compute natural cubic spline coefficients.
fn compute_natural_spline(x: &[f64], y: &[f64]) -> EconResult<Vec<f64>> {
    let n = x.len();

    // Natural spline: M_0 = M_{n-1} = 0 (zero second derivatives at endpoints)
    // We solve a tridiagonal system for the second derivatives M_i
    let h = widths(x);

    // Build the right-hand side
    let mut d = vec![0.0; n];
    for i in 1..n - 1 {
        d[i] = interior_rhs(y, &h, i);
    }

    // Build the tridiagonal matrix (sub-diagonal, diagonal, super-diagonal)
    let mut diag = vec![1.0; n]; // Boundary conditions
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
    }

    // Solve tridiagonal system using Thomas algorithm
    solve_tridiagonal(&sub, &diag, &sup, &d)
}

/// The third divided difference of four points.
fn third_divided_difference(x: &[f64], y: &[f64]) -> f64 {
    let mut f = [y[0], y[1], y[2], y[3]];
    for k in 1..4 {
        for i in 0..4 - k {
            f[i] = (f[i + 1] - f[i]) / (x[i + k] - x[i]);
        }
    }
    f[0]
}

/// Compute FMM cubic spline coefficients (Forsythe, Malcolm and Moler's `spline`, R's
/// `fmm_spline`): on each end interval the spline's third derivative, `(M_1 - M_0) / h_0`, equals
/// that of the cubic through the four end points, `6` times their third divided difference.
fn compute_fmm_spline(x: &[f64], y: &[f64]) -> EconResult<Vec<f64>> {
    let n = x.len();
    let h = widths(x);

    let mut d = vec![0.0; n];
    let mut diag = vec![0.0; n];
    let mut sub = vec![0.0; n];
    let mut sup = vec![0.0; n];

    // -h_0 M_0 + h_0 M_1 = 6 h_0^2 D_L;  h M_{n-2} - h M_{n-1} = -6 h^2 D_R  (h = h_{n-2})
    diag[0] = -h[0];
    sup[0] = h[0];
    d[0] = 6.0 * h[0] * h[0] * third_divided_difference(&x[..4], &y[..4]);
    let hl = h[n - 2];
    sub[n - 1] = hl;
    diag[n - 1] = -hl;
    d[n - 1] = -6.0 * hl * hl * third_divided_difference(&x[n - 4..], &y[n - 4..]);

    for i in 1..n - 1 {
        sub[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        sup[i] = h[i];
        d[i] = interior_rhs(y, &h, i);
    }

    solve_tridiagonal(&sub, &diag, &sup, &d)
}

/// Compute periodic cubic spline coefficients (`y[0] == y[n-1]`, made so by the caller).
fn compute_periodic_spline(x: &[f64], y: &[f64]) -> EconResult<Vec<f64>> {
    let n = x.len();
    let h = widths(x);

    // n - 1 unknowns M_0..M_{n-2} (M_{n-1} = M_0); the row of knot i wraps with
    // h_{-1} = h_{n-2} and y_{-1} = y_{n-2}.
    let m = n - 1;
    let mut d = vec![0.0; m];
    let mut diag = vec![0.0; m];
    let mut sub = vec![0.0; m];
    let mut sup = vec![0.0; m];
    for i in 0..m {
        let (h_prev, y_prev) = if i == 0 {
            (h[n - 2], y[n - 2])
        } else {
            (h[i - 1], y[i - 1])
        };
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y_prev) / h_prev);
        sub[i] = h_prev;
        diag[i] = 2.0 * (h_prev + h[i]);
        sup[i] = h[i];
    }

    let mut second_derivs = solve_cyclic_tridiagonal(&sub, &diag, &sup, &d)?;
    second_derivs.push(second_derivs[0]); // M_{n-1} = M_0 for periodic
    Ok(second_derivs)
}

/// The secant slopes `(y_{i+1} - y_i) / (x_{i+1} - x_i)`.
fn secants(x: &[f64], y: &[f64]) -> Vec<f64> {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xw, yw)| (yw[1] - yw[0]) / (xw[1] - xw[0]))
        .collect()
}

/// Hyman's filter of the FMM slopes (R's `spline(method = "hyman")`, `hyman_filter`). The data
/// must be monotone.
fn compute_hyman_slopes(x: &[f64], y: &[f64]) -> EconResult<Vec<f64>> {
    let n = x.len();
    let s = secants(x, y);
    if !(y.windows(2).all(|w| w[1] >= w[0]) || y.windows(2).all(|w| w[1] <= w[0])) {
        return Err(EconError::InvalidSpecification {
            message: "'y' must be increasing or decreasing for the hyman spline".to_string(),
        });
    }
    let m = compute_fmm_spline(x, y)?;
    let h = widths(x);
    // FMM slopes at the knots.
    let mut b: Vec<f64> = (0..n - 1)
        .map(|i| s[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0)
        .collect();
    b.push(s[n - 2] + h[n - 2] * (m[n - 2] + 2.0 * m[n - 1]) / 6.0);
    for i in 0..n {
        let s0 = if i == 0 { s[0] } else { s[i - 1] };
        let s1 = if i == n - 1 { s[n - 2] } else { s[i] };
        let t1 = s0.abs().min(s1.abs());
        let sigma = if s0 * s1 > 0.0 { s1 } else { b[i] };
        b[i] = if sigma >= 0.0 {
            b[i].max(0.0).min(3.0 * t1)
        } else {
            b[i].min(0.0).max(-3.0 * t1)
        };
    }
    Ok(b)
}

/// Fritsch-Carlson slopes (R's `splinefun(method = "monoH.FC")`): the mean of the adjacent
/// secants (the end secants at the ends), zero on a flat interval, and scaled back on an interval
/// where they would break monotonicity (R's `monoFC_mod`).
fn compute_monotone_fc_slopes(x: &[f64], y: &[f64]) -> Vec<f64> {
    let n = x.len();
    let s = secants(x, y);
    let mut m = Vec::with_capacity(n);
    m.push(s[0]);
    m.extend(s.windows(2).map(|w| (w[0] + w[1]) / 2.0));
    m.push(s[n - 2]);
    for k in 0..n - 1 {
        let sk = s[k];
        if sk == 0.0 {
            m[k] = 0.0;
            m[k + 1] = 0.0;
        } else {
            let alpha = m[k] / sk;
            let beta = m[k + 1] / sk;
            let a2b3 = 2.0 * alpha + beta - 3.0;
            let ab23 = alpha + 2.0 * beta - 3.0;
            if a2b3 > 0.0 && ab23 > 0.0 && alpha * (a2b3 + ab23) < a2b3 * a2b3 {
                let tau_s = 3.0 * sk / (alpha * alpha + beta * beta).sqrt();
                m[k] = tau_s * alpha;
                m[k + 1] = tau_s * beta;
            }
        }
    }
    m
}

/// Solve a tridiagonal system using Thomas algorithm.
fn solve_tridiagonal(sub: &[f64], diag: &[f64], sup: &[f64], rhs: &[f64]) -> EconResult<Vec<f64>> {
    let n = diag.len();

    // Forward elimination
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = sup[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let denom = diag[i] - sub[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            return Err(EconError::Computation(
                "Singular matrix in tridiagonal solve".to_string(),
            ));
        }
        c_prime[i] = sup[i] / denom;
        d_prime[i] = (rhs[i] - sub[i] * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Solve a cyclic tridiagonal system (simplified approach).
fn solve_cyclic_tridiagonal(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> EconResult<Vec<f64>> {
    let n = diag.len();
    if n < 3 {
        return Err(EconError::InsufficientData {
            required: 3,
            provided: n,
            context: "cyclic tridiagonal system".to_string(),
        });
    }

    // Use Sherman-Morrison formula to handle the cyclic part
    // A = T + u*v' where T is regular tridiagonal

    let gamma = -diag[0];
    let mut diag_mod = diag.to_vec();
    diag_mod[0] -= gamma;
    diag_mod[n - 1] -= sup[n - 1] * sub[0] / gamma;

    // Solve T*y = d
    let y = solve_tridiagonal(sub, &diag_mod, sup, rhs)?;

    // Solve T*q = u where u = [gamma, 0, ..., 0, sub[0]]
    let mut u = vec![0.0; n];
    u[0] = gamma;
    u[n - 1] = sub[0];
    let q = solve_tridiagonal(sub, &diag_mod, sup, &u)?;

    // Sherman-Morrison: x = y - (v'*y)/(1 + v'*q) * q
    // where v = [1, 0, ..., 0, sup[n-1]/gamma]
    let vty = y[0] + sup[n - 1] / gamma * y[n - 1];
    let vtq = q[0] + sup[n - 1] / gamma * q[n - 1];

    let factor = vty / (1.0 + vtq);
    let x: Vec<f64> = y
        .iter()
        .zip(q.iter())
        .map(|(&yi, &qi)| yi - factor * qi)
        .collect();

    Ok(x)
}

/// The knot interval for `xi`: the first `j` with `x[j] <= xi <= x[j + 1]`, clamped to
/// `[0, n - 2]`, found by binary search over the sorted knots `x` (`x.len() >= 2`). A NaN
/// point gives 0. Equal to a linear scan from the first knot for every input.
fn find_interval(x: &[f64], xi: f64) -> usize {
    x.partition_point(|&k| k < xi)
        .saturating_sub(1)
        .min(x.len() - 2)
}

/// Linear or constant interpolation at a single point (R's `approx1`).
fn interpolate_approx(
    x: &[f64],
    y: &[f64],
    xi: f64,
    method: ApproxMethod,
    rule: ApproxRule,
    f: f64,
) -> f64 {
    let n = x.len();

    // Handle out-of-range
    if xi < x[0] {
        return match rule {
            ApproxRule::Na => f64::NAN,
            ApproxRule::Nearest => y[0],
        };
    }
    if xi > x[n - 1] {
        return match rule {
            ApproxRule::Na => f64::NAN,
            ApproxRule::Nearest => y[n - 1],
        };
    }

    // Find the interval containing xi; a knot returns its own value.
    let i = find_interval(x, xi);
    if xi == x[i + 1] {
        return y[i + 1];
    }
    if xi == x[i] {
        return y[i];
    }

    match method {
        ApproxMethod::Linear => {
            let t = (xi - x[i]) / (x[i + 1] - x[i]);
            y[i] + t * (y[i + 1] - y[i])
        }
        ApproxMethod::Constant => {
            // (1 - f) y_left + f y_right; a zero weight drops its term, so f = 0 and f = 1 never
            // mix in the other neighbour.
            let f1 = 1.0 - f;
            (if f1 != 0.0 { y[i] * f1 } else { 0.0 }) + (if f != 0.0 { y[i + 1] * f } else { 0.0 })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    // R 4.3.3:
    //   sx <- c(0.0, 1.3, 0.6, 2.2, 3.1, 4.0, 2.7, 5.2, 4.6, 6.0)
    //   sy <- c(1.00, 0.35, 0.92, -0.41, -0.88, -0.62, -0.79, 0.37, -0.12, 0.81)
    //   xo <- c(0.15, 0.95, 1.75, 2.45, 2.9, 3.55, 4.3, 4.9, 5.6, 5.95)
    //   spline(sx, sy, method = "natural", xout = xo)$y
    //   approx(sx, sy, xout = c(-0.5, xo, 6.5), method = "linear", rule = 1 / rule = 2)$y
    const SX: [f64; 10] = [0.0, 1.3, 0.6, 2.2, 3.1, 4.0, 2.7, 5.2, 4.6, 6.0];
    const SY: [f64; 10] = [
        1.00, 0.35, 0.92, -0.41, -0.88, -0.62, -0.79, 0.37, -0.12, 0.81,
    ];
    const XO: [f64; 10] = [0.15, 0.95, 1.75, 2.45, 2.9, 3.55, 4.3, 4.9, 5.6, 5.95];

    #[test]
    fn test_validate_natural_spline_against_r() {
        let r_y = [
            1.0035342841021795,
            0.6746362317515617,
            -0.03664486690060831,
            -0.6229938475881653,
            -0.8581626978141365,
            -0.8276939305271469,
            -0.38793707175681735,
            0.14034763562371816,
            0.6116496312127426,
            0.7860941770568031,
        ];
        let s = spline(&SX, &SY, Some(&XO), None, SplineMethod::Natural).unwrap();
        for (got, want) in s.y.iter().zip(r_y) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn test_validate_approx_linear_against_r() {
        let inner = [
            0.98,
            0.635,
            -0.029999999999999916,
            -0.6,
            -0.835,
            -0.75,
            -0.37,
            0.1250000000000004,
            0.5899999999999997,
            0.7825000000000002,
        ];
        let mut xo = vec![-0.5];
        xo.extend(XO);
        xo.push(6.5);
        let na = approx(
            &SX,
            &SY,
            Some(&xo),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.0,
        )
        .unwrap();
        let nearest = approx(
            &SX,
            &SY,
            Some(&xo),
            None,
            ApproxMethod::Linear,
            ApproxRule::Nearest,
            0.0,
        )
        .unwrap();
        assert!(na.y[0].is_nan() && na.y[11].is_nan());
        assert_eq!((nearest.y[0], nearest.y[11]), (1.0, 0.81));
        for i in 0..10 {
            assert!((na.y[i + 1] - inner[i]).abs() < 1e-12);
            assert!((nearest.y[i + 1] - inner[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_find_interval_matches_linear_scan() {
        fn naive(x: &[f64], xi: f64) -> usize {
            // today's scan, with the spline evaluator's clamps
            let n = x.len();
            let mut i = 0;
            for j in 0..n - 1 {
                if xi >= x[j] && xi <= x[j + 1] {
                    i = j;
                    break;
                }
            }
            if xi < x[0] {
                0
            } else if xi > x[n - 1] {
                n - 2
            } else {
                i
            }
        }
        for knots in [
            vec![0.0, 1.0, 2.0, 3.5, 7.0],
            vec![0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 5.0],
            vec![-1.0, 4.0],
        ] {
            let mut pts: Vec<f64> = knots.clone();
            pts.extend(knots.windows(2).map(|w| 0.5 * (w[0] + w[1])));
            pts.extend([-1e9, 1e9, f64::NEG_INFINITY, f64::INFINITY, f64::NAN]);
            for p in pts {
                assert_eq!(
                    find_interval(&knots, p),
                    naive(&knots, p),
                    "knots {knots:?}, p {p}"
                );
            }
        }
    }

    #[test]
    fn test_splinefun_validates_knots_like_spline() {
        assert!(
            splinefun(
                &[0.0, 1.0, 2.0, f64::NAN],
                &[1.0, 2.0, 3.0, 4.0],
                SplineMethod::Natural
            )
            .is_err()
        );
        // Ties are averaged, so four points with one tie leave three distinct knots: too few.
        let e = splinefun(
            &[0.0, 1.0, 1.0, 2.0],
            &[1.0, 2.0, 3.0, 4.0],
            SplineMethod::Natural,
        )
        .err()
        .unwrap();
        assert!(
            matches!(e, EconError::InsufficientData { provided: 3, .. }),
            "{e}"
        );
    }

    /// Tied x are collapsed to their mean y with R's warning, in spline and approx (they were refused).
    #[test]
    fn test_ties_average_and_warn() {
        let x = [0.0, 1.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 4.0, 0.5, 2.0];
        let tied = spline(&x, &y, Some(&[1.0]), None, SplineMethod::Natural).unwrap();
        assert_eq!(tied.y[0], 3.0);
        assert_eq!(tied.warnings, vec![TIES_WARNING.to_string()]);
        let plain = spline(&x[1..], &y[1..], Some(&[1.0]), None, SplineMethod::Natural);
        assert!(plain.is_err()); // three distinct knots
        let untied = spline(
            &[0.0, 1.0, 2.0, 3.0],
            &[1.0, 3.0, 0.5, 2.0],
            Some(&[0.4, 2.5]),
            None,
            SplineMethod::Natural,
        )
        .unwrap();
        assert!(untied.warnings.is_empty());
        let again = spline(&x, &y, Some(&[0.4, 2.5]), None, SplineMethod::Natural).unwrap();
        assert_eq!(again.y, untied.y);
        let a = approx(
            &x,
            &y,
            Some(&[1.0]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.0,
        )
        .unwrap();
        assert_eq!(a.y, vec![3.0]);
        assert_eq!(a.warnings, vec![TIES_WARNING.to_string()]);
        // A single distinct x is refused instead of indexing past the knots.
        assert!(
            approx(
                &[2.0, 2.0],
                &[1.0, 3.0],
                Some(&[2.0]),
                None,
                ApproxMethod::Linear,
                ApproxRule::Na,
                0.0
            )
            .is_err()
        );
    }

    /// The (x, y) column pair `<s>x`, `<s>y` of `spline_t4.csv` (NA padding dropped).
    fn t4_pair(ds: &crate::data::Dataset, s: &str) -> (Vec<f64>, Vec<f64>) {
        use crate::data::columns::{NaPolicy, numeric_column};
        let x = numeric_column(ds.df(), &format!("{s}x"), NaPolicy::Omit)
            .unwrap()
            .0;
        let y = numeric_column(ds.df(), &format!("{s}y"), NaPolicy::Omit)
            .unwrap()
            .0;
        assert_eq!(x.len(), y.len());
        (x, y)
    }

    fn spline_at(x: &[f64], y: &[f64], xo: &[f64], m: SplineMethod) -> Vec<f64> {
        spline(x, y, Some(xo), None, m).unwrap().y
    }

    /// spline(method = "fmm"): inside and outside the knots, and on n = 11 equally spaced points.
    #[test]
    fn test_validate_spline_fmm() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let (sx, sy) = t4_pair(&ds, "s");
        let (fx, fy) = t4_pair(&ds, "f");
        let xo = case.vector("xo");
        let n11 = spline(&fx, &fy, None, Some(11), SplineMethod::Fmm).unwrap();
        Check::new(&case)
            .vector(
                "fmm_s",
                &spline_at(&sx, &sy, &xo, SplineMethod::Fmm),
                Tol::coef(10),
            )
            .vector("fmm_f_n11", &n11.y, Tol::coef(10))
            .vector(
                "fmm_f_out",
                &spline_at(&fx, &fy, &[-1.0, 6.0], SplineMethod::Fmm),
                Tol::coef(10),
            )
            .finish();
    }

    /// spline(method = "natural") continues linearly outside the knots.
    #[test]
    fn test_validate_spline_natural_outside() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let (sx, sy) = t4_pair(&ds, "s");
        let (fx, fy) = t4_pair(&ds, "f");
        let xo = case.vector("xo");
        Check::new(&case)
            .vector(
                "natural_s",
                &spline_at(&sx, &sy, &xo, SplineMethod::Natural),
                Tol::coef(10),
            )
            .vector(
                "natural_f_out",
                &spline_at(&fx, &fy, &[-1.0, 6.0], SplineMethod::Natural),
                Tol::coef(10),
            )
            .finish();
    }

    /// spline(method = "periodic"), wrapping points outside the knots' range.
    #[test]
    fn test_validate_spline_periodic() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let (px, py) = t4_pair(&ds, "p");
        let (ux, uy) = t4_pair(&ds, "u");
        let p = spline(
            &px,
            &py,
            Some(&case.vector("pxo")),
            None,
            SplineMethod::Periodic,
        )
        .unwrap();
        let u = spline(
            &ux,
            &uy,
            Some(&case.vector("uxo")),
            None,
            SplineMethod::Periodic,
        )
        .unwrap();
        assert!(p.warnings.is_empty() && u.warnings.is_empty());
        Check::new(&case)
            .vector("periodic_p", &p.y, Tol::coef(12))
            .vector("periodic_u", &u.y, Tol::coef(12))
            .finish();
    }

    /// R: spline(x, y, method = "periodic") with y[1] != y[n] warns and uses y[1] for both ends.
    #[test]
    fn test_periodic_first_and_last_y_differ() {
        let x = [0.0, 1.0, 2.5, 3.0, 4.0];
        let y = [1.0, 0.2, -0.7, 0.4, 1.0];
        let mut y2 = y;
        y2[4] = 1.5;
        let xo = [0.3, 1.7, 3.6, 4.0, 5.2];
        let same = spline(&x, &y, Some(&xo), None, SplineMethod::Periodic).unwrap();
        let diff = spline(&x, &y2, Some(&xo), None, SplineMethod::Periodic).unwrap();
        assert_eq!(same.y, diff.y);
        assert_eq!(diff.warnings, vec![PERIODIC_WARNING.to_string()]);
    }

    /// spline(method = "hyman") on increasing and decreasing data; non-monotone data is refused.
    #[test]
    fn test_validate_spline_hyman() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let (hx, hy) = t4_pair(&ds, "h");
        let (mx, my) = t4_pair(&ds, "m");
        let (sx, sy) = t4_pair(&ds, "s");
        let refused = spline(&sx, &sy, None, None, SplineMethod::Hyman);
        assert!(
            matches!(&refused, Err(EconError::InvalidSpecification { message }) if message.contains("increasing or decreasing")),
            "{refused:?}"
        );
        Check::new(&case)
            .vector(
                "hyman_h",
                &spline_at(&hx, &hy, &case.vector("hxo"), SplineMethod::Hyman),
                Tol::coef(12),
            )
            .vector(
                "hyman_m",
                &spline_at(&mx, &my, &case.vector("mxo"), SplineMethod::Hyman),
                Tol::coef(12),
            )
            .scalar(
                "hyman_nonmonotone_error",
                f64::from(u8::from(refused.is_err())),
                Tol::exact(),
            )
            .finish();
    }

    /// splinefun(method = "monoH.FC"), linear outside the knots, on monotone and non-monotone data.
    #[test]
    fn test_validate_splinefun_monoh_fc() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let at = |s: &str, xo: &str| {
            let (x, y) = t4_pair(&ds, s);
            let f = splinefun(&x, &y, SplineMethod::MonotoneFC).unwrap();
            case.vector(xo).iter().map(|&v| f(v)).collect::<Vec<_>>()
        };
        Check::new(&case)
            .vector("monofc_h", &at("h", "hxo"), Tol::coef(12))
            .vector("monofc_m", &at("m", "mxo"), Tol::coef(12))
            .vector("monofc_s", &at("s", "xo"), Tol::coef(12))
            .finish();
    }

    /// Tied x: natural and fmm splines on the averaged knots, and R's default n of 3 per distinct x.
    #[test]
    fn test_validate_spline_ties() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let (tx, ty) = t4_pair(&ds, "t");
        let txo = case.vector("txo");
        let nat = spline(&tx, &ty, Some(&txo), None, SplineMethod::Natural).unwrap();
        assert_eq!(nat.warnings, vec![TIES_WARNING.to_string()]);
        let default_n = spline(&tx, &ty, None, None, SplineMethod::Fmm).unwrap();
        Check::new(&case)
            .vector("ties_natural", &nat.y, Tol::coef(10))
            .vector(
                "ties_fmm",
                &spline_at(&tx, &ty, &txo, SplineMethod::Fmm),
                Tol::coef(10),
            )
            .scalar("ties_default_n", default_n.n as f64, Tol::exact())
            .finish();
    }

    /// approx(method = "constant") returns a knot's own value at the knot, whatever f.
    #[test]
    fn test_validate_approx_constant_at_knots() {
        use p2a_validation::{Check, Tol};
        let (case, _) = crate::test_support::load_case("stat_spline_t4");
        let at = |f: f64| {
            approx(
                &[1.0, 2.0, 3.0],
                &[10.0, 20.0, 30.0],
                Some(&[1.0, 2.0, 2.5, 3.0]),
                None,
                ApproxMethod::Constant,
                ApproxRule::Na,
                f,
            )
            .unwrap()
            .y
        };
        Check::new(&case)
            .vector("approx_const_f0", &at(0.0), Tol::coef(10))
            .vector("approx_const_f05", &at(0.5), Tol::coef(10))
            .vector("approx_const_f1", &at(1.0), Tol::coef(10))
            .finish();
    }

    /// approx with tied x (averaged), linear and constant, and rule = 2 on the tied set.
    #[test]
    fn test_validate_approx_ties() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_spline_t4");
        let (tx, ty) = t4_pair(&ds, "t");
        let ap = |m: ApproxMethod| {
            approx(
                &[1.0, 2.0, 2.0, 3.0],
                &[10.0, 20.0, 40.0, 30.0],
                Some(&[2.0, 1.5, 2.5]),
                None,
                m,
                ApproxRule::Na,
                0.0,
            )
            .unwrap()
            .y
        };
        let rule2 = approx(
            &tx,
            &ty,
            Some(&case.vector("txo")),
            None,
            ApproxMethod::Linear,
            ApproxRule::Nearest,
            0.0,
        )
        .unwrap();
        Check::new(&case)
            .vector(
                "approx_ties_linear",
                &ap(ApproxMethod::Linear),
                Tol::coef(10),
            )
            .vector(
                "approx_ties_const",
                &ap(ApproxMethod::Constant),
                Tol::coef(10),
            )
            .vector("approx_t_rule2", &rule2.y, Tol::coef(10))
            .finish();
    }

    #[test]
    fn test_approxfun_needs_two_finite_points() {
        // pre-fix: the closure indexes x[1] and panics for points equal to the single knot
        assert!(
            approxfun(
                &[1.0, f64::NAN],
                &[1.0, 2.0],
                ApproxMethod::Linear,
                ApproxRule::Na,
                0.0
            )
            .is_err()
        );
    }

    #[test]
    fn test_spline_and_approx_honour_cancellation() {
        let t = CancelToken::new();
        t.cancel();
        resource::with_budget(Budget::new(None, t), || {
            assert!(matches!(
                spline(&SX, &SY, Some(&XO), None, SplineMethod::Natural),
                Err(EconError::Cancelled(_))
            ));
            assert!(matches!(
                approx(
                    &SX,
                    &SY,
                    Some(&XO),
                    None,
                    ApproxMethod::Linear,
                    ApproxRule::Na,
                    0.0
                ),
                Err(EconError::Cancelled(_))
            ));
        });
    }

    #[test]
    fn test_spline_natural_basic() {
        // Simple quadratic function: y = x^2
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = spline(
            &x,
            &y,
            Some(&[0.5, 1.5, 2.5, 3.5]),
            None,
            SplineMethod::Natural,
        )
        .unwrap();

        assert_eq!(result.n, 4);
        // Check interpolated values are close to true values
        for (i, &xi) in result.x.iter().enumerate() {
            let expected = xi * xi;
            let diff = (result.y[i] - expected).abs();
            assert!(
                diff < 0.5,
                "At x={}, expected {}, got {}",
                xi,
                expected,
                result.y[i]
            );
        }
    }

    #[test]
    fn test_spline_passes_through_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];

        let result = spline(&x, &y, Some(&x), None, SplineMethod::Natural).unwrap();

        // Spline should pass through all original points
        for (i, &yi) in y.iter().enumerate() {
            let diff = (result.y[i] - yi).abs();
            assert!(
                diff < 1e-10,
                "At x[{}], expected {}, got {}",
                i,
                yi,
                result.y[i]
            );
        }
    }

    #[test]
    fn test_spline_too_few_points() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let result = spline(&x, &y, None, None, SplineMethod::Natural);
        assert!(result.is_err());
    }

    #[test]
    fn test_spline_monotone() {
        // Monotone increasing data
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![0.0, 1.0, 3.0, 6.0, 10.0];

        let result = spline(&x, &y, None, Some(50), SplineMethod::MonotoneFC).unwrap();

        // Check monotonicity
        for i in 1..result.y.len() {
            assert!(
                result.y[i] >= result.y[i - 1] - 1e-10,
                "Monotone spline not monotone at index {}",
                i
            );
        }
    }

    #[test]
    fn test_approx_linear_basic() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];

        let result = approx(
            &x,
            &y,
            Some(&[0.5, 1.5, 2.5]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.5,
        )
        .unwrap();

        assert_eq!(result.n, 3);
        assert!((result.y[0] - 1.0).abs() < 1e-10); // Linear interpolation at 0.5
        assert!((result.y[1] - 3.0).abs() < 1e-10); // Linear interpolation at 1.5
        assert!((result.y[2] - 5.0).abs() < 1e-10); // Linear interpolation at 2.5
    }

    #[test]
    fn test_approx_constant() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];

        // Left-continuous (f=0): returns left value
        let result = approx(
            &x,
            &y,
            Some(&[0.5, 1.5]),
            None,
            ApproxMethod::Constant,
            ApproxRule::Na,
            0.0,
        )
        .unwrap();
        assert!((result.y[0] - 0.0).abs() < 1e-10);
        assert!((result.y[1] - 2.0).abs() < 1e-10);

        // Right-continuous (f=1): returns right value
        let result = approx(
            &x,
            &y,
            Some(&[0.5, 1.5]),
            None,
            ApproxMethod::Constant,
            ApproxRule::Na,
            1.0,
        )
        .unwrap();
        assert!((result.y[0] - 2.0).abs() < 1e-10);
        assert!((result.y[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_approx_out_of_range() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 2.0];

        // NA rule: out of range returns NaN
        let result = approx(
            &x,
            &y,
            Some(&[-1.0, 3.0]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.5,
        )
        .unwrap();
        assert!(result.y[0].is_nan());
        assert!(result.y[1].is_nan());

        // Nearest rule: returns endpoint value
        let result = approx(
            &x,
            &y,
            Some(&[-1.0, 3.0]),
            None,
            ApproxMethod::Linear,
            ApproxRule::Nearest,
            0.5,
        )
        .unwrap();
        assert!((result.y[0] - 0.0).abs() < 1e-10);
        assert!((result.y[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_splinefun() {
        let x: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        let f = splinefun(&x, &y, SplineMethod::Natural).unwrap();

        // Check that function passes through original points
        for i in 0..x.len() {
            let diff = (f(x[i]) - y[i]).abs();
            assert!(diff < 1e-10, "Function doesn't pass through point {}", i);
        }
    }

    #[test]
    fn test_approxfun() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];

        let f = approxfun(&x, &y, ApproxMethod::Linear, ApproxRule::Nearest, 0.5).unwrap();

        assert!((f(0.5) - 1.0).abs() < 1e-10);
        assert!((f(1.5) - 3.0).abs() < 1e-10);
        assert!((f(-1.0) - 0.0).abs() < 1e-10); // Extrapolation
        assert!((f(5.0) - 6.0).abs() < 1e-10); // Extrapolation
    }

    /// The natural spline passes through its knots and returns the points asked for.
    #[test]
    fn test_spline_passes_through_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![0.0, 0.8, 0.9, 0.1, -0.8, -1.0];

        let result = spline(&x, &y, None, Some(11), SplineMethod::Natural).unwrap();

        // Check that spline passes through original points
        for i in 0..x.len() {
            let f = splinefun(&x, &y, SplineMethod::Natural).unwrap();
            let diff = (f(x[i]) - y[i]).abs();
            assert!(
                diff < 1e-8,
                "Spline doesn't pass through point {}: got {}, expected {}",
                i,
                f(x[i]),
                y[i]
            );
        }

        // Check output has correct number of points
        assert_eq!(result.n, 11);
    }

    /// Validation test against R's approx() function
    #[test]
    fn test_validate_approx_against_r() {
        // R code:
        // x <- c(1, 2, 3, 4, 5)
        // y <- c(1, 4, 9, 16, 25)  # y = x^2
        // approx(x, y, xout=c(1.5, 2.5, 3.5, 4.5))$y
        // Expected: 2.5, 6.5, 12.5, 20.5

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let xout = vec![1.5, 2.5, 3.5, 4.5];

        let result = approx(
            &x,
            &y,
            Some(&xout),
            None,
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.5,
        )
        .unwrap();

        let expected = [2.5, 6.5, 12.5, 20.5];
        for i in 0..expected.len() {
            let diff = (result.y[i] - expected[i]).abs();
            assert!(
                diff < 1e-10,
                "At xout[{}], expected {}, got {}",
                i,
                expected[i],
                result.y[i]
            );
        }
    }

    /// TOOL-25: R's spline() evaluates n = 3 * length(x) points by default (p2a used 50); pinned to spline(method = "natural").
    #[test]
    fn test_validate_spline_t3_default_n() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_ts_acf_spline_t3");
        let kx = numeric_column(ds.df(), "kx", NaPolicy::Omit).unwrap().0;
        let ky = numeric_column(ds.df(), "ky", NaPolicy::Omit).unwrap().0;
        let s = spline(&kx, &ky, None, None, SplineMethod::Natural).unwrap();
        Check::new(&case)
            .scalar("spline_n", s.n as f64, Tol::exact())
            .vector("spline_x", &s.x, Tol::coef(30))
            .vector("spline_y", &s.y, Tol::coef(30))
            .finish();
    }

    /// n = 1 divided 0 by 0; R's seq.int(xmin, xmax, length.out = 1) is xmin.
    #[test]
    fn test_validate_spline_t3_n_one() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("stat_ts_acf_spline_t3");
        let kx = numeric_column(ds.df(), "kx", NaPolicy::Omit).unwrap().0;
        let ky = numeric_column(ds.df(), "ky", NaPolicy::Omit).unwrap().0;
        let s = spline(&kx, &ky, None, Some(1), SplineMethod::Natural).unwrap();
        Check::new(&case)
            .scalar("spline1_x", s.x[0], Tol::coef(10))
            .scalar("spline1_y", s.y[0], Tol::coef(10))
            .finish();
        assert_eq!(s.n, 1);
        assert!(
            spline(&kx, &ky, None, Some(0), SplineMethod::Natural)
                .unwrap()
                .x
                .is_empty()
        );
        let a = approx(
            &kx,
            &ky,
            None,
            Some(1),
            ApproxMethod::Linear,
            ApproxRule::Na,
            0.0,
        )
        .unwrap();
        assert_eq!(a.x, vec![kx.iter().copied().fold(f64::INFINITY, f64::min)]);
    }

    #[test]
    fn test_spline_method_parses_aliases_and_rejects_unknown() {
        assert_eq!(SplineMethod::choices().validate(), Ok(()));
        assert_eq!(ApproxMethod::choices().validate(), Ok(()));
        assert_eq!(ApproxRule::choices().validate(), Ok(()));
        assert_eq!("FMM".parse::<SplineMethod>().unwrap(), SplineMethod::Fmm);
        assert_eq!(
            "monoH.FC".parse::<SplineMethod>().unwrap(),
            SplineMethod::MonotoneFC
        );
        assert_eq!(
            "step".parse::<ApproxMethod>().unwrap(),
            ApproxMethod::Constant
        );
        assert_eq!("2".parse::<ApproxRule>().unwrap(), ApproxRule::Nearest);
        assert_eq!(
            "hyman".parse::<SplineMethod>().unwrap(),
            SplineMethod::Hyman
        );
        assert_eq!(
            "monotone".parse::<SplineMethod>().unwrap(),
            SplineMethod::MonotoneFC
        );
        assert!("akima".parse::<SplineMethod>().is_err());
    }
}
