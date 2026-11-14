//! Smoothing spline regression.
//!
//! A cubic smoothing spline as R's `smooth.spline()`: it minimises a penalized sum of squares
//! over cubic B-splines on x scaled to [0, 1]. If neither df, spar nor lambda is given, spar
//! minimises GCV (R's default) or, with `criterion = Cv` / `cv = true`, ordinary
//! leave-one-out CV; the result reports the criterion used.
//!
//! The algorithm is R's (smooth.spline's setup, `sbart.c`, `sslvrg.f`, `sgram.f`), written in
//! Rust: ties merged at `1e-6 IQR(x)`, R's knots, the integrated `B''` penalty with sgram.f's
//! 0.333 for 1/3, R's `ratio` and spar scale, a banded Cholesky solve with exact leverages, and
//! sbart's Brent search on spar, which reports the best spar with the lambda, df and fit of its
//! last evaluation, as R does. O(n) time and memory.
//!
//! # References
//!
//! - Green, P. J., & Silverman, B. W. (1994). "Nonparametric Regression and
//!   Generalized Linear Models: A Roughness Penalty Approach". Chapman & Hall.
//! - R Core Team. `stats::smooth.spline()` function.
//!   <https://stat.ethz.ch/R-manual/R-devel/library/stats/html/smooth.spline.html>

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::ResultNotes;
use serde::{Deserialize, Serialize};

/// A df target the spar search misses by more than this is reported in `warnings`.
pub const DF_TARGET_TOL: f64 = 1e-2;

/// The criterion that selects lambda when neither df, spar nor lambda is given.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplineCriterion {
    /// Generalized cross-validation (R's default, `cv = FALSE`).
    #[default]
    Gcv,
    /// Ordinary leave-one-out cross-validation (R's `cv = TRUE`).
    Cv,
}

crate::option_enum!(
    SplineCriterion,
    crate::options::Choices {
        option: "criterion",
        values: &[
            (
                &["gcv", "generalized_cv", "generalized_cross_validation"],
                SplineCriterion::Gcv
            ),
            (
                &["cv", "loocv", "loo", "ordinary_cv", "leave_one_out"],
                SplineCriterion::Cv
            ),
        ],
    }
);

/// Configuration for smooth.spline fitting.
#[derive(Debug, Clone)]
pub struct SmoothSplineConfig {
    /// Equivalent degrees of freedom for the spline (1 < df <= number of unique x).
    /// If specified, spar and lambda are ignored.
    pub df: Option<f64>,
    /// Smoothing parameter: `lambda = ratio * 256^(3 spar - 1)` (R's scale). If neither df,
    /// spar nor lambda is given, `criterion` selects spar.
    pub spar: Option<f64>,
    /// Penalty parameter lambda on R's scale (x scaled to [0, 1]), used when neither df nor
    /// spar is given.
    pub lambda: Option<f64>,
    /// Use all unique x values as knots (default: false, R's `.nknots.smspl` subset).
    pub all_knots: bool,
    /// R's meaning: `true` selects ordinary leave-one-out CV (R `cv = TRUE`) and overrides
    /// `criterion`; `false` leaves the choice to `criterion`. Prefer `criterion`.
    pub cv: bool,
    /// The criterion minimised when neither df, spar nor lambda is given (default GCV, as R).
    pub criterion: SplineCriterion,
    /// Weights for observations (optional, finite and non-negative, not all zero).
    pub weights: Option<Vec<f64>>,
}

impl Default for SmoothSplineConfig {
    fn default() -> Self {
        Self {
            df: None,
            spar: None,
            lambda: None,
            all_knots: false,
            cv: false,
            criterion: SplineCriterion::Gcv,
            weights: None,
        }
    }
}

impl SmoothSplineConfig {
    /// Create a new configuration with a specified degrees of freedom.
    pub fn with_df(df: f64) -> Self {
        Self {
            df: Some(df),
            ..Default::default()
        }
    }

    /// Create a new configuration with a specified smoothing parameter.
    pub fn with_spar(spar: f64) -> Self {
        Self {
            spar: Some(spar),
            ..Default::default()
        }
    }

    /// Select lambda by cross-validation: `true` = GCV, `false` = ordinary leave-one-out CV.
    ///
    /// Kept for source compatibility; its argument has the opposite sense of R's `cv`
    /// (and of the `cv` field).
    #[deprecated(note = "use with_criterion")]
    pub fn with_cv(gcv: bool) -> Self {
        Self::with_criterion(if gcv {
            SplineCriterion::Gcv
        } else {
            SplineCriterion::Cv
        })
    }

    /// Select lambda by `criterion`.
    pub fn with_criterion(criterion: SplineCriterion) -> Self {
        Self {
            criterion,
            ..Default::default()
        }
    }

    /// The criterion in force: ordinary CV when `cv` is true (R's meaning), else `criterion`.
    pub fn effective_criterion(&self) -> SplineCriterion {
        if self.cv {
            SplineCriterion::Cv
        } else {
            self.criterion
        }
    }
}

/// The fitted spline itself (R's `fit` component): cubic B-splines on `knots` over x scaled
/// by `(x - min) / range`, with coefficients `coef`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SmoothSplineFit {
    /// The knot sequence on [0, 1], the end knots repeated four times.
    pub knots: Vec<f64>,
    /// B-spline coefficients (`knots.len() - 4` of them).
    pub coef: Vec<f64>,
    /// The smallest x.
    pub min: f64,
    /// The range of x.
    pub range: f64,
}

/// Result of smooth.spline fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothSplineResult {
    /// Unique x values (sorted).
    pub x: Vec<f64>,
    /// Fitted y values at the unique x points.
    pub y: Vec<f64>,
    /// Equivalent degrees of freedom (the sum of the leverages).
    pub df: f64,
    /// Smoothing parameter spar (NaN for a given lambda of 0, which no spar gives).
    pub spar: f64,
    /// Penalty parameter lambda on R's scale (x scaled to [0, 1]).
    pub lambda: f64,
    /// R's `cv.crit` on the observations: ordinary CV when CV selected spar, else GCV.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cv_crit: Option<f64>,
    /// Residuals at original data points.
    pub residuals: Vec<f64>,
    /// Leverage values (diagonal of smoother matrix) at the unique x.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leverage: Option<Vec<f64>>,
    /// Number of original observations.
    pub n_obs: usize,
    /// Number of knots (R's `nknots`: distinct x up to 49, then `.nknots.smspl`).
    pub n_knots: usize,
    /// The criterion that selected lambda; `None` when df, spar or lambda was given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criterion: Option<SplineCriterion>,
    /// Notes on the fit (a df target the search did not reach); absent from JSON when
    /// empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
    /// R's `ratio = tr(X'WX) / tr(Omega)`, the scale of lambda.
    #[serde(default)]
    pub ratio: f64,
    /// The spline (knots and coefficients) [`smooth_spline_predict`] evaluates.
    #[serde(default)]
    pub fit: SmoothSplineFit,
}

impl ResultNotes for SmoothSplineResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// Fit a cubic smoothing spline to data, as R's `smooth.spline()`.
///
/// The spline minimises `sum w_i (y_i - f(x_i))^2 + lambda * integral f''(t)^2 dt` on x scaled
/// to [0, 1], over cubic B-splines on R's knots (`.nknots.smspl`, or every distinct x with
/// `all_knots`). Given `spar`, `lambda = ratio * 256^(3 spar - 1)` with
/// `ratio = tr(X'WX) / tr(Omega)` over the inner coefficients; given `lambda`, it is used as
/// is (R's scale). Otherwise sbart's Brent minimiser searches `spar` in [-1.5, 1.5]
/// (tolerance 1e-4, eps 2e-8, at most 500 steps) for the df target, or for GCV (R's default)
/// or ordinary leave-one-out CV (`criterion = Cv`, `cv = true`).
///
/// As R: x values within `1e-6 * IQR(x)` are one value (their y and weights are combined);
/// when IQR(x) is 0, where R stops ("'tol' must be strictly positive and finite"), only equal x
/// values are combined,
/// weights are rescaled to sum to the number of positive weights, the penalty integrates
/// the quadratic term of `B_i'' B_j''` with sgram.f's 0.333 for 1/3, and a search reports
/// the best spar with the lambda, df, fit and leverages of its last evaluation.
///
/// df must lie in (1, number of distinct x], spar must be finite, lambda finite and
/// non-negative, weights finite and non-negative. A df target that the search misses by more
/// than [`DF_TARGET_TOL`] is reported in `warnings`.
///
/// # Errors
///
/// `InvalidSpecification` for unequal lengths, non-finite data or an invalid control;
/// `InsufficientData` below four (distinct) x values; `InvalidSpecification` when the
/// penalised normal equations are not positive definite; `Cancelled` when the budget runs out.
///
/// # References
///
/// - R function `stats::smooth.spline()` (R 4.3: `sbart.c`, `sslvrg.f`, `sgram.f`)
/// - Green & Silverman (1994). "Nonparametric Regression and GLMs".
/// - Hutchinson, M. F. and de Hoog, F. R. (1985). Smoothing noisy data with spline
///   functions. Numerische Mathematik 47, 99-106.
pub fn smooth_spline(
    x: &[f64],
    y: &[f64],
    config: SmoothSplineConfig,
) -> EconResult<SmoothSplineResult> {
    if x.len() != y.len() {
        return Err(EconError::InvalidSpecification {
            message: format!("x and y must have same length: {} vs {}", x.len(), y.len()),
        });
    }

    let n = x.len();
    if n < 4 {
        return Err(EconError::InsufficientData {
            required: 4,
            provided: n,
            context: "smooth.spline requires at least 4 data points".to_string(),
        });
    }

    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite()) {
        return Err(EconError::InvalidSpecification {
            message: "Input contains non-finite values".to_string(),
        });
    }
    let spec = |message: String| Err(EconError::InvalidSpecification { message });
    if let Some(spar) = config.spar
        && !spar.is_finite()
    {
        return spec(format!("spar must be finite, got {spar}"));
    }
    if let Some(lambda) = config.lambda
        && !(lambda.is_finite() && lambda >= 0.0)
    {
        return spec(format!(
            "lambda must be finite and non-negative, got {lambda}"
        ));
    }
    if let Some(w) = config.weights.as_deref() {
        if w.len() != n {
            return spec(format!(
                "weights must have the same length as the data: {} vs {n}",
                w.len()
            ));
        }
        if let Some(i) = w.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
            return spec(format!(
                "weights must be finite and non-negative; weight {} is {}",
                i + 1,
                w[i]
            ));
        }
        if !w.iter().any(|v| *v > 0.0) {
            return spec("weights must not all be zero".to_string());
        }
    }

    let setup = Setup::new(x, y, config.weights.as_deref(), config.all_knots)?;
    let nx = setup.ux.len();
    if let Some(df) = config.df
        && !(df.is_finite() && df > 1.0 && df <= nx as f64)
    {
        return spec(format!(
            "df must be in (1, {nx}] (the number of unique x values), got {df}"
        ));
    }

    let mut warnings = Vec::new();
    let mut criterion = None;
    // precedence: df, then spar, then lambda, then the criterion
    let (spar, solved) = if let (None, Some(spar)) = (config.df, config.spar) {
        (spar, setup.solve(setup.lambda(spar))?)
    } else if let (None, Some(lambda)) = (config.df, config.lambda) {
        // the spar of the given lambda on R's scale (R reports NA); none for lambda = 0
        let spar = if lambda > 0.0 {
            ((lambda / setup.ratio).ln() / 256f64.ln() + 1.0) / 3.0
        } else {
            f64::NAN
        };
        (spar, setup.solve(lambda)?)
    } else {
        let crit = match config.df {
            Some(df) => Criterion::DfMatch(df),
            None => {
                let c = config.effective_criterion();
                criterion = Some(c);
                match c {
                    SplineCriterion::Gcv => Criterion::Gcv,
                    SplineCriterion::Cv => Criterion::Cv,
                }
            }
        };
        // sbart returns the fit of its last evaluation, not of the spar it reports
        let mut last = None;
        let spar = brent_spar(
            SPAR_LOW,
            SPAR_HIGH,
            SPAR_TOL,
            SPAR_EPS,
            SPAR_MAXIT,
            |spar| {
                let s = setup.solve(setup.lambda(spar))?;
                let v = setup.criterion(&s, crit);
                last = Some(s);
                Ok(v)
            },
        )?;
        let solved = match last {
            Some(s) => s,
            None => setup.solve(setup.lambda(spar))?,
        };
        if let Criterion::DfMatch(target) = crit
            && ((solved.df - target).abs() > DF_TARGET_TOL || solved.df.is_nan())
        {
            warnings.push(format!(
                "df target {target} reached {}: the spar search ended at its bound",
                solved.df
            ));
        }
        (spar, solved)
    };

    let cv = criterion == Some(SplineCriterion::Cv);
    let cv_crit = setup.cv_crit(y, &solved, cv);
    let residuals: Vec<f64> = y
        .iter()
        .zip(&setup.ox)
        .map(|(&yi, &g)| yi - solved.ty[g])
        .collect();

    Ok(SmoothSplineResult {
        df: solved.df,
        spar,
        lambda: solved.lambda,
        ratio: setup.ratio,
        cv_crit: Some(cv_crit),
        residuals,
        leverage: Some(solved.lev),
        n_obs: n,
        n_knots: setup.nknots,
        criterion,
        warnings,
        fit: SmoothSplineFit {
            knots: setup.knot,
            coef: solved.coef,
            min: setup.ux[0],
            range: setup.range,
        },
        y: solved.ty,
        x: setup.ux,
    })
}

/// Predict values at new x points from a fitted smoothing spline, as R's
/// `predict.smooth.spline`: the B-spline coefficients inside the range of x, and linearly
/// with the end slopes beyond it.
///
/// # Errors
///
/// `InvalidSpecification` when the result carries no spline coefficients (a result
/// serialised before they were stored).
pub fn smooth_spline_predict(result: &SmoothSplineResult, xnew: &[f64]) -> EconResult<Vec<f64>> {
    let fit = &result.fit;
    let nk = fit.coef.len();
    if nk < 4 || fit.knots.len() != nk + 4 || fit.range.is_nan() || fit.range <= 0.0 {
        return Err(EconError::InvalidSpecification {
            message: "the smoothing spline result carries no spline coefficients; refit it"
                .to_string(),
        });
    }
    let value = |xs: f64| {
        let left = interval(&fit.knots, nk, xs);
        let b = bspline4(&fit.knots, left, xs);
        (0..4).map(|r| b[r] * fit.coef[left - 3 + r]).sum::<f64>()
    };
    let slope = |xs: f64| {
        let left = interval(&fit.knots, nk, xs);
        let b = bspline4_d1(&fit.knots, left, xs);
        (0..4).map(|r| b[r] * fit.coef[left - 3 + r]).sum::<f64>()
    };
    let (e0, e1) = (value(0.0), value(1.0));
    let (s0, s1) = (slope(0.0), slope(1.0));
    Ok(xnew
        .iter()
        .map(|&xi| {
            let xs = (xi - fit.min) / fit.range;
            if xs < 0.0 {
                e0 + s0 * xs
            } else if xs > 1.0 {
                e1 + s1 * (xs - 1.0)
            } else if xs.is_nan() {
                f64::NAN
            } else {
                value(xs)
            }
        })
        .collect())
}

/// Convenience function to run smooth.spline on a Dataset: a df target, or GCV.
///
/// A missing x or y is an `EconError::NullValues` naming the column, as R's
/// `smooth.spline` refuses missing values; integer columns are numbers.
pub fn run_smooth_spline(
    dataset: &Dataset,
    x_col: &str,
    y_col: &str,
    df: Option<f64>,
) -> EconResult<SmoothSplineResult> {
    // R's smooth.spline refuses missing or infinite values: so does this.
    let cols = ColumnRequest::new(dataset.df())
        .numeric(x_col)
        .numeric(y_col)
        .na(NaPolicy::Fail)
        .extract()?;
    let config = match df {
        Some(d) => SmoothSplineConfig::with_df(d),
        None => SmoothSplineConfig::with_criterion(SplineCriterion::Gcv),
    };

    smooth_spline(cols.numeric(x_col), cols.numeric(y_col), config)
}

// ========== R's algorithm ==========

/// sgram.f integrates the quadratic term of `B_i'' B_j''` with 0.333 for 1/3; R's numbers
/// depend on it (registry note for REG-29).
const SGRAM_THIRD: f64 = 0.333;
/// smooth.spline's `control.spar` defaults.
const SPAR_LOW: f64 = -1.5;
const SPAR_HIGH: f64 = 1.5;
const SPAR_TOL: f64 = 1e-4;
const SPAR_EPS: f64 = 2e-8;
const SPAR_MAXIT: usize = 500;

/// What the spar search minimises (sslvrg.f `icrit`).
#[derive(Debug, Clone, Copy)]
enum Criterion {
    Gcv,
    Cv,
    /// `3 + (target - df)^2`
    DfMatch(f64),
}

/// The data reduced to distinct x, the basis and the normal equations (smooth.spline's
/// R-level setup plus sbart's `sgram` and `stxwx`).
struct Setup {
    /// Distinct x (sorted), in the original units.
    ux: Vec<f64>,
    range: f64,
    /// The distinct x each observation belongs to.
    ox: Vec<usize>,
    /// Rescaled weights of the observations (R's `w`).
    w: Vec<f64>,
    wbar: Vec<f64>,
    ybar: Vec<f64>,
    yssw: f64,
    sumw: f64,
    nknots: usize,
    knot: Vec<f64>,
    nk: usize,
    left: Vec<usize>,
    basis: Vec<[f64; 4]>,
    xwx: Vec<[f64; 4]>,
    xwy: Vec<f64>,
    omega: Vec<[f64; 4]>,
    ratio: f64,
}

/// A solve at one lambda.
struct Solved {
    lambda: f64,
    coef: Vec<f64>,
    ty: Vec<f64>,
    lev: Vec<f64>,
    df: f64,
}

/// R's `.nknots.smspl(n)`: all points below 50, then a slowly growing number.
fn nknots_smspl(n: usize) -> usize {
    if n < 50 {
        return n;
    }
    let (a1, a2, a3, a4) = (50f64.log2(), 100f64.log2(), 140f64.log2(), 200f64.log2());
    let nf = n as f64;
    let v = if n < 200 {
        2f64.powf(a1 + (a2 - a1) * (nf - 50.0) / 150.0)
    } else if n < 800 {
        2f64.powf(a2 + (a3 - a2) * (nf - 200.0) / 600.0)
    } else if n < 3200 {
        2f64.powf(a3 + (a4 - a3) * (nf - 800.0) / 2400.0)
    } else {
        200.0 + (nf - 3200.0).powf(0.2)
    };
    v.trunc() as usize
}

/// The 0-based positions R's `xbar[seq.int(1, nx, length.out = nknots)]` selects.
fn knot_positions(nx: usize, nknots: usize) -> impl Iterator<Item = usize> {
    let (from, to) = (1.0, nx as f64);
    let by = if nknots > 1 {
        (to - from) / (nknots - 1) as f64
    } else {
        0.0
    };
    (0..nknots).map(move |i| {
        let v = if i == 0 {
            from
        } else if i == nknots - 1 {
            to
        } else if i < nknots / 2 {
            from + i as f64 * by
        } else {
            to - (nknots - 1 - i) as f64 * by
        };
        (v.trunc() as usize).saturating_sub(1)
    })
}

/// The interval `left` in `[3, nk - 1]` with `knot[left] <= x < knot[left + 1]` (the last
/// interval for `x` at the right end).
fn interval(knot: &[f64], nk: usize, x: f64) -> usize {
    // knot[3..=nk] are the distinct breakpoints
    let p = knot[4..nk].partition_point(|&k| k <= x);
    3 + p
}

impl Setup {
    fn new(x: &[f64], y: &[f64], weights: Option<&[f64]>, all_knots: bool) -> EconResult<Setup> {
        let n = x.len();
        let what = "smoothing spline";
        // R: w <- (w * sum(w > 0)) / sum(w)
        let mut w = resource::try_vec(n, 1.0, what)?;
        if let Some(ws) = weights {
            let npos = ws.iter().filter(|v| **v > 0.0).count() as f64;
            let total: f64 = ws.iter().sum();
            for (wi, &v) in w.iter_mut().zip(ws) {
                *wi = v * npos / total;
            }
        }
        // ties: x within 1e-6 IQR(x) are one value, keyed by round((x - mean) / tol)
        let mut sorted = resource::try_with_capacity(n, what)?;
        sorted.extend_from_slice(x);
        sorted.sort_by(f64::total_cmp);
        let iqr = quantile_sorted(&sorted, 0.75, QuantileType::T7)
            - quantile_sorted(&sorted, 0.25, QuantileType::T7);
        let tol = 1e-6 * iqr;
        let mean = x.iter().sum::<f64>() / n as f64;
        let key: Vec<f64> = if tol > 0.0 {
            x.iter()
                .map(|&v| ((v - mean) / tol).round_ties_even())
                .collect()
        } else {
            x.to_vec()
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| key[a].total_cmp(&key[b]).then(a.cmp(&b)));
        let mut ox = resource::try_vec(n, 0usize, what)?;
        let (mut ux, mut wbar, mut swy, mut swy2) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut i = 0;
        while i < n {
            let k = key[order[i]];
            // the first observation in data order represents the group (R: !duplicated)
            ux.push(x[order[i]]);
            let (mut sw, mut sy, mut sy2) = (0.0, 0.0, 0.0);
            while i < n && key[order[i]] == k {
                let j = order[i];
                ox[j] = ux.len() - 1;
                sw += w[j];
                sy += w[j] * y[j];
                sy2 += w[j] * y[j] * y[j];
                i += 1;
            }
            wbar.push(sw);
            swy.push(sy);
            swy2.push(sy2);
        }
        let nx = ux.len();
        if nx < 4 {
            return Err(EconError::InsufficientData {
                required: 4,
                provided: nx,
                context: "smooth.spline requires at least 4 unique x values".to_string(),
            });
        }
        let ybar: Vec<f64> = swy
            .iter()
            .zip(&wbar)
            .map(|(s, &wb)| s / if wb > 0.0 { wb } else { 1.0 })
            .collect();
        let yssw: f64 = (0..nx).map(|g| swy2[g] - wbar[g] * ybar[g] * ybar[g]).sum();
        let range = ux[nx - 1] - ux[0];
        let xbar: Vec<f64> = ux.iter().map(|&v| (v - ux[0]) / range).collect();

        let nknots = if all_knots { nx } else { nknots_smspl(nx) };
        let nk = nknots + 2;
        let mut knot = resource::try_with_capacity(nk + 4, what)?;
        knot.extend([xbar[0]; 3]);
        if all_knots {
            knot.extend_from_slice(&xbar);
        } else {
            knot.extend(knot_positions(nx, nknots).map(|p| xbar[p]));
        }
        knot.extend([xbar[nx - 1]; 3]);

        let mut left = resource::try_vec(nx, 0usize, what)?;
        let mut basis = resource::try_vec(nx, [0.0; 4], what)?;
        let mut xwx = resource::try_vec(nk, [0.0; 4], what)?;
        let mut xwy = resource::try_vec(nk, 0.0, what)?;
        let mut l = 3;
        for g in 0..nx {
            while l + 1 < nk && xbar[g] >= knot[l + 1] {
                l += 1;
            }
            let b = bspline4(&knot, l, xbar[g]);
            left[g] = l;
            basis[g] = b;
            for r in 0..4 {
                let wb = wbar[g] * b[r];
                xwy[l - 3 + r] += wb * ybar[g];
                for s in r..4 {
                    xwx[l - 3 + r][s - r] += wb * b[s];
                }
            }
        }
        let mut omega = resource::try_vec(nk, [0.0; 4], what)?;
        penalty_bands(&knot, nk, SGRAM_THIRD, &mut omega);
        let (mut t1, mut t2) = (0.0, 0.0);
        for i in 2..nk - 3 {
            t1 += xwx[i][0];
            t2 += omega[i][0];
        }
        let ratio = t1 / t2;
        if !(ratio.is_finite() && ratio > 0.0) {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "smooth.spline: the penalty cannot be scaled on {nx} distinct x values \
                     (the weights of the inner knots' basis are zero)"
                ),
            });
        }
        Ok(Setup {
            sumw: wbar.iter().sum(),
            ux,
            range,
            ox,
            w,
            wbar,
            ybar,
            yssw,
            nknots,
            knot,
            nk,
            left,
            basis,
            xwx,
            xwy,
            omega,
            ratio,
        })
    }

    fn lambda(&self, spar: f64) -> f64 {
        self.ratio * 256f64.powf(3.0 * spar - 1.0)
    }

    /// Banded Cholesky of `X'WX + lambda Omega` (bandwidth 3), the coefficients, the fit and
    /// the leverages `wbar_i b_i' S b_i` from the band of `S = M^-1` (Hutchinson and de Hoog).
    fn solve(&self, lambda: f64) -> EconResult<Solved> {
        resource::checkpoint()?;
        let nk = self.nk;
        let what = "smoothing spline";
        let mut m = resource::try_vec(nk, [0.0; 4], what)?;
        for i in 0..nk {
            for d in 0..4 {
                m[i][d] = self.xwx[i][d] + lambda * self.omega[i][d];
            }
        }
        let mut r = resource::try_vec(nk, [0.0; 4], what)?;
        band_cholesky(&m, &mut r).map_err(|pivot| EconError::InvalidSpecification {
            message: format!(
                "smooth.spline: the penalised normal equations are not positive definite at \
                 lambda = {lambda} (pivot {pivot})"
            ),
        })?;
        let mut coef = self.xwy.clone();
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
        let inv = band_inverse_diag(&r)?;
        let nx = self.ux.len();
        let mut ty = resource::try_vec(nx, 0.0, what)?;
        let mut lev = resource::try_vec(nx, 0.0, what)?;
        let mut df = 0.0;
        for g in 0..nx {
            let (l, b) = (self.left[g] - 3, &self.basis[g]);
            ty[g] = (0..4).map(|q| b[q] * coef[l + q]).sum();
            let mut v = 0.0;
            for q in 0..4 {
                v += b[q] * b[q] * inv[l + q][0];
                for s in q + 1..4 {
                    v += 2.0 * b[q] * b[s] * inv[l + q][s - q];
                }
            }
            lev[g] = v * self.wbar[g];
            df += lev[g];
        }
        Ok(Solved {
            lambda,
            coef,
            ty,
            lev,
            df,
        })
    }

    /// sslvrg.f's criterion on the distinct x; a non-finite value is twice sbart's BIG.
    fn criterion(&self, s: &Solved, crit: Criterion) -> f64 {
        let v = match crit {
            Criterion::Gcv => {
                let mut rss = self.yssw;
                for g in 0..self.ux.len() {
                    let e = self.ybar[g] - s.ty[g];
                    rss += self.wbar[g] * e * e;
                }
                let denom = 1.0 - s.df / self.sumw;
                (rss / self.sumw) / (denom * denom)
            }
            Criterion::Cv => {
                let mut cv = 0.0;
                for g in 0..self.ux.len() {
                    let e = (self.ybar[g] - s.ty[g]) / (1.0 - s.lev[g]);
                    cv += self.wbar[g] * e * e;
                }
                cv / self.sumw
            }
            Criterion::DfMatch(target) => 3.0 + (target - s.df) * (target - s.df),
        };
        if v.is_finite() { v } else { 2.0 * BRENT_BIG }
    }

    /// smooth.spline's `cv.crit` on the observations: ordinary CV with `cv`, else GCV.
    fn cv_crit(&self, y: &[f64], s: &Solved, cv: bool) -> f64 {
        let n = y.len() as f64;
        let (mut num, mut den) = (0.0, 0.0);
        for (i, &yi) in y.iter().enumerate() {
            let g = self.ox[i];
            let mut r = yi - s.ty[g];
            if cv {
                let ww = if self.wbar[g] == 0.0 {
                    1.0
                } else {
                    self.wbar[g]
                };
                r /= 1.0 - s.lev[g] * self.w[i] / ww;
            }
            if self.w[i] != 0.0 {
                num += r * r * self.w[i];
            }
            den += self.w[i];
        }
        let mean = num / den;
        if cv {
            mean
        } else {
            let f = 1.0 - s.df / n;
            mean / (f * f)
        }
    }
}

/// The upper Cholesky factor (`r[i][d]` = `R[i][i + d]`), written to `r`, of a symmetric
/// positive definite bandwidth-3 matrix `m` (`m[i][d]` = entry `(i, i + d)`); `Err(pivot)`
/// (1-based) when a pivot is not positive.
pub(crate) fn band_cholesky(m: &[[f64; 4]], r: &mut [[f64; 4]]) -> Result<(), usize> {
    let nk = m.len();
    for i in 0..nk {
        for d in 0..4.min(nk - i) {
            let j = i + d;
            let mut s = m[i][d];
            for k in j.saturating_sub(3)..i {
                s -= r[k][i - k] * r[k][j - k];
            }
            if d > 0 {
                r[i][d] = s / r[i][0];
            } else if s > 0.0 {
                r[i][0] = s.sqrt();
            } else {
                return Err(i + 1);
            }
        }
    }
    Ok(())
}

/// The band (`inv[i][d]` = entry `(i, i + d)`, `d <= 3`) of `M^-1` from the upper Cholesky
/// factor `r` of a bandwidth-3 `M = R'R` (`r[i][d]` = `R[i][i + d]`), by the recursion of
/// Hutchinson and de Hoog: `R_ii S_ij = delta_ij / R_ii - sum_{k > i} R_ik S_kj`, from the
/// last row up. O(nk).
pub(crate) fn band_inverse_diag(r: &[[f64; 4]]) -> EconResult<Vec<[f64; 4]>> {
    let nk = r.len();
    let mut inv = resource::try_vec(nk, [0.0; 4], "band inverse")?;
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
    Ok(inv)
}

/// The four cubic B-splines non-zero on `[t[left], t[left + 1])`, at `x` (Cox-de Boor).
#[inline]
pub(crate) fn bspline4(t: &[f64], left: usize, x: f64) -> [f64; 4] {
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

/// First derivatives at `x` of the four cubic B-splines of the interval `left`, from the
/// three quadratic B-splines there.
fn bspline4_d1(t: &[f64], left: usize, x: f64) -> [f64; 4] {
    let ratio = |num: f64, den: f64| if den > 0.0 { num / den } else { 0.0 };
    // quadratic B-splines B_{left-2..left, 3} at x
    let mut q = [1.0, 0.0, 0.0];
    let (mut dl, mut dr) = ([0.0; 2], [0.0; 2]);
    for j in 0..2 {
        dr[j] = t[left + j + 1] - x;
        dl[j] = x - t[left - j];
        let mut saved = 0.0;
        for i in 0..=j {
            let term = q[i] / (dr[i] + dl[j - i]);
            q[i] = saved + dr[i] * term;
            saved = dl[j - i] * term;
        }
        q[j + 1] = saved;
    }
    let qq = [0.0, q[0], q[1], q[2], 0.0];
    let mut d1 = [0.0; 4];
    for k in 0..4 {
        let i = left - 3 + k;
        d1[k] = 3.0 * (ratio(qq[k], t[i + 3] - t[i]) - ratio(qq[k + 1], t[i + 4] - t[i + 1]));
    }
    d1
}

/// Second derivatives at `x` of the four cubic B-splines of the interval `left`, with the
/// derivative recursion on the two linear B-splines there. Knots coincide only under a spline
/// that vanishes on this interval, whose term is 0.
pub(crate) fn bspline4_d2(t: &[f64], left: usize, x: f64) -> [f64; 4] {
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

/// The penalty band `Omega_ij = integral of B_i'' B_j''` over the knot range, added into `out`
/// (`out[i][d]` = entry `(i, i + d)`): B'' is linear on each interval, so the integral of a
/// product is `h (a_i a_j + (e_i a_j + e_j a_i) / 2 + e_i e_j third)`, exact with
/// `third = 1/3`; sgram.f uses [`SGRAM_THIRD`].
pub(crate) fn penalty_bands(t: &[f64], nk: usize, third: f64, out: &mut [[f64; 4]]) {
    for left in 3..nk {
        let h = t[left + 1] - t[left];
        if !h.gt(&0.0) {
            continue;
        }
        let a = bspline4_d2(t, left, t[left]);
        let c = bspline4_d2(t, left, t[left + 1]);
        for i in 0..4 {
            let ei = c[i] - a[i];
            for j in i..4 {
                let ej = c[j] - a[j];
                out[left - 3 + i][j - i] +=
                    h * (a[i] * a[j] + (ei * a[j] + ej * a[i]) * 0.5 + ei * ej * third);
            }
        }
    }
}

/// sbart's BIG: a non-finite criterion counts as `2 * BIG`, and any of `fx`, `fv`, `fw` at
/// or above it forces a golden-section step.
const BRENT_BIG: f64 = 1e100;

/// sbart.c's Brent minimiser of `f` on `spar` in `[low, high]` (`tol`, `eps`, at most `maxit`
/// steps), as R writes it. Returns the minimiser; a non-finite `f` counts as `2 * BIG`. One
/// cancellation checkpoint per step.
pub(crate) fn brent_spar(
    low: f64,
    high: f64,
    tol: f64,
    eps: f64,
    maxit: usize,
    mut f: impl FnMut(f64) -> EconResult<f64>,
) -> EconResult<f64> {
    const C_GOLD: f64 = 0.381_966_011_250_105_15;
    let finite = |v: f64| if v.is_finite() { v } else { 2.0 * BRENT_BIG };
    let sign = |a: f64, b: f64| if b >= 0.0 { a.abs() } else { -a.abs() };

    let (mut a, mut b) = (low, high);
    let mut x = a + C_GOLD * (b - a);
    let (mut v, mut w) = (x, x);
    let (mut d, mut e) = (0.0_f64, 0.0_f64);
    let mut fx = finite(f(x)?);
    let (mut fv, mut fw) = (fx, fx);
    let mut iter = 0;
    loop {
        resource::checkpoint()?;
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + tol / 3.0;
        let tol2 = tol1 * 2.0;
        iter += 1;
        if (x - xm).abs() <= tol2 - (b - a) * 0.5 || iter > maxit {
            return Ok(x);
        }
        let mut golden = e.abs() <= tol1 || fx >= BRENT_BIG || fv >= BRENT_BIG || fw >= BRENT_BIG;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn noisy_sine(n: usize, step: f64) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64 * step).collect();
        let y = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.1 * ((i * 7919 % 101) as f64 / 101.0 - 0.5))
            .collect();
        (x, y)
    }

    /// Cubic B-spline `i` (order 4) on `t` at `x` by the Cox-de Boor recursion, and its
    /// derivatives by the derivative recursion: the textbook definition, no shared code.
    fn naive_b(t: &[f64], i: usize, k: usize, x: f64, deriv: usize) -> f64 {
        let frac = |num: f64, den: f64| if den > 0.0 { num / den } else { 0.0 };
        if deriv > 0 {
            let p = (k - 1) as f64;
            return p
                * (frac(naive_b(t, i, k - 1, x, deriv - 1), t[i + k - 1] - t[i])
                    - frac(naive_b(t, i + 1, k - 1, x, deriv - 1), t[i + k] - t[i + 1]));
        }
        if k == 1 {
            return if t[i] <= x && x < t[i + 1] { 1.0 } else { 0.0 };
        }
        frac(x - t[i], t[i + k - 1] - t[i]) * naive_b(t, i, k - 1, x, 0)
            + frac(t[i + k] - x, t[i + k] - t[i + 1]) * naive_b(t, i + 1, k - 1, x, 0)
    }

    fn test_knots() -> Vec<f64> {
        let mut t = vec![0.0; 3];
        t.extend([0.0, 0.07, 0.2, 0.31, 0.55, 0.6, 0.83, 1.0]);
        t.extend([1.0; 3]);
        t
    }

    #[test]
    fn test_bspline4_matches_de_boor_reference() {
        let t = test_knots();
        let nk = t.len() - 4;
        for &x in &[0.0, 0.01, 0.07, 0.15, 0.3099, 0.5, 0.6, 0.71, 0.9999] {
            let left = interval(&t, nk, x);
            assert!(t[left] <= x && x < t[left + 1]);
            let (b, b1, b2) = (
                bspline4(&t, left, x),
                bspline4_d1(&t, left, x),
                bspline4_d2(&t, left, x),
            );
            assert!((b.iter().sum::<f64>() - 1.0).abs() < 1e-15);
            for r in 0..4 {
                let i = left - 3 + r;
                for (got, d) in [(b[r], 0), (b1[r], 1), (b2[r], 2)] {
                    let want = naive_b(&t, i, 4, x, d);
                    assert!(
                        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "x {x} B{i} d{d}: {got} vs {want}"
                    );
                }
            }
        }
        // the right end belongs to the last interval
        assert_eq!(interval(&t, nk, 1.0), nk - 1);
    }

    /// Gauss-Legendre with three points per interval is exact for the quadratic `B_i'' B_j''`.
    #[test]
    fn test_penalty_bands_match_quadrature() {
        let t = test_knots();
        let nk = t.len() - 4;
        let mut band = vec![[0.0; 4]; nk];
        penalty_bands(&t, nk, 1.0 / 3.0, &mut band);
        let g = (0.6f64).sqrt();
        let (nodes, wts) = ([-g, 0.0, g], [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]);
        let mut dense = vec![vec![0.0; nk]; nk];
        for left in 3..nk {
            let (a, b) = (t[left], t[left + 1]);
            for (z, wz) in nodes.iter().zip(wts) {
                let x = 0.5 * (a + b) + 0.5 * (b - a) * z;
                for i in 0..nk {
                    for j in 0..nk {
                        dense[i][j] +=
                            0.5 * (b - a) * wz * naive_b(&t, i, 4, x, 2) * naive_b(&t, j, 4, x, 2);
                    }
                }
            }
        }
        for i in 0..nk {
            for j in 0..nk {
                let got = if j >= i && j - i < 4 {
                    band[i][j - i]
                } else if i > j && i - j < 4 {
                    band[j][i - j]
                } else {
                    0.0
                };
                let want = dense[i][j];
                assert!(
                    (got - want).abs() <= 1e-13 * want.abs().max(1.0),
                    "({i},{j}): {got} vs {want}"
                );
            }
        }
        // sgram's 0.333 only moves the quadratic term
        let mut r = vec![[0.0; 4]; nk];
        penalty_bands(&t, nk, SGRAM_THIRD, &mut r);
        assert!(r[3][0] < band[3][0] && (r[3][0] - band[3][0]).abs() < 1e-2 * band[3][0]);
    }

    #[test]
    fn test_band_inverse_diag_matches_dense_inverse() {
        let n = 40;
        let mut band = vec![[0.0; 4]; n];
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for (i, row) in band.iter_mut().enumerate() {
            row[0] = 4.0 + next();
            for d in 1..4 {
                if i + d < n {
                    row[d] = 0.6 * (next() - 0.5);
                }
            }
        }
        let mut r = vec![[0.0; 4]; n];
        band_cholesky(&band, &mut r).unwrap();
        let inv = band_inverse_diag(&r).unwrap();
        // dense inverse by Gauss-Jordan
        let mut a = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for d in 0..4 {
                if i + d < n {
                    a[i][i + d] = band[i][d];
                    a[i + d][i] = band[i][d];
                }
            }
            a[i][n + i] = 1.0;
        }
        for c in 0..n {
            let p = a[c][c];
            for v in a[c].iter_mut() {
                *v /= p;
            }
            for rr in 0..n {
                if rr != c {
                    let f = a[rr][c];
                    let pivot_row = a[c].clone();
                    for (v, pv) in a[rr].iter_mut().zip(&pivot_row) {
                        *v -= f * pv;
                    }
                }
            }
        }
        for i in 0..n {
            for d in 0..4.min(n - i) {
                let want = a[i][n + i + d];
                assert!(
                    (inv[i][d] - want).abs() <= 1e-13,
                    "({i},{d}): {} vs {want}",
                    inv[i][d]
                );
            }
        }
    }

    #[test]
    fn test_brent_spar_finds_quadratic_minimum() {
        let mut evals = 0;
        let x = brent_spar(-1.5, 1.5, 1e-4, 2e-8, 500, |s| {
            evals += 1;
            Ok((s - 0.3) * (s - 0.3) + 1.0)
        })
        .unwrap();
        assert!((x - 0.3).abs() < 1e-4, "{x}");
        // a parabola is found by the interpolation steps, not by golden sections
        assert!(evals < 12, "{evals}");
        // a non-finite criterion is avoided
        let x = brent_spar(-1.5, 1.5, 1e-4, 2e-8, 500, |s| {
            Ok(if s < 0.0 { f64::NAN } else { (s - 0.8).abs() })
        })
        .unwrap();
        assert!((x - 0.8).abs() < 1e-3, "{x}");
    }

    #[test]
    fn test_nknots_smspl_is_r() {
        // R: sapply(c(10, 49, 50, 120, 199, 200, 500, 799, 800, 3199, 3200, 1e5), .nknots.smspl);
        // 2^log2(50) rounds below 50, so 50 distinct x get 49 knots in R
        let want = [10, 49, 49, 69, 99, 99, 118, 139, 140, 199, 200, 209];
        for (n, w) in [
            10, 49, 50, 120, 199, 200, 500, 799, 800, 3199, 3200, 100_000,
        ]
        .into_iter()
        .zip(want)
        {
            assert_eq!(nknots_smspl(n), w, "n = {n}");
        }
        let p: Vec<usize> = knot_positions(10, 4).collect();
        assert_eq!(p, vec![0, 3, 6, 9]);
    }

    /// x scaled to [0, 1]: rescaling or shifting x leaves the fit as it is.
    #[test]
    fn test_smooth_spline_scale_invariant() {
        let (x, y) = noisy_sine(80, 0.1);
        let base = smooth_spline(&x, &y, SmoothSplineConfig::with_df(7.0)).unwrap();
        for map in [|v: f64| v * 1e5, |v: f64| v + 1e6] {
            let xs: Vec<f64> = x.iter().map(|&v| map(v)).collect();
            let r = smooth_spline(&xs, &y, SmoothSplineConfig::with_df(7.0)).unwrap();
            for (a, b) in r.y.iter().zip(&base.y) {
                assert!((a - b).abs() <= 1e-8, "{a} vs {b}");
            }
            assert!((r.df - base.df).abs() <= 1e-8 && (r.spar - base.spar).abs() <= 1e-8);
        }
    }

    /// x values closer than 1e-6 IQR(x) are one value, as R's smooth.spline merges them.
    #[test]
    fn test_smooth_spline_ties_merge_at_iqr_tolerance() {
        let (x, y) = noisy_sine(40, 0.25);
        let iqr = 4.875; // R: IQR((0:39) * 0.25)
        let mut xt = x.clone();
        xt[10] = x[11] + 1e-8 * iqr; // within 1e-6 IQR of x[11]
        xt[20] = x[21] + 1e-4 * iqr; // well apart
        let r = smooth_spline(&xt, &y, SmoothSplineConfig::with_spar(0.4)).unwrap();
        assert_eq!(r.x.len(), 39);
        // the merged pair shares one fitted value; the first observation in data order is its x
        assert!(r.x.contains(&xt[10]) && !r.x.contains(&x[11]));
        assert_eq!(r.residuals.len(), 40);
        let fit10 = y[10] - r.residuals[10];
        let fit11 = y[11] - r.residuals[11];
        assert_eq!(fit10.to_bits(), fit11.to_bits());
    }

    #[test]
    fn test_smooth_spline_stops_when_cancelled() {
        let (x, y) = noisy_sine(60, 0.1);
        // Every way of choosing lambda, including the two that run no search loop
        for (label, config) in [
            (
                "gcv",
                SmoothSplineConfig::with_criterion(SplineCriterion::Gcv),
            ),
            ("df", SmoothSplineConfig::with_df(6.0)),
            ("spar", SmoothSplineConfig::with_spar(0.5)),
            (
                "lambda",
                SmoothSplineConfig {
                    lambda: Some(0.01),
                    ..Default::default()
                },
            ),
        ] {
            let e =
                resource::with_budget(cancelled(), || smooth_spline(&x, &y, config)).unwrap_err();
            assert!(matches!(e, EconError::Cancelled(_)), "{label}: {e:?}");
        }
    }

    #[test]
    fn test_smooth_spline_large_n_is_linear_memory() {
        // Post-fix guard. Do NOT run against the dense code: it allocates two
        // 80 GB n x n tables.
        let n = 100_000;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / 1000.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.1 * ((i * 7919 % 1009) as f64 / 1009.0 - 0.5))
            .collect();
        let r = smooth_spline(&x, &y, SmoothSplineConfig::with_df(10.0)).unwrap();
        assert_eq!(r.n_obs, n);
        assert_eq!(r.n_knots, nknots_smspl(n));
    }

    #[test]
    fn test_smooth_spline_basic() {
        // Noisy sine curve
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| xi.sin() + 0.1 * (xi * 10.0).sin())
            .collect();

        let config = SmoothSplineConfig::with_df(10.0);
        let result = smooth_spline(&x, &y, config).unwrap();

        assert_eq!(result.n_obs, 50);
        assert!(result.df > 5.0 && result.df < 20.0);
        assert!(!result.residuals.is_empty());
    }

    #[test]
    fn test_smooth_spline_cv() {
        let x: Vec<f64> = (0..30).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi.powi(2) + 0.1 * xi + 0.05 * ((i * 7) % 5) as f64)
            .collect();

        let config = SmoothSplineConfig::with_criterion(SplineCriterion::Cv);
        let result = smooth_spline(&x, &y, config).unwrap();

        assert!(result.cv_crit.is_some());
        assert!(result.df > 2.0);
    }

    #[test]
    fn test_smooth_spline_too_few_points() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];

        let result = smooth_spline(&x, &y, SmoothSplineConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_smooth_spline_predict() {
        let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi.powi(2) + 0.3 * ((i * 7) % 5) as f64)
            .collect();

        let result = smooth_spline(&x, &y, SmoothSplineConfig::with_df(5.0)).unwrap();

        // at the data, predict gives the fitted values
        let at = smooth_spline_predict(&result, &result.x).unwrap();
        for (a, b) in at.iter().zip(&result.y) {
            assert!((a - b).abs() <= 1e-10 * b.abs().max(1.0), "{a} vs {b}");
        }
        let xnew = vec![5.5, 10.5, 15.5];
        let ynew = smooth_spline_predict(&result, &xnew).unwrap();
        for (i, &xi) in xnew.iter().enumerate() {
            let expected = xi.powi(2);
            let diff = (ynew[i] - expected).abs();
            assert!(
                diff < 10.0,
                "At x={xi}, expected ~{expected}, got {}",
                ynew[i]
            );
        }
        // linear beyond the data
        let out = smooth_spline_predict(&result, &[-2.0, -1.0, 0.0, 19.0, 20.0, 21.0]).unwrap();
        assert!(((out[1] - out[0]) - (out[2] - out[1])).abs() < 1e-9);
        assert!(((out[5] - out[4]) - (out[4] - out[3])).abs() < 1e-6 * out[5].abs());
        // a result without coefficients (serialised before they were stored) is refused
        let mut old = result.clone();
        old.fit = SmoothSplineFit::default();
        assert!(smooth_spline_predict(&old, &xnew).is_err());
    }

    #[test]
    fn test_smooth_spline_high_df_interpolates() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![0.0, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 5.0];

        // High df should be close to interpolation
        let config = SmoothSplineConfig::with_df(9.5);
        let result = smooth_spline(&x, &y, config).unwrap();

        for (i, &yi) in y.iter().enumerate() {
            let diff = (result.y[i] - yi).abs();
            assert!(
                diff < 1.0,
                "At index {i}, expected ~{yi}, got {}",
                result.y[i]
            );
        }
    }

    #[test]
    fn test_smooth_spline_low_df_smooths() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 / 10.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi + 0.5 * (if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();

        let config = SmoothSplineConfig::with_df(3.0);
        let result = smooth_spline(&x, &y, config).unwrap();

        // Fitted values should follow the trend y = x, not the noise
        let trend_match = x
            .iter()
            .zip(&result.y)
            .filter(|(xi, fi)| (*fi - *xi).abs() < 1.0)
            .count();
        assert!(
            trend_match as f64 / x.len() as f64 > 0.8,
            "Low df should follow the linear trend"
        );
    }

    /// REG-44: df = 6 is met within [`DF_TARGET_TOL`], with no warning.
    #[test]
    fn test_smooth_spline_df_target_is_met() {
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| (xi / 5.0).sin() + 0.05 * (((i * 7) % 5) as f64 - 2.0))
            .collect();

        let result = smooth_spline(&x, &y, SmoothSplineConfig::with_df(6.0)).unwrap();
        assert!(
            (result.df - 6.0).abs() < DF_TARGET_TOL,
            "Expected df ~6, got {}",
            result.df
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);

        let max_resid = result.residuals.iter().map(|r| r.abs()).fold(0.0, f64::max);
        assert!(max_resid < 0.5, "Residuals too large: max = {max_resid}");
    }

    /// REG-53: with neither df, spar nor lambda the smoothing parameter minimises GCV (R's
    /// default cv = FALSE); SmoothSplineConfig::cv = true is R's cv = TRUE (ordinary
    /// leave-one-out CV). The reported criterion is R's `cv.crit` on the returned fit.
    #[test]
    #[allow(deprecated)]
    fn test_smooth_spline_default_criterion_is_gcv() {
        use crate::data::columns::{NaPolicy, numeric_column};
        let ds = crate::regression::ols::na_fixtures::repo_dataset(
            "validation/datasets/generated/smoothers_t3.csv",
        );
        let x = numeric_column(ds.df(), "x", NaPolicy::Fail).unwrap().0;
        let y = numeric_column(ds.df(), "y", NaPolicy::Fail).unwrap().0;
        let def = smooth_spline(&x, &y, SmoothSplineConfig::default()).unwrap();
        let gcv = smooth_spline(
            &x,
            &y,
            SmoothSplineConfig::with_criterion(SplineCriterion::Gcv),
        )
        .unwrap();
        let cv = smooth_spline(
            &x,
            &y,
            SmoothSplineConfig::with_criterion(SplineCriterion::Cv),
        )
        .unwrap();
        let mut r_cv = SmoothSplineConfig::default();
        r_cv.cv = true;
        let r_cv = smooth_spline(&x, &y, r_cv).unwrap();
        assert_eq!(def.criterion, Some(SplineCriterion::Gcv));
        assert_eq!(
            (def.lambda.to_bits(), def.df.to_bits()),
            (gcv.lambda.to_bits(), gcv.df.to_bits())
        );
        assert_eq!(
            (r_cv.criterion, r_cv.lambda.to_bits()),
            (Some(SplineCriterion::Cv), cv.lambda.to_bits())
        );
        assert_ne!(def.lambda.to_bits(), cv.lambda.to_bits());
        // the reported criterion is R's definition on the returned fit (100 distinct x,
        // unit weights)
        let n = x.len() as f64;
        let rss: f64 = def.residuals.iter().map(|r| r * r).sum();
        let gcv_formula = rss / n / (1.0 - def.df / n).powi(2);
        assert!(
            (def.cv_crit.unwrap() - gcv_formula).abs() <= 1e-12 * gcv_formula,
            "{:?} vs {gcv_formula}",
            def.cv_crit
        );
        let lev = cv.leverage.as_ref().unwrap();
        let cv_formula = y
            .iter()
            .zip(&x)
            .map(|(yi, xi)| {
                let g = cv.x.iter().position(|u| u == xi).unwrap();
                ((yi - cv.y[g]) / (1.0 - lev[g])).powi(2)
            })
            .sum::<f64>()
            / n;
        assert!((cv.cv_crit.unwrap() - cv_formula).abs() <= 1e-12 * cv_formula);
        // with_cv keeps its old meaning for source compatibility: true = GCV
        assert_eq!(
            smooth_spline(&x, &y, SmoothSplineConfig::with_cv(true))
                .unwrap()
                .criterion,
            Some(SplineCriterion::Gcv)
        );
        assert_eq!(
            smooth_spline(&x, &y, SmoothSplineConfig::with_df(6.0))
                .unwrap()
                .criterion,
            None
        );
    }

    #[test]
    fn test_spline_criterion_parses_aliases_and_rejects_unknown() {
        SplineCriterion::choices().validate().unwrap();
        for s in [
            "gcv",
            "GCV",
            "generalized-cv",
            "Generalized Cross Validation",
        ] {
            assert_eq!(
                s.parse::<SplineCriterion>().unwrap(),
                SplineCriterion::Gcv,
                "{s}"
            );
        }
        for s in ["cv", "LOOCV", "loo", "ordinary_cv", "leave-one-out"] {
            assert_eq!(
                s.parse::<SplineCriterion>().unwrap(),
                SplineCriterion::Cv,
                "{s}"
            );
        }
        assert_eq!(SplineCriterion::Gcv.to_string(), "gcv");
        match "aic".parse::<SplineCriterion>().unwrap_err() {
            EconError::InvalidSpecification { message } => assert!(
                message.contains("criterion") && message.contains("gcv") && message.contains("cv"),
                "{message}"
            ),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn test_smooth_spline_rejects_invalid_controls() {
        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.3).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v.sin() + 0.05 * ((i * 7) % 5) as f64)
            .collect();
        let bad = |c: SmoothSplineConfig, what: &str| match smooth_spline(&x, &y, c).unwrap_err() {
            EconError::InvalidSpecification { message } => {
                assert!(message.contains(what), "{what}: {message}")
            }
            e => panic!("{what}: {e:?}"),
        };
        bad(SmoothSplineConfig::with_df(1.0), "df");
        bad(SmoothSplineConfig::with_df(21.0), "df");
        bad(SmoothSplineConfig::with_df(f64::NAN), "df");
        bad(SmoothSplineConfig::with_spar(f64::INFINITY), "spar");
        let mut c = SmoothSplineConfig::default();
        c.lambda = Some(-1.0);
        bad(c, "lambda");
        let mut c = SmoothSplineConfig::default();
        c.weights = Some(vec![-1.0; 20]);
        bad(c, "weights");
    }

    /// REG-25: run_smooth_spline refuses missing values as R's smooth.spline does
    /// ("missing or infinite values in inputs are not allowed") and accepts Int64. A
    /// property test: no value is compared with R (the R pins are the `test_validate_*` tests),
    /// so it loads the datasets without a case.
    #[test]
    fn test_na_smooth_spline_columns() {
        use crate::regression::ols::na_fixtures::{as_float, mtcars, smoothers_na};
        let mt = mtcars();
        // x = hp is Int64, with ties
        let a = run_smooth_spline(&mt, "hp", "mpg", Some(4.0)).unwrap();
        let b = run_smooth_spline(&as_float(&mt, "hp"), "hp", "mpg", Some(4.0)).unwrap();
        assert_eq!(
            (a.n_obs, &a.x, &a.y, a.df, a.lambda),
            (32, &b.x, &b.y, b.df, b.lambda)
        );
        let sm = smoothers_na();
        for (x, y, column) in [("x", "y", "x"), ("id", "y", "y")] {
            match run_smooth_spline(&sm, x, y, None) {
                Err(EconError::NullValues { column: c, count }) => {
                    assert_eq!((c.as_str(), count), (column, 2))
                }
                other => panic!("{other:?}"),
            }
        }
        assert!(matches!(
            run_smooth_spline(&mt, "model", "mpg", None),
            Err(EconError::NonNumericColumn { .. })
        ));
        assert!(matches!(
            run_smooth_spline(&mt, "nope", "mpg", None),
            Err(EconError::ColumnNotFound { .. })
        ));
    }

    fn t4_columns(file: &str, cols: &[&str]) -> Vec<Vec<f64>> {
        use crate::data::columns::{NaPolicy, numeric_column};
        let ds = crate::regression::ols::na_fixtures::repo_dataset(&format!(
            "validation/datasets/generated/{file}.csv"
        ));
        cols.iter()
            .map(|c| numeric_column(ds.df(), c, NaPolicy::Fail).unwrap().0)
            .collect()
    }

    /// Case `smooth_spline_t4` (REG-29, REG-30): fits at a given spar or lambda on data A
    /// (n = 120, 69 knots), and with every distinct x a knot. The a_* keys are smooth.spline's
    /// algorithm written out; the r_* keys smooth.spline itself.
    #[test]
    fn test_validate_smooth_spline_spar_fixed() {
        use p2a_validation::{Check, Tol};
        let case = p2a_validation::Case::load("smooth_spline_t4");
        let a = t4_columns("smooth_spline_t4a", &["x", "y"]);
        let (x, y) = (&a[0], &a[1]);
        let t = Tol::coef(120);
        let r = smooth_spline(x, y, SmoothSplineConfig::with_spar(0.5)).unwrap();
        let mut c = Check::new(&case);
        c.scalar("a_nknots", r.n_knots as f64, Tol::exact())
            .scalar("a_nx", r.x.len() as f64, Tol::exact())
            .scalar("a_spar05_lambda", r.lambda, t)
            .scalar("r_a_spar05_lambda", r.lambda, t)
            .scalar("a_spar05_df", r.df, t)
            .scalar("r_a_spar05_df", r.df, t)
            .vector("a_spar05_fit", &r.y, t)
            .vector("r_a_spar05_fit", &r.y, t)
            .vector("a_spar05_lev", r.leverage.as_deref().unwrap(), t)
            .scalar("a_spar05_gcv_crit", r.cv_crit.unwrap(), t)
            .vector(
                "a_spar05_predict",
                &smooth_spline_predict(&r, &case.vector("a_predict_x")).unwrap(),
                t,
            );
        let mut lam = SmoothSplineConfig::default();
        lam.lambda = Some(1e-4);
        let r = smooth_spline(x, y, lam).unwrap();
        c.scalar("a_lambda1e4_df", r.df, t)
            .vector("a_lambda1e4_fit", &r.y, t);
        let mut all = SmoothSplineConfig::with_spar(0.5);
        all.all_knots = true;
        let r = smooth_spline(x, y, all).unwrap();
        c.scalar("a_allknots_spar05_df", r.df, t)
            .vector("a_allknots_spar05_fit", &r.y, t);
        c.finish();
        assert_eq!(r.n_knots, 120);
    }

    /// Case `smooth_spline_t4` (REG-29, REG-30): spar searched by sbart's Brent minimiser for
    /// a df target, GCV and CV on data A; GCV and df = 5 with weights and tied x on data B
    /// (150 rows, 81 distinct x); df = 6 on data C (x = 1:30). Brent stops at 1e-4 in spar,
    /// so these pins hold because both sides run the same search: iterative tolerance.
    #[test]
    fn test_validate_smooth_spline_searched() {
        use p2a_validation::{Check, Tol};
        let case = p2a_validation::Case::load("smooth_spline_t4");
        let t = Tol::iterative(1e-6);
        let mut c = Check::new(&case);
        let put = |c: &mut Check, p: &str, r: &SmoothSplineResult, lev: bool| {
            c.scalar(&format!("{p}spar"), r.spar, t)
                .scalar(&format!("{p}lambda"), r.lambda, t)
                .scalar(&format!("{p}df"), r.df, t)
                .scalar(&format!("{p}cv_crit"), r.cv_crit.unwrap(), t)
                .vector(&format!("{p}fit"), &r.y, t);
            if lev {
                c.vector(&format!("{p}lev"), r.leverage.as_deref().unwrap(), t);
            }
        };
        let a = t4_columns("smooth_spline_t4a", &["x", "y"]);
        for (tag, config) in [
            ("df6", SmoothSplineConfig::with_df(6.0)),
            ("gcv", SmoothSplineConfig::default()),
            (
                "cv",
                SmoothSplineConfig::with_criterion(SplineCriterion::Cv),
            ),
        ] {
            let r = smooth_spline(&a[0], &a[1], config).unwrap();
            put(&mut c, &format!("a_{tag}_"), &r, true);
            put(&mut c, &format!("r_a_{tag}_"), &r, false);
        }
        let b = t4_columns("smooth_spline_t4b", &["x", "y", "w"]);
        for (tag, df) in [("gcv", None), ("df5", Some(5.0))] {
            let mut config = SmoothSplineConfig::default();
            config.df = df;
            config.weights = Some(b[2].clone());
            let r = smooth_spline(&b[0], &b[1], config).unwrap();
            put(&mut c, &format!("b_{tag}_"), &r, true);
            put(&mut c, &format!("r_b_{tag}_"), &r, false);
            c.vector(&format!("b_{tag}_residuals"), &r.residuals, t)
                .vector("b_ux", &r.x, Tol::exact())
                .scalar("b_nknots", r.n_knots as f64, Tol::exact());
        }
        let cc = t4_columns("smooth_spline_t4c", &["x", "y"]);
        let r = smooth_spline(&cc[0], &cc[1], SmoothSplineConfig::with_df(6.0)).unwrap();
        c.scalar("c_nknots", r.n_knots as f64, Tol::exact())
            .scalar("c_df6_spar", r.spar, t)
            .scalar("c_df6_lambda", r.lambda, t)
            .scalar("c_df6_df", r.df, t)
            .vector("c_df6_fit", &r.y, t)
            .scalar("r_c_df6_df", r.df, t)
            .vector("r_c_df6_fit", &r.y, t);
        c.finish();
    }

    /// Case `smoothers_gcv_periodic_loess` (x = sorted runif(100), y = sin(2 pi x) + noise):
    /// smooth.spline with cv = FALSE, cv = TRUE and df = 6. Searched, so iterative as
    /// `test_validate_smooth_spline_searched`.
    #[test]
    fn test_validate_smooth_spline_gcv_cv_t3() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("smoothers_gcv_periodic_loess");
        let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
        let (x, y) = (col("x"), col("y"));
        let t = Tol::iterative(1e-6);
        let gcv = smooth_spline(&x, &y, SmoothSplineConfig::default()).unwrap();
        let cv = smooth_spline(
            &x,
            &y,
            SmoothSplineConfig::with_criterion(SplineCriterion::Cv),
        )
        .unwrap();
        let d6 = smooth_spline(&x, &y, SmoothSplineConfig::with_df(6.0)).unwrap();
        Check::new(&case)
            .scalar("gcv_df", gcv.df, t)
            .scalar("gcv_lambda", gcv.lambda, t)
            .scalar("gcv_crit", gcv.cv_crit.unwrap(), t)
            .vector(
                "gcv_fit_head",
                &smooth_spline_predict(&gcv, &x[..10]).unwrap(),
                t,
            )
            .scalar("cv_df", cv.df, t)
            .scalar("cv_lambda", cv.lambda, t)
            .scalar("cv_crit", cv.cv_crit.unwrap(), t)
            .scalar("dfmatch6_df", d6.df, t)
            .scalar("dfmatch6_lambda", d6.lambda, t)
            .finish();
    }
}
