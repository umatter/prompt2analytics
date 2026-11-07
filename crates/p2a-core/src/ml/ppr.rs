//! Projection pursuit regression: Friedman's SMART, as R's `stats::ppr`.
//!
//! The model is `y = ybar + sy sum_k b_k f_k(alpha_k' x) + e`, with unit directions `alpha_k`
//! and ridge functions `f_k` that are smooths, standardised to mean 0 and variance 1. PPR's
//! estimate is the local optimum this algorithm reaches, so the algorithm is the method's
//! definition; R's `ppr.f` (R 4.3.3) is followed in its steps and its stopping rules:
//!
//! 1. **Forward pass** (`subfit`): add terms up to `max_terms`, each from a linear start
//!    (`pprdir`), stopping when another term lowers the RSS by less than `tol` (relative).
//! 2. **One term** (`oneone`): a Gauss-Newton search on the direction with step halving. The
//!    ridge function is the smooth of the residuals on the sorted projections, and the step
//!    (`pprdir`, one conjugate-gradient pass of `ppconj`) uses its derivative (`pprder`, on
//!    projections pooled by `pool`).
//! 3. **Backfitting** (`fulfit`) cycles through the terms until the RSS stops improving.
//! 4. **Backward elimination** (`smart1`): from `max_terms` down to `nterms`, drop the term
//!    with the smallest `|b|` (ties to the lowest term index) and backfit again.
//!
//! # Ridge smoothers
//!
//! - `supsmu`: Friedman's super smoother, with `bass`, and `span = 0` (cross-validated) or a
//!   fixed span.
//! - `spline`: a 15-coefficient penalised cubic B-spline, the smoothing parameter matched to
//!   `df`.
//! - `gcvspline`: the same spline, the smoothing parameter chosen by GCV with penalty `gcvpen`.
//!
//! # Optimisation levels (R's `optlevel`)
//!
//! - 0: no backfitting and no backward elimination: the forward pass is the model.
//! - 1: backfit, re-estimating `b` and `f` only (no direction steps; the step halving stops at
//!   once).
//! - 2: as 1, and each direction takes one Gauss-Newton step per pass (the default).
//! - 3: backfitting runs the full direction search for each term.
//!
//! # Fitted values and prediction
//!
//! Fitted values, residuals and `predict` come from the stored ridge functions (R's `pppred`):
//! exact at training projections, linear between them, constant beyond their range.
//!
//! # Differences from R
//!
//! - `optlevel = 0` with `max_terms > nterms` keeps every forward term and warns. R's object
//!   lists only `nterms` directions but predicts with all of them.
//! - `max_terms < nterms` and fewer than 4 rows are errors, as are projections pooling into
//!   fewer than three groups. R returns unrelated storage or reads out of bounds there.
//! - The spline smoothers integrate the penalty exactly and return the fit at the reported
//!   smoothing parameter (`ppr/spline.rs`; R does neither), so spline fits agree with R
//!   only to about 1%. A term's `edf` belongs to its stored ridge function; R reports that of the
//!   last smoother call, a rejected trial included, and at `optlevel = 0` only for the first term.
//!
//! # References
//!
//! - Friedman, J.H., & Stuetzle, W. (1981). Projection pursuit regression.
//!   *Journal of the American Statistical Association*, 76(376), 817-823.
//!   https://doi.org/10.1080/01621459.1981.10477729
//!
//! - Friedman, J.H. (1984). SMART: Smooth multiple additive regression technique.
//!   Technical Report No. 1, Laboratory for Computational Statistics, Stanford University.
//!
//! - Friedman, J.H. (1984). A variable span smoother. Technical Report No. 5,
//!   Laboratory for Computational Statistics, Stanford University.
//!
//! R equivalent: `stats::ppr()`

use ndarray::{ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::fmt::Write as _;

use crate::errors::{EconError, EconResult};
use crate::resource;
use crate::traits::convergence::{Convergence, ResultNotes, notes_text};

pub(crate) mod spline;
pub(crate) mod supsmu;

/// SMART's convergence threshold `conv`: R's block data stores the REAL literal 0.005 in a DOUBLE
/// PRECISION variable, so the value is `0.005_f32`.
pub const PPR_CONV: f64 = 0.005_f32 as f64;
/// The step halving gives up below this factor (`cutmin`).
const CUTMIN: f64 = 0.1_f32 as f64;
/// Projections closer than this share of the interquartile range are pooled (`fdel`).
const FDEL: f64 = 0.02_f32 as f64;
/// Conjugate gradients' stopping threshold (`cjeps`).
const CJEPS: f64 = 0.001_f32 as f64;
/// The largest RSS (`big`).
const BIG: f64 = 1.0e20_f32 as f64;
/// Passes of conjugate gradients per direction step (`mitcj`).
const MITCJ: usize = 1;

/// Result of projection pursuit regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PprResult {
    /// Number of terms in the returned model
    pub nterms: usize,
    /// Projection directions: `alpha[k]` is the unit direction of term k (R's `alpha[, k]`)
    pub alpha: Vec<Vec<f64>>,
    /// Ridge coefficients on the response scale (R's `beta`, `y_scale * b`)
    pub beta: Vec<f64>,
    /// Fitted values from the ridge functions
    pub fitted: Vec<f64>,
    /// `y - fitted`
    pub residuals: Vec<f64>,
    /// R's `gofn`, of length `max_terms`: the goodness of fit of the model with `l + 1` terms, 0
    /// for sizes below `nterms` and for sizes not refitted (`optlevel = 0`)
    pub gofn: Vec<f64>,
    /// Number of observations
    pub n: usize,
    /// Number of predictors
    pub p: usize,
    /// The smoothing method used
    pub sm_method: SmoothingMethod,
    /// Terms of the forward pass requested
    #[serde(default)]
    pub max_terms: usize,
    /// Optimisation level
    #[serde(default)]
    pub optlevel: usize,
    /// R's `gof`: the RSS the search tracked, as `n / sum(w) * sum(w (y - fitted)^2)`; just above
    /// that of `fitted` when the last term of the final pass kept its old fit
    #[serde(default)]
    pub gof: f64,
    /// Equivalent degrees of freedom per term (spline methods; 0 for `supsmu`)
    #[serde(default)]
    pub edf: Vec<f64>,
    /// Weighted mean of the response (R's `yb`)
    #[serde(default)]
    pub y_mean: f64,
    /// Weighted standard deviation of the response (R's `ys`)
    #[serde(default)]
    pub y_scale: f64,
    /// SMART's coefficients on the standardised response (`beta = y_scale * b`)
    #[serde(default)]
    pub beta_standardized: Vec<f64>,
    /// Per term: the training projections, sorted
    #[serde(default)]
    pub ridge_projections: Vec<Vec<f64>>,
    /// Per term: the standardised ridge function at those projections
    #[serde(default)]
    pub ridge_values: Vec<Vec<f64>>,
    /// The super smoother's `bass`, as given
    #[serde(default)]
    pub bass: f64,
    /// The super smoother's span, as given (0 = cross-validated)
    #[serde(default)]
    pub span: f64,
    /// The df `spline` matches, as given
    #[serde(default)]
    pub df: f64,
    /// The GCV penalty of `gcvspline`, as given
    #[serde(default)]
    pub gcvpen: f64,
    /// The final backfitting (`None` for `optlevel = 0`, which does none)
    #[serde(default)]
    pub convergence: Option<Convergence>,
    /// What the caller should know about this fit
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl PprResult {
    /// The prediction at one row of `p` values (R's `pppred`).
    fn predict_row(&self, row: &[f64]) -> f64 {
        let mut y = 0.0;
        for k in 0..self.nterms {
            let (Some(t), Some(f)) = (self.ridge_projections.get(k), self.ridge_values.get(k))
            else {
                return f64::NAN;
            };
            if t.is_empty() || t.len() != f.len() {
                return f64::NAN;
            }
            let mut s = 0.0;
            for (a, v) in self.alpha[k].iter().zip(row) {
                s += a * v;
            }
            y += self.beta_standardized[k] * ridge_value(t, f, s);
        }
        self.y_scale * y + self.y_mean
    }

    /// Predict for new rows; a row whose length is not `p`, or with a non-finite value, gives
    /// NaN. [`PprResult::try_predict`] refuses them instead.
    pub fn predict(&self, x_new: &[Vec<f64>]) -> Vec<f64> {
        x_new
            .iter()
            .map(|row| {
                if row.len() == self.p && row.iter().all(|v| v.is_finite()) {
                    self.predict_row(row)
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Predict for a matrix of new rows.
    ///
    /// # Errors
    ///
    /// `InvalidSpecification` when `x` does not have `p` columns or holds a non-finite value.
    pub fn try_predict(&self, x: ArrayView2<f64>) -> EconResult<Vec<f64>> {
        if x.ncols() != self.p {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "PPR: wrong number of columns in 'x' ({} expected, got {})",
                    self.p,
                    x.ncols()
                ),
            });
        }
        require_finite_columns(x)?;
        let mut out = resource::try_with_capacity(x.nrows(), "PPR predictions")?;
        let mut row = resource::try_vec(self.p, 0.0, "PPR predictions")?;
        for r in x.outer_iter() {
            for (slot, v) in row.iter_mut().zip(r.iter()) {
                *slot = *v;
            }
            out.push(self.predict_row(&row));
        }
        Ok(out)
    }
}

/// The ridge function at `s`: exact at a projection, linear between two, constant beyond them.
fn ridge_value(t: &[f64], f: &[f64], s: f64) -> f64 {
    let n = t.len();
    if !(s > t[0]) {
        return f[0];
    }
    if !(s < t[n - 1]) {
        return f[n - 1];
    }
    // t[low] < s < t[high]
    let (mut low, mut high) = (0, n - 1);
    while low + 1 < high {
        let place = (low + high) / 2;
        let tv = t[place];
        if s == tv {
            return f[place];
        }
        if s < tv {
            high = place;
        } else {
            low = place;
        }
    }
    f[low] + (f[high] - f[low]) * (s - t[low]) / (t[high] - t[low])
}

impl ResultNotes for PprResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        self.convergence.clone()
    }
}

impl std::fmt::Display for PprResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Projection pursuit regression (SMART, as R's ppr)")?;
        writeln!(f, "Observations: {}  Predictors: {}", self.n, self.p)?;
        writeln!(f, "Terms: {}  max.terms: {}", self.nterms, self.max_terms)?;
        let options = match self.sm_method {
            SmoothingMethod::Supsmu if self.span > 0.0 => {
                format!("span = {}, bass = {}", self.span, self.bass)
            }
            SmoothingMethod::Supsmu => format!("span = cv, bass = {}", self.bass),
            SmoothingMethod::Spline => format!("df = {}", self.df),
            SmoothingMethod::GcvSpline => format!("gcvpen = {}", self.gcvpen),
        };
        writeln!(
            f,
            "Smoother: {} ({options})  optlevel: {}",
            self.sm_method, self.optlevel
        )?;
        writeln!(f, "Goodness of fit: {:.6}", self.gof)?;
        for (l, g) in self.gofn.iter().enumerate() {
            if l + 1 >= self.nterms {
                writeln!(f, "  {} terms: {g:.6}", l + 1)?;
            }
        }
        writeln!(f, "Projection direction vectors (alpha):")?;
        for j in 0..self.p {
            let mut line = format!("  X{:<6}", j + 1);
            for a in &self.alpha {
                let _ = write!(line, " {:>12.6}", a[j]);
            }
            writeln!(f, "{line}")?;
        }
        let mut line = String::from("Coefficients of ridge terms (beta):\n  ");
        for b in &self.beta {
            let _ = write!(line, " {b:>12.6}");
        }
        writeln!(f, "{line}")?;
        if self.sm_method != SmoothingMethod::Supsmu {
            let mut line = String::from("Equivalent df:\n  ");
            for d in &self.edf {
                let _ = write!(line, " {d:>12.2}");
            }
            writeln!(f, "{line}")?;
        }
        f.write_str(&notes_text(self))
    }
}

/// Smoothing method for ridge functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmoothingMethod {
    /// Friedman's SuperSmoother
    Supsmu,
    /// Penalised cubic B-spline with the smoothing parameter matched to `df`
    Spline,
    /// Penalised cubic B-spline with the smoothing parameter chosen by GCV
    GcvSpline,
}

crate::option_enum!(
    SmoothingMethod,
    crate::options::Choices {
        option: "sm.method",
        values: &[
            (
                &["supsmu", "super_smoother", "supersmoother"],
                SmoothingMethod::Supsmu
            ),
            (&["spline", "smoothing_spline"], SmoothingMethod::Spline),
            (
                &["gcvspline", "gcv_spline", "gcv"],
                SmoothingMethod::GcvSpline
            ),
        ],
    }
);

/// Configuration for PPR.
#[derive(Debug, Clone)]
pub struct PprConfig {
    /// Number of terms in the final model, at least 1.
    pub nterms: usize,
    /// Terms of the forward pass (R's `max.terms`), at least `nterms` and at most the number of
    /// observations.
    pub max_terms: usize,
    /// Optimisation level, 0 to 3 (R's `optlevel`; see the module documentation).
    pub optlevel: usize,
    /// Ridge smoother
    pub sm_method: SmoothingMethod,
    /// `supsmu` only: bass control in [0, 10], larger gives smoother functions.
    pub bass: f64,
    /// `supsmu` only: 0 for the cross-validated span, or a fixed span in (0, 1].
    pub span: f64,
    /// `spline` only: equivalent df of each ridge function, in (1, min(n, 15)].
    pub df: f64,
    /// SMART's `maxit` and `mitone`: at least 1.
    pub max_iter: usize,
    /// SMART's `conv`: relative decrease of the RSS below which a search stops.
    pub tol: f64,
    /// `gcvspline` only: the penalty of each df in the GCV denominator, at least 0.
    pub gcvpen: f64,
}

impl Default for PprConfig {
    fn default() -> Self {
        PprConfig {
            nterms: 1,
            max_terms: 1,
            optlevel: 2,
            sm_method: SmoothingMethod::Supsmu,
            bass: 0.0,
            span: 0.0,
            df: 5.0,
            max_iter: 20,
            tol: PPR_CONV,
            gcvpen: 1.0,
        }
    }
}

fn invalid(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

fn require_finite_columns(x: ArrayView2<f64>) -> EconResult<()> {
    for (j, column) in x.axis_iter(Axis(1)).enumerate() {
        let k = column.iter().filter(|v| !v.is_finite()).count();
        if k > 0 {
            return Err(invalid(format!(
                "PPR: feature column {j} (0-based) has {k} non-finite values (NaN or infinite); \
                 remove or impute them first"
            )));
        }
    }
    Ok(())
}

/// Every parameter against what the data can support, before any work.
fn validate(
    x: ArrayView2<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    c: &PprConfig,
) -> EconResult<()> {
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(invalid(format!(
            "y has length {} but x has {n} rows",
            y.len()
        )));
    }
    if n < 4 {
        return Err(invalid(format!(
            "PPR needs at least 4 observations, got {n}"
        )));
    }
    if p == 0 {
        return Err(invalid("PPR needs at least one predictor".to_string()));
    }
    if c.nterms == 0 {
        return Err(invalid("nterms must be at least 1".to_string()));
    }
    let largest = c.max_terms.max(c.nterms);
    if largest > n {
        return Err(invalid(format!(
            "the number of terms ({largest}; the larger of nterms and max_terms) must not \
             exceed the number of observations ({n})"
        )));
    }
    if c.max_terms < c.nterms {
        return Err(invalid(format!(
            "max_terms ({}) must be at least nterms ({})",
            c.max_terms, c.nterms
        )));
    }
    if c.optlevel > 3 {
        return Err(invalid(format!(
            "optlevel must be 0 to 3, got {}",
            c.optlevel
        )));
    }
    if c.max_iter == 0 {
        return Err(invalid("max_iter must be at least 1, got 0".to_string()));
    }
    if !(c.tol.is_finite() && c.tol > 0.0) {
        return Err(invalid(format!(
            "tol must be finite and positive, got {}",
            c.tol
        )));
    }
    match c.sm_method {
        SmoothingMethod::Supsmu => {
            if !(c.bass.is_finite() && (0.0..=10.0).contains(&c.bass)) {
                return Err(invalid(format!("bass must be in [0, 10], got {}", c.bass)));
            }
            if !(c.span.is_finite() && (0.0..=1.0).contains(&c.span)) {
                return Err(invalid(format!(
                    "span must be 0 (cross-validated) or in (0, 1], got {}",
                    c.span
                )));
            }
        }
        SmoothingMethod::Spline | SmoothingMethod::GcvSpline => {
            if n < 6 {
                return Err(invalid(format!(
                    "the PPR spline smoothers need at least 6 observations, got {n}"
                )));
            }
            if c.sm_method == SmoothingMethod::Spline {
                let bound = n.min(15) as f64;
                if !(c.df.is_finite() && c.df > 1.0 && c.df <= bound) {
                    return Err(invalid(format!(
                        "df must be above 1 and at most min(n, 15) = {bound}, got {}",
                        c.df
                    )));
                }
            } else if !(c.gcvpen.is_finite() && c.gcvpen >= 0.0) {
                return Err(invalid(format!(
                    "gcvpen must be finite and non-negative, got {}",
                    c.gcvpen
                )));
            }
        }
    }
    require_finite_columns(x)?;
    let k = y.iter().filter(|v| !v.is_finite()).count();
    if k > 0 {
        return Err(invalid(format!(
            "PPR: the response has {k} non-finite values (NaN or infinite); remove or impute \
             them first"
        )));
    }
    if let Some(wt) = weights {
        if wt.len() != n {
            return Err(invalid("Weights must have same length as y".to_string()));
        }
        if wt.iter().any(|&v| !(v.is_finite() && v >= 0.0)) {
            return Err(invalid(
                "PPR: weights must be finite and non-negative".to_string(),
            ));
        }
        if !(wt.iter().sum::<f64>() > 0.0) {
            return Err(invalid("PPR: the weights sum to zero".to_string()));
        }
    }
    let w = |i: usize| weights.map_or(1.0, |wt| wt[i]);
    let sw: f64 = (0..n).map(w).sum();
    let yb = (0..n).map(|i| w(i) * y[i]).sum::<f64>() / sw;
    if !((0..n).any(|i| w(i) > 0.0 && y[i] != yb)) {
        return Err(invalid(
            "PPR: the response is constant on the rows with positive weight".to_string(),
        ));
    }
    if x.axis_iter(Axis(1)).all(|c| c.iter().all(|&v| v == c[0])) {
        return Err(invalid("PPR: every predictor is constant".to_string()));
    }
    Ok(())
}

/// Options the chosen smoother does not use and that differ from their defaults.
fn unused_options(c: &PprConfig) -> Vec<String> {
    let mut out = Vec::new();
    let mut push = |name: &str, value: f64, applies: &str| {
        out.push(format!(
            "PPR: {name} = {value} has no effect with sm.method = {} (it applies to {applies} only)",
            c.sm_method
        ));
    };
    if c.sm_method != SmoothingMethod::Supsmu {
        if c.bass != 0.0 {
            push("bass", c.bass, "supsmu");
        }
        if c.span != 0.0 {
            push("span", c.span, "supsmu");
        }
    }
    if c.sm_method != SmoothingMethod::Spline && c.df != 5.0 {
        push("df", c.df, "spline");
    }
    if c.sm_method != SmoothingMethod::GcvSpline && c.gcvpen != 1.0 {
        push("gcvpen", c.gcvpen, "gcvspline");
    }
    out
}

/// Fit a projection pursuit regression model.
///
/// # Arguments
/// * `x` - Predictor matrix (n x p)
/// * `y` - Response vector (length n)
/// * `weights` - Optional case weights
/// * `config` - PPR configuration
///
/// # Errors
///
/// The message of [`ppr_with`]'s error.
pub fn ppr(
    x: ArrayView2<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    config: PprConfig,
) -> Result<PprResult, String> {
    ppr_with(x, y, weights, &config).map_err(|e| e.to_string())
}

/// Fit a projection pursuit regression model.
///
/// # Errors
///
/// `InvalidSpecification` when a parameter is out of range, when `y` (or `weights`) does not
/// match the rows of `x`, when there are fewer than 4 observations (6 for the spline smoothers),
/// when `x`, `y` or a weight is non-finite, a weight is negative or the weights sum to zero, when
/// `y` or every predictor is constant, and when the projections pool into fewer than three
/// groups. `Cancelled` once the installed computation budget expires (checked per term, per
/// search step and per backfitting pass).
pub fn ppr_with(
    x: ArrayView2<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    config: &PprConfig,
) -> EconResult<PprResult> {
    resource::checkpoint()?;
    validate(x, y, weights, config)?;
    let (n, p) = x.dim();
    let unit;
    let w: &[f64] = match weights {
        Some(wt) => wt,
        None => {
            unit = resource::try_vec(n, 1.0, "PPR weights")?;
            &unit
        }
    };
    let copied;
    let flat: &[f64] = match x.as_slice() {
        Some(s) => s,
        None => {
            let mut v = resource::try_with_capacity(
                resource::checked_len(&[n, p], "PPR design")?,
                "PPR design",
            )?;
            v.extend(x.iter().copied());
            copied = v;
            &copied
        }
    };
    let smart = Smart {
        x: flat,
        w,
        sw: w.iter().sum(),
        n,
        p,
        method: config.sm_method,
        span: config.span,
        bass: config.bass,
        df: config.df,
        gcvpen: config.gcvpen,
        conv: config.tol,
        maxit: config.max_iter,
        control: Control {
            cutmin: CUTMIN,
            mitone: config.max_iter,
        },
    };
    smart.fit(y, config)
}

/// The step-halving floor and the search length of one direction search.
#[derive(Clone, Copy)]
struct Control {
    cutmin: f64,
    mitone: usize,
}

/// The data and options of one fit.
struct Smart<'a> {
    /// n x p, row-major
    x: &'a [f64],
    w: &'a [f64],
    sw: f64,
    n: usize,
    p: usize,
    method: SmoothingMethod,
    span: f64,
    bass: f64,
    df: f64,
    gcvpen: f64,
    conv: f64,
    maxit: usize,
    control: Control,
}

/// Work space of the direction search.
struct Buffers {
    order: Vec<(f64, usize)>,
    xs: Vec<f64>,
    ys: Vec<f64>,
    ws: Vec<f64>,
    smo: Vec<f64>,
    yy: Vec<f64>,
    resid: Vec<f64>,
    deriv: Vec<f64>,
    deriv_sorted: Vec<f64>,
    step: Vec<f64>,
    trial: Vec<f64>,
    supsmu: supsmu::SupsmuWork,
    spline: Option<spline::SplineWork>,
}

/// The terms found so far: `a` is `terms x p`, `f` and `t` are `terms x n` (flat).
struct Model {
    a: Vec<f64>,
    b: Vec<f64>,
    f: Vec<f64>,
    t: Vec<f64>,
    /// `asr[0]`: the current RSS; `asr[1 + l]`: after backfitting `l + 1` terms
    asr: Vec<f64>,
    r: Vec<f64>,
    edf: Vec<f64>,
    trial_a: Vec<f64>,
    trial_f: Vec<f64>,
    trial_t: Vec<f64>,
    lm: usize,
    /// The final backfitting: passes, start and end RSS of the last pass
    last_fulfit: Option<(usize, f64, f64)>,
}

/// One term's fit, from [`Smart::onetrm`].
struct TermFit {
    b: f64,
    asr: f64,
    edf: f64,
}

impl Smart<'_> {
    fn fit(&self, y: &[f64], config: &PprConfig) -> EconResult<PprResult> {
        let (n, p, w, sw) = (self.n, self.p, self.w, self.sw);
        let (mu, ml, lf) = (config.nterms, config.max_terms, config.optlevel);
        let yb = {
            let mut s = 0.0;
            for j in 0..n {
                s += w[j] * y[j];
            }
            s / sw
        };
        let mut r: Vec<f64> = resource::try_vec(n, 0.0, "PPR residuals")?;
        let mut ss = 0.0;
        for j in 0..n {
            r[j] = y[j] - yb;
            ss += w[j] * r[j] * r[j];
        }
        let ys = (ss / sw).sqrt();
        if !(ys > 0.0) {
            return Err(invalid(
                "PPR: the response is constant on the rows with positive weight".to_string(),
            ));
        }
        let s = 1.0 / ys;
        r.iter_mut().for_each(|v| *v *= s);

        let term_rows = resource::checked_len(&[ml, n], "PPR ridge functions")?;
        let term_cols = resource::checked_len(&[ml, p], "PPR directions")?;
        let mut m = Model {
            a: resource::try_vec(term_cols, 0.0, "PPR directions")?,
            b: resource::try_vec(ml, 0.0, "PPR terms")?,
            f: resource::try_vec(term_rows, 0.0, "PPR ridge functions")?,
            t: resource::try_vec(term_rows, 0.0, "PPR ridge functions")?,
            asr: resource::try_vec(ml + 1, 0.0, "PPR terms")?,
            r,
            edf: resource::try_vec(ml, 0.0, "PPR terms")?,
            trial_a: resource::try_vec(p, 0.0, "PPR directions")?,
            trial_f: resource::try_vec(n, 0.0, "PPR ridge functions")?,
            trial_t: resource::try_vec(n, 0.0, "PPR ridge functions")?,
            lm: 0,
            last_fulfit: None,
        };
        let mut b = self.buffers()?;

        self.subfit(&mut m, &mut b, ml, lf)?;
        if lf > 0 {
            self.fulfit(&mut m, &mut b, lf, true)?;
            while m.lm > mu {
                resource::checkpoint()?;
                // the term with the smallest |b|, ties to the lowest index
                let mut drop = 0;
                for l in 1..m.lm {
                    if m.b[l].abs().total_cmp(&m.b[drop].abs()).is_lt() {
                        drop = l;
                    }
                }
                self.restandardise_residual(&mut m, y, yb, ys);
                let mut asr1 = 0.0;
                for j in 0..n {
                    m.r[j] += m.b[drop] * m.f[drop * n + j];
                    asr1 += w[j] * m.r[j] * m.r[j];
                }
                m.asr[0] = asr1 / sw;
                let last = m.lm - 1;
                if drop < last {
                    m.a.copy_within(last * p..(last + 1) * p, drop * p);
                    m.f.copy_within(last * n..(last + 1) * n, drop * n);
                    m.t.copy_within(last * n..(last + 1) * n, drop * n);
                    m.b[drop] = m.b[last];
                    m.edf[drop] = m.edf[last];
                }
                m.lm = last;
                self.fulfit(&mut m, &mut b, lf, true)?;
            }
        }
        self.result(y, config, m, yb, ys)
    }

    fn buffers(&self) -> EconResult<Buffers> {
        let (n, p) = (self.n, self.p);
        let col = || resource::try_vec(n, 0.0, "PPR work");
        Ok(Buffers {
            order: resource::try_vec(n, (0.0, 0), "PPR work")?,
            xs: col()?,
            ys: col()?,
            ws: col()?,
            smo: col()?,
            yy: col()?,
            resid: col()?,
            deriv: col()?,
            deriv_sorted: col()?,
            step: resource::try_vec(p, 0.0, "PPR work")?,
            trial: resource::try_vec(p, 0.0, "PPR work")?,
            supsmu: supsmu::SupsmuWork::new(n)?,
            spline: match self.method {
                SmoothingMethod::Supsmu => None,
                _ => Some(spline::SplineWork::new(n)?),
            },
        })
    }

    /// `r = (y - yb) / ys - sum_l b_l f_l`, free of what backfitting's updates accumulated.
    fn restandardise_residual(&self, m: &mut Model, y: &[f64], yb: f64, ys: f64) {
        let n = self.n;
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..m.lm {
                s += m.b[l] * m.f[l * n + j];
            }
            m.r[j] = (y[j] - yb) / ys - s;
        }
    }

    /// The forward pass: terms up to `ml`, stopping when a term no longer helps.
    fn subfit(&self, m: &mut Model, b: &mut Buffers, ml: usize, lf: usize) -> EconResult<()> {
        let (n, p) = (self.n, self.p);
        m.asr[0] = BIG;
        m.lm = 0;
        for _ in 0..ml {
            resource::checkpoint()?;
            let k = m.lm;
            m.lm += 1;
            let asrold = m.asr[0];
            m.b[k] = 1.0;
            let fit = self.onetrm(
                b,
                self.control,
                0,
                &m.r,
                1.0,
                &mut m.a[k * p..(k + 1) * p],
                &mut m.f[k * n..(k + 1) * n],
                &mut m.t[k * n..(k + 1) * n],
            )?;
            m.b[k] = fit.b;
            m.asr[0] = fit.asr;
            m.edf[k] = fit.edf;
            for j in 0..n {
                m.r[j] -= m.b[k] * m.f[k * n + j];
            }
            if m.lm == 1 {
                continue;
            }
            if lf > 0 {
                if m.lm == ml {
                    return Ok(());
                }
                self.fulfit(m, b, 1, false)?;
            }
            if m.asr[0] <= 0.0 || (asrold - m.asr[0]) / asrold < self.conv {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Backfitting. Level below 3: no step halving, and `lbf - 1` direction steps per term. A
    /// term keeps a refit only if it lowers the RSS at the start of the pass.
    fn fulfit(&self, m: &mut Model, b: &mut Buffers, lbf: usize, record: bool) -> EconResult<()> {
        let (n, p) = (self.n, self.p);
        let lm = m.lm;
        let mut asri = m.asr[0];
        let control = if lbf < 3 {
            Control {
                cutmin: 1.0,
                mitone: lbf - 1,
            }
        } else {
            self.control
        };
        let mut iter = 0;
        loop {
            resource::checkpoint()?;
            let asrold = asri;
            iter += 1;
            for lp in 0..lm {
                let bt = m.b[lp];
                m.trial_a.copy_from_slice(&m.a[lp * p..(lp + 1) * p]);
                for j in 0..n {
                    m.r[j] += bt * m.f[lp * n + j];
                }
                let fit = self.onetrm(
                    b,
                    control,
                    1,
                    &m.r,
                    bt,
                    &mut m.trial_a,
                    &mut m.trial_f,
                    &mut m.trial_t,
                )?;
                asri = fit.asr;
                if asri < asrold {
                    m.b[lp] = fit.b;
                    m.edf[lp] = fit.edf;
                    m.a[lp * p..(lp + 1) * p].copy_from_slice(&m.trial_a);
                    m.f[lp * n..(lp + 1) * n].copy_from_slice(&m.trial_f);
                    m.t[lp * n..(lp + 1) * n].copy_from_slice(&m.trial_t);
                } else {
                    asri = asrold;
                }
                for j in 0..n {
                    m.r[j] -= m.b[lp] * m.f[lp * n + j];
                }
            }
            if !(iter <= self.maxit && asri > 0.0 && (asrold - asri) / asrold >= self.conv) {
                if record {
                    m.last_fulfit = Some((iter, asrold, asri));
                }
                break;
            }
        }
        if record {
            m.asr[lm] = asri;
            m.asr[0] = asri;
        }
        Ok(())
    }

    /// One term against `y`: the direction search on `b y`, then `b` by least squares on the
    /// standardised ridge function, and the term's RSS.
    fn onetrm(
        &self,
        buf: &mut Buffers,
        control: Control,
        start: usize,
        y: &[f64],
        b: f64,
        a: &mut [f64],
        f: &mut [f64],
        t: &mut [f64],
    ) -> EconResult<TermFit> {
        let (n, w, sw) = (self.n, self.w, self.sw);
        let mut yy = std::mem::take(&mut buf.yy);
        for j in 0..n {
            yy[j] = b * y[j];
        }
        let searched = self.oneone(buf, control, start, &yy, a, f, t);
        buf.yy = yy;
        let (_, edf) = searched?;
        let mut s = 0.0;
        for j in 0..n {
            s += w[j] * y[j] * f[j];
        }
        let b = s / sw;
        let mut s = 0.0;
        for j in 0..n {
            let e = y[j] - b * f[j];
            s += w[j] * e * e;
        }
        Ok(TermFit {
            b,
            asr: s / sw,
            edf,
        })
    }

    /// The Gauss-Newton search for one direction with step halving. `start == 0` starts from the
    /// linear fit, otherwise from `a`. Returns the search's RSS and the equivalent df of the
    /// accepted smooth; `a`, `f` and `t` hold the accepted direction, the standardised ridge
    /// function and the projections, by row.
    fn oneone(
        &self,
        buf: &mut Buffers,
        control: Control,
        start: usize,
        y: &[f64],
        a: &mut [f64],
        f: &mut [f64],
        t: &mut [f64],
    ) -> EconResult<(f64, f64)> {
        let (n, p, w, sw) = (self.n, self.p, self.w, self.sw);
        if start == 0 {
            buf.deriv.fill(1.0);
            a.copy_from_slice(&pprdir(self.x, p, w, sw, y, &buf.deriv)?);
        }
        let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            let s = 1.0 / norm;
            a.iter_mut().for_each(|v| *v *= s);
        } else {
            // y is orthogonal to every predictor: any direction is as good a start
            a.fill(0.0);
            a[0] = 1.0;
        }
        buf.step.fill(0.0);
        f.fill(0.0);
        t.fill(0.0);
        let mut iter = 0;
        let mut asr = BIG;
        let mut cut = 1.0;
        let mut edf = 0.0;
        loop {
            iter += 1;
            let asrold = asr;
            let mut accepted = None;
            loop {
                resource::checkpoint()?;
                for i in 0..p {
                    buf.trial[i] = a[i] + buf.step[i];
                }
                let norm = buf.trial.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 && norm.is_finite() {
                    let s = 1.0 / norm;
                    buf.trial.iter_mut().for_each(|v| *v *= s);
                    let df = self.smooth_on(buf, y)?;
                    let mut s = 0.0;
                    for j in 0..n {
                        let e = buf.ys[j] - buf.smo[j];
                        s += buf.ws[j] * e * e;
                    }
                    let s = s / sw;
                    if s < asr {
                        accepted = Some((s, df));
                        break;
                    }
                }
                cut *= 0.5;
                if cut < control.cutmin {
                    break;
                }
                buf.step.iter_mut().for_each(|v| *v *= cut);
            }
            let Some((s, df)) = accepted else {
                break;
            };
            asr = s;
            edf = df;
            cut = 1.0;
            a.copy_from_slice(&buf.trial);
            for (pos, &(_, row)) in buf.order.iter().enumerate() {
                t[row] = buf.xs[pos];
                f[row] = buf.smo[pos];
            }
            if asr <= 0.0 || (asrold - asr) / asrold < self.conv {
                break;
            }
            if iter > control.mitone || p <= 1 {
                break;
            }
            pprder(&buf.xs, &buf.smo, &buf.ws, FDEL, &mut buf.deriv_sorted)?;
            for j in 0..n {
                buf.resid[j] = y[j] - f[j];
            }
            for (pos, &(_, row)) in buf.order.iter().enumerate() {
                buf.deriv[row] = buf.deriv_sorted[pos];
            }
            buf.step
                .copy_from_slice(&pprdir(self.x, p, w, sw, &buf.resid, &buf.deriv)?);
        }
        let mut s = 0.0;
        for j in 0..n {
            s += w[j] * f[j];
        }
        s /= sw;
        let mut v = 0.0;
        for j in 0..n {
            f[j] -= s;
            v += w[j] * f[j] * f[j];
        }
        if v > 0.0 {
            let v = 1.0 / (v / sw).sqrt();
            f.iter_mut().for_each(|e| *e *= v);
        }
        Ok((asr, edf))
    }

    /// The smooth of `y` on the projections on `buf.trial`, in sorted order; ties keep row order
    /// (the projections are finite: the data were validated before the fit). Returns the
    /// equivalent df.
    fn smooth_on(&self, buf: &mut Buffers, y: &[f64]) -> EconResult<f64> {
        let (n, p) = (self.n, self.p);
        const SML: f64 = 1.0 / BIG;
        for (j, row) in self.x.chunks_exact(p).enumerate() {
            let mut s = 0.0;
            for (a, v) in buf.trial.iter().zip(row) {
                s += a * v;
            }
            buf.order[j] = (s, j);
        }
        buf.order
            .sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (pos, &(s, row)) in buf.order.iter().enumerate() {
            buf.xs[pos] = s;
            buf.ys[pos] = y[row];
            buf.ws[pos] = self.w[row].max(SML);
        }
        match self.method {
            SmoothingMethod::Supsmu => {
                supsmu::supsmu_sorted(
                    &buf.xs,
                    &buf.ys,
                    &buf.ws,
                    self.span,
                    self.bass,
                    &mut buf.smo,
                    &mut buf.supsmu,
                );
                Ok(0.0)
            }
            method => {
                if !(buf.xs[n - 1] > buf.xs[0]) {
                    let (mut sy, mut sw) = (0.0, 0.0);
                    for j in 0..n {
                        sy += buf.ws[j] * buf.ys[j];
                        sw += buf.ws[j];
                    }
                    buf.smo.fill(sy / sw);
                    return Ok(0.0);
                }
                let Some(work) = buf.spline.as_mut() else {
                    return Err(EconError::Computation(
                        "PPR: the spline smoother has no work space".to_string(),
                    ));
                };
                spline::ppr_spline(
                    &buf.xs,
                    &buf.ys,
                    &buf.ws,
                    method,
                    self.df,
                    self.gcvpen,
                    &mut buf.smo,
                    work,
                )
            }
        }
    }

    fn result(
        &self,
        y: &[f64],
        config: &PprConfig,
        m: Model,
        yb: f64,
        ys: f64,
    ) -> EconResult<PprResult> {
        let (n, p) = (self.n, self.p);
        let lm = m.lm;
        let mut warnings = unused_options(config);
        if config.optlevel == 0 && lm > config.nterms {
            warnings.push(format!(
                "PPR: optlevel 0 does no backfitting or backward elimination, so the model keeps \
                 all {lm} terms of the forward pass (nterms = {} was requested; R's ppr reports \
                 only the first nterms directions but predicts with all {lm})",
                config.nterms
            ));
        }
        if lm < config.nterms {
            warnings.push(format!(
                "PPR: the forward pass stopped after {lm} term(s) because another term reduced \
                 the weighted RSS by less than tol = {} (relative); nterms = {} was requested",
                config.tol, config.nterms
            ));
        }
        let convergence = m.last_fulfit.map(|(iter, asrold, asri)| {
            let change = if asrold > 0.0 {
                (asrold - asri) / asrold
            } else {
                0.0
            };
            Convergence::new(
                !(iter > self.maxit && asri > 0.0 && change >= self.conv),
                iter,
                self.maxit + 1,
                "relative decrease in the weighted RSS",
                change,
                self.conv,
            )
        });
        if let Some(note) = convergence.as_ref().and_then(|c| c.note("PPR backfitting")) {
            warnings.push(note);
        }
        let scale = n as f64 * ys * ys;
        let gofn: Vec<f64> = (0..config.max_terms)
            .map(|l| m.asr[1 + l] * scale)
            .collect();
        let mut fitted = resource::try_vec(n, 0.0, "PPR fitted values")?;
        let mut residuals = resource::try_vec(n, 0.0, "PPR residuals")?;
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..lm {
                s += m.b[l] * m.f[l * n + j];
            }
            fitted[j] = ys * s + yb;
            residuals[j] = y[j] - fitted[j];
        }
        let mut alpha = Vec::with_capacity(lm);
        let mut ridge_projections = Vec::with_capacity(lm);
        let mut ridge_values = Vec::with_capacity(lm);
        let mut order: Vec<usize> = resource::try_vec(n, 0, "PPR ridge functions")?;
        for l in 0..lm {
            alpha.push(m.a[l * p..(l + 1) * p].to_vec());
            let (t, f) = (&m.t[l * n..(l + 1) * n], &m.f[l * n..(l + 1) * n]);
            order.iter_mut().enumerate().for_each(|(i, o)| *o = i);
            order.sort_by(|&i, &j| t[i].total_cmp(&t[j]));
            let mut ts = resource::try_with_capacity(n, "PPR ridge functions")?;
            let mut fs = resource::try_with_capacity(n, "PPR ridge functions")?;
            for &i in &order {
                ts.push(t[i]);
                fs.push(f[i]);
            }
            ridge_projections.push(ts);
            ridge_values.push(fs);
        }
        Ok(PprResult {
            nterms: lm,
            alpha,
            beta: m.b[..lm].iter().map(|b| ys * b).collect(),
            fitted,
            residuals,
            gofn,
            n,
            p,
            sm_method: config.sm_method,
            max_terms: config.max_terms,
            optlevel: config.optlevel,
            gof: m.asr[0] * scale,
            edf: m.edf[..lm].to_vec(),
            y_mean: yb,
            y_scale: ys,
            beta_standardized: m.b[..lm].to_vec(),
            ridge_projections,
            ridge_values,
            bass: config.bass,
            span: config.span,
            df: config.df,
            gcvpen: config.gcvpen,
            convergence,
            warnings,
        })
    }
}

/// The Gauss-Newton step on rows `x` (n x p, row-major): least squares of `r` on `d x_j` centred
/// by weighted means, with weights `w`, by [`ppconj`].
fn pprdir(x: &[f64], p: usize, w: &[f64], sw: f64, r: &[f64], d: &[f64]) -> EconResult<Vec<f64>> {
    resource::checkpoint()?;
    let packed = resource::checked_len(&[p, p + 1], "PPR direction step")? / 2;
    let mut e = resource::try_vec(p, 0.0, "PPR direction step")?;
    let mut z = resource::try_vec(p, 0.0, "PPR direction step")?;
    let mut c = resource::try_vec(p, 0.0, "PPR direction step")?;
    let mut g = resource::try_vec(packed, 0.0, "PPR direction step")?;
    for (l, row) in x.chunks_exact(p).enumerate() {
        let wd = w[l] * d[l];
        for (ei, v) in e.iter_mut().zip(row) {
            *ei += wd * v;
        }
    }
    e.iter_mut().for_each(|v| *v /= sw);
    for (l, row) in x.chunks_exact(p).enumerate() {
        if l % 256 == 0 {
            resource::checkpoint()?;
        }
        let wl = w[l];
        let dl = d[l];
        for j in 0..p {
            z[j] = dl * row[j] - e[j];
        }
        let wr = wl * r[l];
        let mut k = 0;
        for j in 0..p {
            let zj = z[j];
            c[j] += wr * zj;
            let wz = wl * zj;
            for &zi in &z[..=j] {
                g[k] += wz * zi;
                k += 1;
            }
        }
    }
    g.iter_mut().for_each(|v| *v /= sw);
    c.iter_mut().for_each(|v| *v /= sw);
    ppconj(&g, &c, p)
}

/// `out = G v` for `G` a packed upper triangle by columns.
fn packed_mul(g: &[f64], v: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    let mut k = 0;
    for j in 0..v.len() {
        for i in 0..j {
            out[i] += g[k] * v[j];
            out[j] += g[k] * v[i];
            k += 1;
        }
        out[j] += g[k] * v[j];
        k += 1;
    }
}

/// Conjugate gradients on `G x = c`, restarted `MITCJ` times.
fn ppconj(g: &[f64], c: &[f64], p: usize) -> EconResult<Vec<f64>> {
    let col = || resource::try_vec(p, 0.0, "PPR conjugate gradients");
    let (mut x, mut res, mut dir, mut gd, mut prev) = (col()?, col()?, col()?, col()?, col()?);
    for _ in 0..MITCJ {
        prev.copy_from_slice(&x);
        packed_mul(g, &x, &mut gd);
        let mut h = 0.0;
        for i in 0..p {
            res[i] = gd[i] - c[i];
            h += res[i] * res[i];
        }
        if !(h > 0.0) {
            break;
        }
        let mut beta = 0.0;
        for _ in 0..p {
            resource::checkpoint()?;
            for i in 0..p {
                dir[i] = beta * dir[i] - res[i];
            }
            packed_mul(g, &dir, &mut gd);
            let t: f64 = gd.iter().zip(&dir).map(|(a, b)| a * b).sum();
            if !(t > 0.0) {
                // no curvature left along dir (G is singular here): x is as good as it gets
                break;
            }
            let alpha = h / t;
            let mut s = 0.0;
            for i in 0..p {
                x[i] += alpha * dir[i];
                res[i] += alpha * gd[i];
                s += res[i] * res[i];
            }
            if !(s > 0.0) {
                break;
            }
            beta = s / h;
            h = s;
        }
        let change = x
            .iter()
            .zip(&prev)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if !(change >= CJEPS) {
            break;
        }
    }
    Ok(x)
}

/// Merge runs of `x` closer than `del` (ppr.f `pool`): each run gets the weighted means of its `x`
/// and `y` and the sum of its weights. Runs of equal `x` hold their sum in every element.
fn pool(x: &mut [f64], y: &mut [f64], w: &mut [f64], del: f64) {
    // `first` and `last` are 0-based and inclusive.
    let merge = |x: &mut [f64], y: &mut [f64], w: &mut [f64], first: usize, last: usize| {
        let pw = w[first] + w[last];
        let px = (x[first] * w[first] + x[last] * w[last]) / pw;
        let py = (y[first] * w[first] + y[last] * w[last]) / pw;
        x[first..=last].fill(px);
        y[first..=last].fill(py);
        w[first..=last].fill(pw);
    };
    let n = x.len();
    let mut next = 0;
    while next < n {
        let mut first = next;
        let mut last = first;
        while last + 1 < n && x[first] == x[last + 1] {
            last += 1;
        }
        if last + 1 < n && x[last + 1] - x[last] < del {
            // the run to the right, unless it is closer to its own right neighbour
            let right = last + 1;
            let mut right_end = right;
            while right_end + 1 < n && x[right_end + 1] == x[right] {
                right_end += 1;
            }
            if right_end + 1 < n && x[right_end + 1] - x[right_end] < x[right] - x[last] {
                next = last + 1;
                continue;
            }
            last = right_end;
            merge(x, y, w, first, last);
        }
        while first > 0 && x[first] - x[first - 1] < del {
            let left_end = first - 1;
            let mut left = left_end;
            while left > 0 && x[left - 1] == x[left_end] {
                left -= 1;
            }
            first = left;
            merge(x, y, w, first, last);
        }
        next = last + 1;
    }
}

/// The derivative of the smooth `s` on sorted `x` (ppr.f `pprder`): central differences between
/// the groups of `x` pooled at `fdel` of the interquartile range, one-sided at the two ends.
fn pprder(x: &[f64], s: &[f64], w: &[f64], fdel: f64, d: &mut [f64]) -> EconResult<()> {
    let n = x.len();
    if !(x[n - 1] > x[0]) {
        d.fill(0.0);
        return Ok(());
    }
    let mut i = n / 4;
    let mut j = 3 * i;
    let mut scale = x[j - 1] - x[i - 1];
    while !(scale > 0.0) {
        if j < n {
            j += 1;
        }
        if i > 1 {
            i -= 1;
        }
        scale = x[j - 1] - x[i - 1];
    }
    let del = fdel * scale * 2.0;
    let mut sx = resource::try_vec(n, 0.0, "PPR derivative")?;
    let mut sy = resource::try_vec(n, 0.0, "PPR derivative")?;
    let mut sw = resource::try_vec(n, 0.0, "PPR derivative")?;
    sx.copy_from_slice(x);
    sy.copy_from_slice(s);
    sw.copy_from_slice(w);
    pool(&mut sx, &mut sy, &mut sw, del);
    // The end (exclusive) of the group starting at `start`.
    let group_end = |start: usize| {
        let mut e = start + 1;
        while e < n && sx[e] == sx[start] {
            e += 1;
        }
        e
    };
    let too_few = || {
        invalid(
            "PPR: the projections pool into fewer than three distinct values, so the \
             ridge-function derivative is undefined (R stops with 'br is too large'); use fewer \
             terms or more distinct data"
                .to_string(),
        )
    };
    let (mut bl, mut bc) = (0, group_end(0));
    if bc >= n {
        return Err(too_few());
    }
    let mut ec = group_end(bc);
    if ec >= n {
        return Err(too_few());
    }
    let slope = |a: usize, b: usize| (sy[b] - sy[a]) / (sx[b] - sx[a]);
    d[bl..bc].fill(slope(bl, bc));
    loop {
        let br = ec;
        let er = group_end(br);
        d[bc..ec].fill(slope(bl, br));
        if er == n {
            d[br..er].fill(slope(bc, br));
            return Ok(());
        }
        bl = bc;
        bc = br;
        ec = er;
    }
}

/// Run PPR (convenience wrapper).
pub fn run_ppr(
    x: ArrayView2<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    nterms: usize,
    max_terms: Option<usize>,
    sm_method: Option<SmoothingMethod>,
    bass: Option<f64>,
) -> Result<PprResult, String> {
    let config = PprConfig {
        nterms,
        max_terms: max_terms.unwrap_or(nterms),
        sm_method: sm_method.unwrap_or(SmoothingMethod::Supsmu),
        bass: bass.unwrap_or(0.0),
        ..Default::default()
    };
    ppr(x, y, weights, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{Budget, CancelToken, with_budget};
    use ndarray::Array2;
    use p2a_validation::{Check, Tol};

    fn cancelled_budget() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    fn ppr_data(n: usize) -> (Array2<f64>, Vec<f64>) {
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * (3 + 4 * j)) % 17) as f64 / 4.0);
        let y = (0..n)
            .map(|i| x[[i, 0]].sin() + 0.5 * x[[i, 1]] + ((i * 7) % 5) as f64 * 0.05)
            .collect();
        (x, y)
    }

    #[test]
    fn test_ppr_terms_are_bounded_by_observations() {
        let (x, y) = ppr_data(20);
        for (nterms, max_terms) in [(1usize, 21usize), (21, 1), (usize::MAX, usize::MAX)] {
            let cfg = PprConfig {
                nterms,
                max_terms,
                ..Default::default()
            };
            let err = ppr(x.view(), &y, None, cfg).unwrap_err();
            assert!(
                err.contains("must not exceed the number of observations (20)"),
                "{nterms}/{max_terms}: {err}"
            );
        }
    }

    #[test]
    fn test_ppr_rejects_bad_values_and_honours_cancellation() {
        let (x, mut y) = ppr_data(20);
        let w: Vec<f64> = (0..20).map(|i| if i == 4 { -1.0 } else { 1.0 }).collect();
        assert!(
            ppr(x.view(), &y, Some(&w), PprConfig::default())
                .unwrap_err()
                .contains("weights")
        );
        let err = with_budget(cancelled_budget(), || {
            ppr(x.view(), &y, None, PprConfig::default())
        })
        .unwrap_err();
        assert!(err.contains("cancelled"), "{err}");
        y[2] = f64::NAN;
        assert!(
            ppr(x.view(), &y, None, PprConfig::default())
                .unwrap_err()
                .contains("non-finite")
        );
    }

    #[test]
    fn test_ppr_basic() {
        // Simple linear relationship
        let x_data: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let x = Array2::from_shape_vec((15, 2), x_data).unwrap();

        let y: Vec<f64> = (0..15)
            .map(|i| 2.0 * i as f64 + 3.0 * (i + 15) as f64)
            .collect();

        let config = PprConfig {
            nterms: 1,
            max_terms: 1,
            ..Default::default()
        };

        let result = ppr(x.view(), &y, None, config).unwrap();

        assert_eq!(result.nterms, 1);
        assert_eq!(result.n, 15);
        assert_eq!(result.p, 2);
        assert_eq!(result.fitted.len(), 15);
        assert_eq!(result.residuals.len(), 15);
    }

    #[test]
    fn test_ppr_multiple_terms() {
        let n = 50;
        let p = 3;
        let x_data: Vec<f64> = (0..n * p).map(|i| (i as f64) / (n * p) as f64).collect();
        let x = Array2::from_shape_vec((n, p), x_data).unwrap();

        let y: Vec<f64> = (0..n)
            .map(|i| {
                let xi: f64 = (0..p).map(|j| x[[i, j]]).sum();
                xi.sin() + 0.1 * (i as f64 % 5.0)
            })
            .collect();

        let config = PprConfig {
            nterms: 2,
            max_terms: 3,
            ..Default::default()
        };

        let result = ppr(x.view(), &y, None, config).unwrap();

        // Should have at most 2 terms
        assert!(result.nterms <= 2);
        assert!(!result.gofn.is_empty());
    }

    #[test]
    fn test_ppr_with_weights() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Vec<f64> = (0..10).map(|i| i as f64 * 2.0).collect();
        let weights: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { 2.0 }).collect();

        let config = PprConfig::default();
        let result = ppr(x.view(), &y, Some(&weights), config).unwrap();

        assert_eq!(result.n, 10);
    }

    #[test]
    fn test_ppr_predict() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Vec<f64> = (0..10).map(|i| i as f64).collect();

        let config = PprConfig::default();
        let result = ppr(x.view(), &y, None, config).unwrap();

        // Predict on new data
        let x_new = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let predictions = result.predict(&x_new);

        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_ppr_length_mismatch() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let y: Vec<f64> = (0..5).map(|i| i as f64).collect(); // Wrong length

        let config = PprConfig::default();
        let result = ppr(x.view(), &y, None, config);

        assert!(result.is_err());
    }

    mod validate {
        use super::super::*;
        use crate::data::columns::{NaPolicy, numeric_column};
        use ndarray::Array2;
        use p2a_validation::{Case, Check, Tol};

        pub(super) fn ppr_case() -> (Case, Array2<f64>, Vec<f64>, Vec<f64>) {
            let (case, ds) = crate::test_support::load_case("ppr_methods");
            let col = |n: &str| numeric_column(ds.df(), n, NaPolicy::Fail).unwrap().0;
            let (x1, x2, x3) = (col("x1"), col("x2"), col("x3"));
            let x = Array2::from_shape_fn((x1.len(), 3), |(i, j)| [x1[i], x2[i], x3[i]][j]);
            (case, x, col("y"), col("w"))
        }
        /// nterms = 2, max.terms = 3 (every run of the case except supsmu_p1)
        pub(super) fn cfg() -> PprConfig {
            PprConfig {
                nterms: 2,
                max_terms: 3,
                ..Default::default()
            }
        }
        /// The super-smoother runs: the same search at the same stopping rules, with projections
        /// that have no ties, so only rounding separates them from R.
        pub(super) const SUPSMU_TOL: Tol = Tol::new(1e-8, 1e-10);
        /// The spline runs: R's penalty (0.333 for 1/3) and last-evaluation fit steer its search,
        /// so these are R-compared at about 3x the measured gap, not to digits (validation/ml/ppr.md).
        pub(super) const SPLINE_TOL: Tol = Tol::new(0.03, 0.02);

        pub(super) fn check(case: &Case, tag: &str, r: &PprResult, tol: Tol) {
            let alpha: Vec<f64> = r.alpha.iter().flatten().copied().collect(); // = as.numeric(f$alpha)
            let rss: f64 = r.residuals.iter().map(|e| e * e).sum();
            let mut c = Check::new(case);
            c.vector(&format!("{tag}_alpha"), &alpha, tol)
                .vector(&format!("{tag}_beta"), &r.beta, tol)
                .vector(&format!("{tag}_gofn"), &r.gofn, tol)
                .scalar(&format!("{tag}_gof"), r.gof, tol)
                .vector(&format!("{tag}_fitted"), &r.fitted, tol)
                .scalar(&format!("{tag}_rss"), rss, tol)
                .scalar(&format!("{tag}_yb"), r.y_mean, Tol::coef(r.n))
                .scalar(&format!("{tag}_ys"), r.y_scale, Tol::coef(r.n));
            let edf_key = format!("{tag}_edf");
            if case.keys().any(|k| k == edf_key) {
                c.vector(&edf_key, &r.edf, tol);
            }
            c.finish();
        }
        pub(super) fn newx(case: &Case) -> Array2<f64> {
            let v = case.vector("predict_newx"); // 7 x 3, column-major
            Array2::from_shape_fn((7, 3), |(i, j)| v[j * 7 + i])
        }
        pub(super) fn run(tag: &str, f: impl Fn(&mut PprConfig), tol: Tol) -> (Case, PprResult) {
            let (case, x, y, _) = ppr_case();
            let mut c = cfg();
            f(&mut c);
            let r = ppr_with(x.view(), &y, None, &c).unwrap();
            check(&case, tag, &r, tol);
            (case, r)
        }
    }

    /// `ppr_methods` supsmu_b0: the default super smoother (cross-validated span). Before: a
    /// running mean with alpha replaced by the gradient (ML-34) and linear fitted values (ML-32).
    #[test]
    fn test_validate_ppr_supsmu_b0() {
        let (case, x, _, _) = validate::ppr_case();
        let (_, r) = validate::run("supsmu_b0", |_| {}, validate::SUPSMU_TOL);
        assert_eq!((r.nterms, r.max_terms, r.optlevel), (2, 3, 2));
        let nx = validate::newx(&case);
        let rows: Vec<Vec<f64>> = nx.outer_iter().map(|r| r.to_vec()).collect();
        let pred = r.try_predict(nx.view()).unwrap();
        let mut c = Check::new(&case);
        c.vector("supsmu_b0_predict", &pred, validate::SUPSMU_TOL);
        c.finish();
        assert_eq!(r.predict(&rows), pred);
        assert_eq!(x.nrows(), r.n);
    }
    #[test]
    fn test_validate_ppr_supsmu_b5() {
        validate::run("supsmu_b5", |c| c.bass = 5.0, validate::SUPSMU_TOL);
    }
    #[test]
    fn test_validate_ppr_supsmu_span03() {
        validate::run("supsmu_span03", |c| c.span = 0.3, validate::SUPSMU_TOL);
    }
    #[test]
    fn test_validate_ppr_spline_df5() {
        let (case, r) = validate::run(
            "spline_df5",
            |c| {
                c.sm_method = SmoothingMethod::Spline;
                c.df = 5.0;
            },
            validate::SPLINE_TOL,
        );
        let pred = r.try_predict(validate::newx(&case).view()).unwrap();
        let mut c = Check::new(&case);
        c.vector("spline_df5_predict", &pred, validate::SPLINE_TOL);
        c.finish();
    }
    #[test]
    fn test_validate_ppr_gcvspline() {
        validate::run(
            "gcvspline",
            |c| c.sm_method = SmoothingMethod::GcvSpline,
            validate::SPLINE_TOL,
        );
    }
    #[test]
    fn test_validate_ppr_gcvspline_pen2() {
        validate::run(
            "gcvspline_pen2",
            |c| {
                c.sm_method = SmoothingMethod::GcvSpline;
                c.gcvpen = 2.0;
            },
            validate::SPLINE_TOL,
        );
    }
    /// optlevel 1 and 3 as R; optlevel 0 keeps the 3 forward terms and gofn stays 0 as R's.
    #[test]
    fn test_validate_ppr_optlevels() {
        let (case, x, y, _) = validate::ppr_case();
        for (ol, tag) in [(1, "supsmu_ol1"), (3, "supsmu_ol3"), (0, "supsmu_ol0")] {
            let mut c = validate::cfg();
            c.optlevel = ol;
            let r = ppr_with(x.view(), &y, None, &c).unwrap();
            validate::check(&case, tag, &r, validate::SUPSMU_TOL);
            if ol == 0 {
                assert_eq!(r.nterms, 3);
                assert!(r.convergence.is_none());
                assert!(
                    r.warnings
                        .iter()
                        .any(|w| w.contains("optlevel 0") && w.contains("3 terms")),
                    "{:?}",
                    r.warnings
                );
            }
        }
    }
    #[test]
    fn test_validate_ppr_weighted() {
        let (case, x, y, w) = validate::ppr_case();
        let r = ppr_with(x.view(), &y, Some(&w), &validate::cfg()).unwrap();
        validate::check(&case, "supsmu_weighted", &r, validate::SUPSMU_TOL);
    }
    /// p = 1: the direction is +-1 and the search stops after one smooth.
    #[test]
    fn test_validate_ppr_one_predictor() {
        let (case, x, y, _) = validate::ppr_case();
        let x1 = x.slice(ndarray::s![.., 0..1]).to_owned();
        let r = ppr_with(x1.view(), &y, None, &PprConfig::default()).unwrap();
        validate::check(&case, "supsmu_p1", &r, validate::SUPSMU_TOL);
    }
    /// ML-32: the returned model is the smoother model: predict on the training rows reproduces
    /// fitted exactly, residuals = y - fitted, alpha has unit norm, and gof, the RSS the search
    /// tracked, is the weighted RSS of the fitted values or just above it (a pass whose last term
    /// keeps its old fit goes back to the RSS at the start of the pass, as in ppr.f).
    #[test]
    fn test_ppr_fitted_equals_predict_on_training_rows() {
        let (_, x, y, wt) = validate::ppr_case();
        for (m, weights) in [
            (SmoothingMethod::Supsmu, None),
            (SmoothingMethod::Spline, None),
            (SmoothingMethod::Supsmu, Some(&wt[..])),
        ] {
            let mut c = validate::cfg();
            c.sm_method = m;
            let r = ppr_with(x.view(), &y, weights, &c).unwrap();
            assert_eq!(r.try_predict(x.view()).unwrap(), r.fitted);
            for i in 0..y.len() {
                assert_eq!(r.residuals[i], y[i] - r.fitted[i]);
            }
            assert_eq!(r.ridge_projections.len(), r.nterms);
            assert!(
                r.ridge_projections
                    .iter()
                    .all(|t| t.windows(2).all(|w| w[0] <= w[1]))
            );
            for a in &r.alpha {
                let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
                assert!((norm - 1.0).abs() <= 1e-12, "{norm}");
            }
            let w = |i: usize| weights.map_or(1.0, |v: &[f64]| v[i]);
            let sw: f64 = (0..y.len()).map(w).sum();
            let wrss: f64 = (0..y.len()).map(|i| w(i) * r.residuals[i].powi(2)).sum();
            let want = wrss * y.len() as f64 / sw;
            assert!(
                r.gof >= want * (1.0 - 1e-10) && r.gof <= want * 1.01,
                "{} {want}",
                r.gof
            );
        }
    }
    /// pppred's rule, on a small table: exact, interpolated, constant beyond the range.
    #[test]
    fn test_ppr_ridge_value_interpolates_and_extrapolates_flat() {
        let (t, f) = ([0.0, 1.0, 1.0, 3.0], [2.0, 4.0, 4.0, 8.0]);
        assert_eq!(ridge_value(&t, &f, -5.0), 2.0);
        assert_eq!(ridge_value(&t, &f, 0.0), 2.0);
        assert_eq!(ridge_value(&t, &f, 0.5), 3.0);
        assert_eq!(ridge_value(&t, &f, 1.0), 4.0);
        assert_eq!(ridge_value(&t, &f, 2.0), 6.0);
        assert_eq!(ridge_value(&t, &f, 3.0), 8.0);
        assert_eq!(ridge_value(&t, &f, 9.0), 8.0);
    }
    /// Hand-computed from ppr.f's rule (R exposes no pprder).
    #[test]
    fn test_ppr_pprder_pools_and_differentiates() {
        let x = [0.0, 0.5, 0.51, 1.2, 2.0, 2.0, 3.5, 5.0];
        let s = [0.1, 0.4, 0.45, 0.9, 1.3, 1.35, 2.2, 2.0];
        let w = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 1.0];
        let mut d = [0.0; 8];
        pprder(&x, &s, &w, FDEL, &mut d).unwrap();
        let want = [
            0.65789473684210531,
            0.66666666666666674,
            0.66666666666666674,
            0.5803571428571429,
            0.565217391304348,
            0.565217391304348,
            0.23333333333333331,
            -0.13333333333333344,
        ];
        for (a, b) in d.iter().zip(want) {
            assert!((a - b).abs() <= 1e-14 * b.abs().max(1.0), "{d:?}");
        }
        let (mut px, mut py, mut pw) = (x, s, w);
        pool(&mut px, &mut py, &mut pw, FDEL * 2.0 * (2.0 - 0.51));
        assert!((px[1] - 0.50666666666666671).abs() < 1e-15, "{px:?}");
        assert!((py[2] - 0.43333333333333335).abs() < 1e-15, "{py:?}");
        assert_eq!(pw[1], 3.0);
        let x2 = [0.0, 0.001, 0.002, 0.003, 1.0, 1.001, 1.002, 1.003];
        let err = pprder(&x2, &[0.1; 8], &[1.0; 8], FDEL, &mut d)
            .unwrap_err()
            .to_string();
        assert!(err.contains("br is too large"), "{err}");
    }
    #[test]
    fn test_ppr_pprdir_solves_gauss_newton_step() {
        let x: Vec<f64> = (0..10)
            .flat_map(|j| (0..3).map(move |i| ((3 * (i + 1) + 7 * (j + 1)) % 11) as f64 / 10.0))
            .collect();
        let r: Vec<f64> = (1..=10).map(|j| (j as f64).sin()).collect();
        let d: Vec<f64> = (1..=10).map(|j| (j as f64).cos()).collect();
        let w = [1.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 3.0, 1.0, 1.0];
        let e = pprdir(&x, 3, &w, 12.5, &r, &d).unwrap();
        let want = [
            1.245593551338366,
            -0.76430883039952446,
            -0.12125488790032093,
        ];
        for (a, b) in e.iter().zip(want) {
            assert!((a - b).abs() <= 1e-12 * b.abs(), "{e:?}");
        }
    }
    /// Design section 3.2: options R distinguishes give different fits (before: bit-identical).
    #[test]
    fn test_ppr_options_change_the_fit() {
        let (_, x, y, _) = validate::ppr_case();
        let rss = |f: &dyn Fn(&mut PprConfig)| {
            let mut c = validate::cfg();
            f(&mut c);
            ppr_with(x.view(), &y, None, &c)
                .unwrap()
                .residuals
                .iter()
                .map(|e| e * e)
                .sum::<f64>()
        };
        let base = rss(&|_| {});
        for (what, v) in [
            ("bass", rss(&|c| c.bass = 5.0)),
            ("span", rss(&|c| c.span = 0.3)),
            ("spline", rss(&|c| c.sm_method = SmoothingMethod::Spline)),
            (
                "gcvspline",
                rss(&|c| c.sm_method = SmoothingMethod::GcvSpline),
            ),
            ("optlevel 3", rss(&|c| c.optlevel = 3)),
        ] {
            assert!(
                (v - base).abs() > 1e-6 * base,
                "{what} did not change the fit"
            );
        }
        let df5 = rss(&|c| {
            c.sm_method = SmoothingMethod::Spline;
            c.df = 5.0
        });
        let df8 = rss(&|c| {
            c.sm_method = SmoothingMethod::Spline;
            c.df = 8.0
        });
        assert!((df5 - df8).abs() > 1e-6 * df5);
        let p1 = rss(&|c| c.sm_method = SmoothingMethod::GcvSpline);
        let p2 = rss(&|c| {
            c.sm_method = SmoothingMethod::GcvSpline;
            c.gcvpen = 2.0
        });
        assert!((p1 - p2).abs() > 1e-6 * p1);
    }
    #[test]
    fn test_smoothing_method_parses_aliases_and_rejects_unknown() {
        SmoothingMethod::choices().validate().unwrap();
        for (s, m) in [
            ("supsmu", SmoothingMethod::Supsmu),
            ("Super Smoother", SmoothingMethod::Supsmu),
            ("spline", SmoothingMethod::Spline),
            ("GCVspline", SmoothingMethod::GcvSpline),
            ("gcv-spline", SmoothingMethod::GcvSpline),
        ] {
            assert_eq!(s.parse::<SmoothingMethod>().unwrap(), m, "{s}");
        }
        let e = "loess".parse::<SmoothingMethod>().unwrap_err().to_string();
        assert!(
            e.contains("sm.method") && e.contains("supsmu") && e.contains("gcvspline"),
            "{e}"
        );
        assert_eq!(SmoothingMethod::GcvSpline.to_string(), "gcvspline");
    }
    #[test]
    fn test_ppr_rejects_invalid_options() {
        let (_, x, y, _) = validate::ppr_case();
        let err = |f: &dyn Fn(&mut PprConfig)| {
            let mut c = validate::cfg();
            f(&mut c);
            ppr_with(x.view(), &y, None, &c).unwrap_err().to_string()
        };
        assert!(err(&|c| c.optlevel = 4).contains("optlevel"));
        assert!(err(&|c| c.bass = 11.0).contains("bass"));
        assert!(err(&|c| c.bass = f64::NAN).contains("bass"));
        assert!(err(&|c| c.span = 1.5).contains("span"));
        assert!(err(&|c| c.span = -0.1).contains("span"));
        assert!(
            err(&|c| {
                c.sm_method = SmoothingMethod::Spline;
                c.df = 1.0
            })
            .contains("df")
        );
        assert!(
            err(&|c| {
                c.sm_method = SmoothingMethod::Spline;
                c.df = 16.0
            })
            .contains("df")
        );
        assert!(
            err(&|c| {
                c.sm_method = SmoothingMethod::GcvSpline;
                c.gcvpen = -1.0
            })
            .contains("gcvpen")
        );
        assert!(
            err(&|c| {
                c.nterms = 3;
                c.max_terms = 2
            })
            .contains("max_terms")
        );
        assert!(err(&|c| c.tol = 0.0).contains("tol"));
        assert!(err(&|c| c.max_iter = 0).contains("max_iter"));
        let zero = vec![0.0; y.len()];
        assert!(
            ppr_with(x.view(), &y, Some(&zero), &validate::cfg())
                .unwrap_err()
                .to_string()
                .contains("sum to zero")
        );
        let flat = vec![2.5; y.len()];
        assert!(
            ppr_with(x.view(), &flat, None, &validate::cfg())
                .unwrap_err()
                .to_string()
                .contains("constant")
        );
        let k = ndarray::Array2::from_elem((y.len(), 2), 1.0);
        assert!(
            ppr_with(k.view(), &y, None, &validate::cfg())
                .unwrap_err()
                .to_string()
                .contains("every predictor")
        );
        let x3 = x.slice(ndarray::s![0..3, ..]).to_owned();
        assert!(
            ppr_with(x3.view(), &y[..3], None, &PprConfig::default())
                .unwrap_err()
                .to_string()
                .contains("at least 4")
        );
        let x5 = x.slice(ndarray::s![0..5, ..]).to_owned();
        let spline5 = PprConfig {
            sm_method: SmoothingMethod::Spline,
            df: 3.0,
            ..Default::default()
        };
        assert!(
            ppr_with(x5.view(), &y[..5], None, &spline5)
                .unwrap_err()
                .to_string()
                .contains("at least 6")
        );
        let (case, _, _, _) = validate::ppr_case();
        let r = ppr_with(x.view(), &y, None, &validate::cfg()).unwrap();
        let e = r
            .try_predict(validate::newx(&case).slice(ndarray::s![.., 0..2]))
            .unwrap_err()
            .to_string();
        assert!(
            e.contains("wrong number of columns in 'x' (3 expected, got 2)"),
            "{e}"
        );
        assert!(r.predict(&[vec![0.0, 1.0]])[0].is_nan());
    }
    /// An option the chosen smoother does not use is reported, not silently ignored.
    #[test]
    fn test_ppr_reports_options_the_smoother_ignores() {
        let (_, x, y, _) = validate::ppr_case();
        let mut c = validate::cfg();
        c.sm_method = SmoothingMethod::Spline;
        c.bass = 5.0;
        c.gcvpen = 2.0;
        let r = ppr_with(x.view(), &y, None, &c).unwrap();
        let has = |a: &str, b: &str| r.warnings.iter().any(|w| w.contains(a) && w.contains(b));
        assert!(has("bass = 5", "supsmu"), "{:?}", r.warnings);
        assert!(has("gcvpen = 2", "gcvspline"), "{:?}", r.warnings);
        let s = r.to_string();
        assert!(
            s.contains("Projection direction vectors") && s.contains("Warning: PPR: bass = 5"),
            "{s}"
        );
        let mut d = validate::cfg();
        d.df = 8.0; // supsmu ignores df
        assert!(
            ppr_with(x.view(), &y, None, &d)
                .unwrap()
                .warnings
                .iter()
                .any(|w| w.contains("df = 8"))
        );
    }
}
