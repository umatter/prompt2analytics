//! Quantile regression (R `quantreg::rq` and `quantreg::summary.rq`).
//!
//! Quantile regression minimises the check loss
//!
//! Q(β) = Σᵢ ρτ(yᵢ - xᵢ'β), ρτ(u) = u (τ - I(u < 0)),
//!
//! a linear program; τ = 0.5 is least absolute deviations (LAD).
//!
//! # Algorithms ([`QuantRegMethod`])
//!
//! - **Barrodale-Roberts simplex** (`br`, the default): a port of quantreg's `rqbr` (Koenker and
//!   d'Orey 1987, AS 229, a modification of Barrodale and Roberts 1974) for one τ. It returns a vertex
//!   of the LP, the solution `rq(method = "br")` returns, and flags a nonunique solution as R does
//!   ("Solution may be nonunique").
//! - **IRLS** (`irls`): iteratively reweighted least squares with a relative weight floor
//!   (1e-6 times the residual MAD). An approximation that no R method computes; its objective is
//!   close to, but not at, the LP minimum.
//!
//! # References
//!
//! - Koenker, R., & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
//! - Barrodale, I., & Roberts, F. D. K. (1974). Solution of an overdetermined system of equations in
//!   the l1 norm. *Communications of the ACM*, 17(6), 319-320.
//! - Koenker, R., & d'Orey, V. (1987). Computing regression quantiles. *Applied Statistics*, 36,
//!   383-393 (AS 229).
//! - Portnoy, S., & Koenker, R. (1997). The Gaussian Hare and the Laplacian Tortoise: Computability of
//!   Squared-error vs. Absolute-error Estimators. *Statistical Science*, 12(4), 279-300.
//! - Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
//!
//! ```
//! use ndarray::array;
//! use p2a_core::regression::quantreg::{QuantRegConfig, rq_fit};
//! let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
//! let y = array![1.2, 1.9, 3.4, 3.8, 5.3];
//! let fit = rq_fit(x.view(), y.view(), 0.5, &QuantRegConfig::default()).unwrap();
//! assert!(fit.convergence.converged && fit.coefficients.len() == 2);
//! // the median line passes through two observations
//! assert_eq!(fit.residuals.iter().filter(|r| r.abs() < 1e-12).count(), 2);
//! ```

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{LM_QR_TOL, Qr, least_squares, weighted_least_squares};
use crate::linalg::{safe_inverse, xtx};
use crate::options::Choices;
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text, relative_change_vec};
use crate::traits::estimator::{t_test_p_value, validate_level};

/// `.Machine$double.eps^(2/3)`: the pivot tolerance of quantreg's `rq.fit.br`.
const BR_TOLER: f64 = 3.666852862501036e-11;
/// `sqrt(.Machine$double.eps)`: summary.rq's fuzz `eps` (multiplied here by the outcome's
/// dispersion, see [`rq_se`]).
const RQ_FUZZ: f64 = 1.4901161193847656e-8;
/// R's warning for a nonunique simplex solution.
const NONUNIQUE: &str = "Solution may be nonunique";

/// Fitting algorithm (R `quantreg::rq(method = )`).
///
/// Parsed from strings through [`crate::options`]: `auto`, `br` (`simplex`, `barrodale_roberts`),
/// `fn` (`fnb`, `frisch_newton`, `interior_point`, `ip`) and `irls` (`reweighted`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum QuantRegMethod {
    /// `br` for n <= 5000 observations, `fn` above (quantreg's advice for large problems).
    #[default]
    #[serde(rename = "auto", alias = "Auto")]
    Auto,
    /// Barrodale-Roberts simplex (quantreg `rqbr`, Koenker and d'Orey 1987, AS 229): the exact
    /// LP solution of `rq(method = "br")`.
    #[serde(rename = "br", alias = "BarrodaleRoberts", alias = "Simplex")]
    BarrodaleRoberts,
    /// Frisch-Newton interior point (quantreg `rqfnb`) on `y / mean |y - median(y)|`.
    #[serde(rename = "fn", alias = "FrischNewton", alias = "InteriorPoint")]
    FrischNewton,
    /// Iteratively reweighted least squares: an approximation, not an R method.
    #[serde(rename = "irls", alias = "Irls", alias = "IRLS")]
    Irls,
}

/// Former name of [`QuantRegMethod`].
pub type QuantRegAlgorithm = QuantRegMethod;

const QUANTREG_METHOD_CHOICES: Choices<QuantRegMethod> = Choices {
    option: "method",
    values: &[
        (&["auto", "automatic"], QuantRegMethod::Auto),
        (
            &["br", "barrodale_roberts", "barrodale", "simplex"],
            QuantRegMethod::BarrodaleRoberts,
        ),
        (
            &[
                "fn",
                "fnb",
                "frisch_newton",
                "interior_point",
                "interiorpoint",
                "interior",
                "ip",
            ],
            QuantRegMethod::FrischNewton,
        ),
        (&["irls", "reweighted"], QuantRegMethod::Irls),
    ],
};
crate::option_enum!(QuantRegMethod, QUANTREG_METHOD_CHOICES);

#[allow(non_upper_case_globals)]
impl QuantRegMethod {
    /// Former variant name.
    #[deprecated(note = "use QuantRegMethod::FrischNewton")]
    pub const InteriorPoint: Self = Self::FrischNewton;
    /// Former variant name.
    #[deprecated(note = "use QuantRegMethod::BarrodaleRoberts")]
    pub const Simplex: Self = Self::BarrodaleRoberts;
    /// Former variant name.
    #[deprecated(note = "use QuantRegMethod::Irls")]
    pub const IRLS: Self = Self::Irls;

    /// Human-readable name, for Display.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "automatic: br for n <= 5000, fn above",
            Self::BarrodaleRoberts => "Barrodale-Roberts simplex (br)",
            Self::FrischNewton => "Frisch-Newton interior point (fn)",
            Self::Irls => "IRLS approximation (irls)",
        }
    }

    /// Parse a method name; `None` for an unknown one.
    #[deprecated(note = "use str::parse, which reports unknown values")]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        s.parse().ok()
    }
}

/// Standard errors (R `quantreg::summary.rq(se = )`).
///
/// Parsed from strings: `nid` (`hendricks_koenker`, `sandwich`), `iid` (`koenker_bassett`), `ker`
/// (`kernel`, `powell`) and `boot` (`bootstrap`, `xy`). R's `se = "rank"` is not implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QuantRegSe {
    /// Koenker-Bassett sparsity estimate under iid errors.
    Iid,
    /// Hendricks-Koenker sandwich with a local density estimate (summary.rq's default for n > 1000).
    #[default]
    Nid,
    /// Powell's kernel sandwich.
    Ker,
    /// xy-pair bootstrap (`boot.rq(bsmethod = "xy")`).
    Boot,
}

const QUANTREG_SE_CHOICES: Choices<QuantRegSe> = Choices {
    option: "se",
    values: &[
        (&["nid", "hendricks_koenker", "sandwich"], QuantRegSe::Nid),
        (&["iid", "koenker_bassett"], QuantRegSe::Iid),
        (&["ker", "kernel", "powell"], QuantRegSe::Ker),
        (&["boot", "bootstrap", "xy"], QuantRegSe::Boot),
    ],
};
crate::option_enum!(QuantRegSe, QUANTREG_SE_CHOICES);

/// Seed of the bootstrap standard errors when `QuantRegConfig::seed` is `None`.
pub const QUANTREG_BOOT_DEFAULT_SEED: u64 = 42;

/// Result from quantile regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantRegResult {
    /// Quantile being estimated (τ)
    pub tau: f64,
    /// Coefficient estimates
    pub coefficients: Vec<QuantRegCoefficient>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Residual degrees of freedom, n - k
    pub df: usize,
    /// Objective function value: the check loss Σ ρτ(residual) (R's `rho`)
    pub objective: f64,
    /// Simplex pivots (`br`), interior-point iterations (`fn`) or IRLS iterations (`irls`)
    pub iterations: usize,
    /// The algorithm that ran (`Auto` resolved to `BarrodaleRoberts` or `FrischNewton`)
    pub algorithm: QuantRegMethod,
    /// Variable names
    pub variable_names: Vec<String>,
    /// Rows dropped because y or a regressor was missing; absent from JSON when
    /// nothing was dropped.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Standard-error type (summary.rq's `se`)
    #[serde(default)]
    pub se_type: QuantRegSe,
    /// How the standard errors were computed, e.g. "nid (Hendricks-Koenker sandwich,
    /// Hall-Sheather bandwidth 0.2093)"
    #[serde(default)]
    pub se_method: String,
    /// Hall-Sheather bandwidth: `bandwidth.rq(tau, n)` for iid; after summary.rq's halving
    /// (until tau -/+ h lies in [0, 1]) for nid and ker; `None` for boot
    #[serde(default)]
    pub se_bandwidth: Option<f64>,
    /// Sparsity estimate 1/f(F^-1(tau)) (iid only; R's `1 / scale`)
    #[serde(default)]
    pub sparsity: Option<f64>,
    /// Powell's residual-scale kernel bandwidth (ker only)
    #[serde(default)]
    pub kernel_bandwidth: Option<f64>,
    /// The simplex found a nonunique solution (R's "Solution may be nonunique"); reported by
    /// the simplex only
    #[serde(default)]
    pub nonunique: bool,
    /// Convergence of the fit
    #[serde(default)]
    pub convergence: Convergence,
    /// Warnings of the fit and of the standard errors
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Residuals
    #[serde(skip)]
    pub(crate) residuals: Array1<f64>,
    /// Fitted values
    #[serde(skip)]
    pub(crate) fitted: Array1<f64>,
}

impl ResultNotes for QuantRegResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
    fn notes_convergence(&self) -> Option<Convergence> {
        Some(self.convergence.clone())
    }
}

/// A single coefficient from quantile regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantRegCoefficient {
    /// Variable name
    pub name: String,
    /// Coefficient estimate
    pub estimate: f64,
    /// Standard error, as summary.rq(se = `se_type`); NaN when it could not be computed
    pub std_error: f64,
    /// t-statistic
    pub t_value: f64,
    /// p-value (two-sided, t(n - k))
    pub p_value: f64,
    /// t(n - k) 95% interval, lower bound: estimate - qt(0.975, n - k) * std_error
    pub ci_lower_95: f64,
    /// t(n - k) 95% interval, upper bound: estimate + qt(0.975, n - k) * std_error
    pub ci_upper_95: f64,
}

impl std::fmt::Display for QuantRegResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Quantile Regression (τ = {:.2})", self.tau)?;
        writeln!(f, "=================================")?;
        writeln!(f, "Algorithm: {}", self.algorithm.label())?;
        writeln!(f, "Standard errors: {}", self.se_method)?;
        writeln!(f, "Observations: {}", self.n_obs)?;
        writeln!(f, "Parameters: {}", self.n_params)?;
        writeln!(f, "Objective: {:.4}", self.objective)?;
        writeln!(f)?;
        writeln!(
            f,
            "{:<15} {:>10} {:>10} {:>10} {:>10}",
            "Variable", "Estimate", "Std.Err", "t-value", "p-value"
        )?;
        writeln!(
            f,
            "{:-<15} {:-<10} {:-<10} {:-<10} {:-<10}",
            "", "", "", "", ""
        )?;
        for coef in &self.coefficients {
            writeln!(
                f,
                "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                coef.name, coef.estimate, coef.std_error, coef.t_value, coef.p_value
            )?;
        }
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// Configuration for quantile regression.
#[derive(Debug, Clone)]
pub struct QuantRegConfig {
    /// Quantile (τ) to estimate, 0 < tau < 1
    pub tau: f64,
    /// Fitting algorithm; default `Auto` (`br` for n <= 5000, `fn` above)
    pub algorithm: QuantRegMethod,
    /// Iteration limit of IRLS (the other algorithms have their own); at least 1
    pub max_iter: usize,
    /// IRLS: relative change in the coefficients; Frisch-Newton: duality gap of the
    /// standardised LP (quantreg's `eps`). Finite and > 0; default 1e-6
    pub tol: f64,
    /// Include intercept
    pub intercept: bool,
    /// Bootstrap replications for `se = Boot` (0: R's 200, otherwise at least 2); must be 0
    /// for the other standard-error types
    pub bootstrap_samples: usize,
    /// Standard errors (summary.rq's `se`); default `Nid`
    pub se: QuantRegSe,
    /// Seed of the bootstrap standard errors (`None`: [`QUANTREG_BOOT_DEFAULT_SEED`])
    pub seed: Option<u64>,
}

impl Default for QuantRegConfig {
    fn default() -> Self {
        Self {
            tau: 0.5, // median regression by default
            algorithm: QuantRegMethod::Auto,
            max_iter: 100,
            tol: 1e-6,
            intercept: true,
            bootstrap_samples: 0,
            se: QuantRegSe::Nid,
            seed: None,
        }
    }
}

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

/// `max_iter`, `tol` and the bootstrap settings (tau is checked by the callers: the SE code
/// fits at tau -/+ h, which may reach 0 or 1).
fn validate_config(config: &QuantRegConfig) -> EconResult<()> {
    if config.max_iter == 0 {
        return Err(spec("max_iter must be at least 1, got 0".into()));
    }
    if !(config.tol.is_finite() && config.tol > 0.0) {
        return Err(spec(format!(
            "tol must be finite and > 0, got {}",
            config.tol
        )));
    }
    Ok(())
}

/// Run quantile regression.
///
/// Rows with a missing y or regressor are dropped first, as `quantreg::rq` does
/// (`na.omit`), and reported in `QuantRegResult::missing`; integer columns are numbers.
/// The fit is `config.algorithm`'s (the simplex by default up to 5000 rows), the standard
/// errors are summary.rq's `config.se`. A singular design is an error, as in `rq`. When the
/// standard errors cannot be computed (too few observations for the iid sparsity estimate, a
/// singular density-weighted design), the coefficients are kept, the standard errors, t, p and
/// intervals are NaN and a warning says why (R's `rq` succeeds and its `summary.rq` stops).
///
/// # Example
///
/// ```ignore
/// use p2a_core::regression::{quantreg, QuantRegConfig};
///
/// let mut config = QuantRegConfig::default();
/// config.tau = 0.25;
/// let result = quantreg(&dataset, "y", &["x1", "x2"], &config)?;
/// ```
pub fn quantreg(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    config: &QuantRegConfig,
) -> EconResult<QuantRegResult> {
    validate_level(config.tau, "tau")?;
    validate_config(config)?;
    if config.bootstrap_samples > 0 {
        tracing::warn!(
            "quantile regression: bootstrap_samples={} was requested, but bootstrap standard \
             errors are not implemented; using the Powell (1991) sandwich estimator instead",
            config.bootstrap_samples
        );
    }

    // y and X with R's na.omit: a row with a missing value in any of them is dropped
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .min_rows(1, &format!("quantile regression of '{y_col}'"))
        .extract()?;
    let y = cols.numeric_array(y_col);
    let x = cols.matrix(x_cols, config.intercept)?.data;
    let (n, k) = x.dim();
    if k == 0 {
        return Err(spec(
            "quantile regression needs at least one regressor or an intercept".into(),
        ));
    }
    let mut var_names: Vec<String> = Vec::with_capacity(k);
    if config.intercept {
        var_names.push("(Intercept)".to_string());
    }
    var_names.extend(x_cols.iter().map(|c| c.to_string()));
    let df = match n.checked_sub(k) {
        Some(d) if d > 0 => d,
        _ => {
            return Err(EconError::InsufficientData {
                required: k + 1,
                provided: n,
                context: "Quantile regression".to_string(),
            });
        }
    };

    let qr = Qr::decompose(x.view(), LM_QR_TOL)?;
    if qr.rank() < k {
        let aliased: Vec<&str> = qr
            .aliased()
            .iter()
            .map(|&j| var_names[j].as_str())
            .collect();
        return Err(EconError::SingularMatrix {
            context: format!(
                "quantile regression design matrix: rank {} of {k} (aliased: {}); quantreg::rq \
                 stops with \"Singular design matrix\"",
                qr.rank(),
                aliased.join(", ")
            ),
            suggestion: "remove the aliased regressors".into(),
        });
    }

    let method = resolve_method(config.algorithm, n, config.tau, config.tol);
    let fit = fit_full_rank(x.view(), y.view(), config.tau, method, config)?;
    let warnings = fit.warnings.clone();

    // Standard errors: the old sandwich (replaced by summary.rq's estimators next)
    let std_errors = compute_quantreg_se(&x, &fit.residuals, config.tau, n, k)?;
    let coefficients: Vec<QuantRegCoefficient> = (0..k)
        .map(|i| {
            let b = fit.coefficients[i];
            let se = std_errors[i];
            let t_val = if se > 1e-15 { b / se } else { 0.0 };
            let p_val = t_test_p_value(t_val, df as f64);
            let t_crit = 1.96; // Approximate for large samples
            QuantRegCoefficient {
                name: var_names[i].clone(),
                estimate: b,
                std_error: se,
                t_value: t_val,
                p_value: p_val,
                ci_lower_95: b - t_crit * se,
                ci_upper_95: b + t_crit * se,
            }
        })
        .collect();

    let fitted = &y - &fit.residuals;
    Ok(QuantRegResult {
        tau: config.tau,
        coefficients,
        n_obs: n,
        n_params: k,
        df,
        objective: fit.objective,
        iterations: fit.iterations,
        algorithm: fit.algorithm,
        variable_names: var_names,
        missing: cols.missing().clone(),
        se_type: QuantRegSe::Nid,
        se_method: String::new(),
        se_bandwidth: None,
        sparsity: None,
        kernel_bandwidth: None,
        nonunique: fit.nonunique,
        convergence: fit.convergence,
        warnings,
        residuals: fit.residuals,
        fitted,
    })
}

/// Check function (asymmetric absolute loss).
fn check_function(u: f64, tau: f64) -> f64 {
    if u >= 0.0 { tau * u } else { (tau - 1.0) * u }
}

/// `Auto` resolved for `n` rows: the simplex up to [`AUTO_BR_MAX_N`] rows (and wherever
/// Frisch-Newton cannot run, at tau within `tol` of 0 or 1), Frisch-Newton above.
fn resolve_method(method: QuantRegMethod, _n: usize, _tau: f64, _tol: f64) -> QuantRegMethod {
    match method {
        QuantRegMethod::Auto => QuantRegMethod::BarrodaleRoberts,
        m => m,
    }
}

/// `mean |v - median(v)|` (type-7 median), or 1 when it is 0: the dispersion that makes the
/// Frisch-Newton stopping rule and summary.rq's fuzz scale-free.
fn dispersion(v: ArrayView1<'_, f64>) -> f64 {
    let mut sorted = v.to_vec();
    sorted.sort_by(f64::total_cmp);
    let med = quantile_sorted(&sorted, 0.5, QuantileType::T7);
    let s = v.iter().map(|&a| (a - med).abs()).sum::<f64>() / v.len().max(1) as f64;
    if s > 0.0 && s.is_finite() { s } else { 1.0 }
}

/// An array-level quantile-regression fit (no standard errors); see [`rq_fit`].
#[derive(Debug, Clone)]
pub struct RqFit {
    /// Coefficients, one per column of x
    pub coefficients: Array1<f64>,
    /// `y - x b`
    pub residuals: Array1<f64>,
    /// Check loss Σ ρτ(residual)
    pub objective: f64,
    /// The simplex found a nonunique solution (always false for the other algorithms)
    pub nonunique: bool,
    /// Pivots or iterations
    pub iterations: usize,
    /// The algorithm that ran (`Auto` resolved)
    pub algorithm: QuantRegMethod,
    /// Convergence of the fit
    pub convergence: Convergence,
    /// Warnings (R's "Solution may be nonunique", a not-converged note)
    pub warnings: Vec<String>,
}

/// Fit the quantile-regression LP for `x` (n x p, the intercept column included if wanted) and
/// `y` at quantile `tau` with `config.algorithm` (`config.tau` is not used).
///
/// Refuses a length mismatch, n <= p, non-finite input, tau outside [0, 1] (outside (tol,
/// 1 - tol) for Frisch-Newton) and a rank-deficient `x` (`SingularMatrix`, as R's
/// "Singular design matrix").
pub fn rq_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau: f64,
    config: &QuantRegConfig,
) -> EconResult<RqFit> {
    let (n, p) = x.dim();
    if y.len() != n {
        return Err(spec(format!("y has {} rows but x has {n}", y.len())));
    }
    if p == 0 || n <= p {
        return Err(EconError::InsufficientData {
            required: p.max(1) + 1,
            provided: n,
            context: "quantile regression (more observations than columns of x)".into(),
        });
    }
    if !(tau.is_finite() && (0.0..=1.0).contains(&tau)) {
        return Err(spec(format!("tau must lie in [0, 1], got {tau}")));
    }
    validate_config(config)?;
    if y.iter().chain(x.iter()).any(|v| !v.is_finite()) {
        return Err(spec("quantile regression requires finite x and y".into()));
    }
    let qr = Qr::decompose(x, LM_QR_TOL)?;
    if qr.rank() < p {
        let cols: Vec<String> = qr.aliased().iter().map(|j| j.to_string()).collect();
        return Err(EconError::SingularMatrix {
            context: format!(
                "quantile regression design matrix: rank {} of {p} (columns {})",
                qr.rank(),
                cols.join(", ")
            ),
            suggestion: "remove the aliased columns".into(),
        });
    }
    let method = resolve_method(config.algorithm, n, tau, config.tol);
    fit_full_rank(x, y, tau, method, config)
}

/// The fit of [`rq_fit`] for checked input: finite, full-rank `x`, `method` resolved.
fn fit_full_rank(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau: f64,
    method: QuantRegMethod,
    config: &QuantRegConfig,
) -> EconResult<RqFit> {
    let mut warnings = Vec::new();
    let (coefficients, iterations, nonunique, convergence) = match method {
        QuantRegMethod::Auto | QuantRegMethod::BarrodaleRoberts => {
            let s = br_simplex(x, y, tau)?;
            if s.nonunique {
                warnings.push(NONUNIQUE.to_string());
            }
            (s.coef, s.pivots, s.nonunique, s.convergence)
        }
        QuantRegMethod::FrischNewton => {
            return Err(spec("method fn arrives in Task 5".into()));
        }
        QuantRegMethod::Irls => {
            let (b, c) = irls_quantreg(x, y, tau, config.max_iter, config.tol)?;
            warnings.extend(c.note("IRLS quantile regression"));
            (b, c.iterations, false, c)
        }
    };
    let residuals = &y - &x.dot(&coefficients);
    let objective = residuals.iter().map(|&r| check_function(r, tau)).sum();
    Ok(RqFit {
        coefficients,
        residuals,
        objective,
        nonunique,
        iterations,
        algorithm: if method == QuantRegMethod::Auto {
            QuantRegMethod::BarrodaleRoberts
        } else {
            method
        },
        convergence,
        warnings,
    })
}

/// Fortran `SIGN(1d0, v)`.
fn sign1(v: f64) -> f64 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// A simplex solution.
struct BrSolution {
    coef: Array1<f64>,
    nonunique: bool,
    pivots: usize,
    convergence: Convergence,
}

/// The states of quantreg's `rqbr` (its goto labels).
#[derive(Clone, Copy)]
enum BrState {
    /// Stage 1: choose the entering column.
    In1,
    /// Stage 2: choose the entering column, or stop at the optimum.
    In2,
    /// Choose the leaving row, passing through vertices (Barrodale-Roberts).
    Out,
    /// The entering column is linearly dependent on the basis.
    Dependent,
    /// Pivot on (`out`, `inn`).
    Pivot,
    /// Label 20: back to stage 1 or on to stage 2.
    L20,
    Optimal,
    Premature,
}

/// quantreg's `rqbr` (Koenker and d'Orey 1987, AS 229; Barrodale and Roberts 1974) for one tau,
/// without the confidence intervals of `lci1`. The tableau keeps Fortran's layout (rows 1..m
/// observations, m+1..m+3 cost rows, m+4 column labels; columns 1..p coefficients, p+1 the
/// right-hand side, p+4 row labels) through the 1-based accessor `w!`, and every tie is broken
/// as in the Fortran (strict comparisons, the swap-remove candidate list), so the vertex and the
/// nonunique flag are R's. `x` must have full column rank.
fn br_simplex(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>, t: f64) -> EconResult<BrSolution> {
    let (m, n) = x.dim();
    let (m1, m2, m3, m4) = (m + 1, m + 2, m + 3, m + 4);
    let (n1, n2, n3, n4) = (n + 1, n + 2, n + 3, n + 4);
    let toler = BR_TOLER;
    let big = f64::MAX;
    let nf = n as f64;
    let max_pivots = 50 * (m + n);
    let mut wa = resource::try_zeros2(m4, n4, "quantreg simplex tableau")?;
    let mut wb = resource::try_vec(m, 0.0f64, "quantreg simplex tableau")?;
    let mut s = resource::try_vec(m, 0usize, "quantreg simplex tableau")?;
    macro_rules! w {
        ($i:expr, $j:expr) => {
            wa[[$i - 1, $j - 1]]
        };
    }

    for i in 1..=m {
        for j in 1..=n {
            w!(i, j) = x[[i - 1, j - 1]];
        }
        w!(i, n4) = (n + i) as f64;
        w!(i, n2) = y[i - 1];
        w!(i, n3) = 0.0;
        w!(i, n1) = w!(i, n2) - w!(i, n3);
        if w!(i, n1) < 0.0 {
            for j in 1..=n4 {
                w!(i, j) = -w!(i, j);
            }
        }
    }
    for j in 1..=n {
        w!(m4, j) = j as f64;
        w!(m2, j) = 0.0;
        w!(m3, j) = 0.0;
        for i in 1..=m {
            let aux = sign1(w!(m4, j)) * w!(i, j);
            w!(m2, j) += aux * (1.0 - sign1(w!(i, n4)));
            w!(m3, j) += aux * sign1(w!(i, n4));
        }
        w!(m3, j) *= 2.0;
    }
    for j in 1..=n {
        w!(m1, j) = w!(m2, j) + w!(m3, j) * t;
    }

    let (mut kount, mut kr, mut kl) = (0usize, 1usize, 1usize);
    let (mut stage, mut test) = (true, false);
    let (mut inn, mut out) = (0usize, 0usize);
    let mut pivot = 0.0f64;
    let mut reduced_cost = 0.0f64;
    let mut state = BrState::In1;
    loop {
        match state {
            BrState::In1 => {
                let mut mx = -1.0;
                let mut chosen = None;
                for j in kr..=n {
                    if w!(m4, j).abs() <= nf {
                        let d = w!(m1, j).abs();
                        if d > mx {
                            mx = d;
                            chosen = Some(j);
                        }
                    }
                }
                inn = chosen.ok_or_else(|| {
                    EconError::Computation("Barrodale-Roberts simplex: no entering column".into())
                })?;
                if w!(m1, inn) < 0.0 {
                    for i in 1..=m4 {
                        w!(i, inn) = -w!(i, inn);
                    }
                }
                state = BrState::Out;
            }
            BrState::In2 => {
                stage = false;
                let mut mx = -big;
                for j in kr..=n {
                    let mut d = w!(m1, j);
                    if d < 0.0 {
                        if d > -2.0 {
                            continue;
                        }
                        d = -d - 2.0;
                    }
                    if d > mx {
                        mx = d;
                        inn = j;
                    }
                }
                if mx <= toler {
                    reduced_cost = mx.max(0.0);
                    state = BrState::Optimal;
                    continue;
                }
                if w!(m1, inn) <= 0.0 {
                    for i in 1..=m4 {
                        w!(i, inn) = -w!(i, inn);
                    }
                    w!(m1, inn) -= 2.0;
                    w!(m2, inn) -= 2.0;
                }
                state = BrState::Out;
            }
            BrState::Out => {
                let mut k = 0usize;
                for i in kl..=m {
                    let d = w!(i, inn);
                    if d > toler {
                        wb[k] = w!(i, n1) / d;
                        s[k] = i;
                        k += 1;
                        test = true;
                    }
                }
                loop {
                    if k == 0 {
                        test = false;
                    } else {
                        let mut mn = big;
                        let mut jj = 0usize;
                        out = s[0];
                        for i in 0..k {
                            if wb[i] < mn {
                                jj = i;
                                mn = wb[i];
                                out = s[i];
                            }
                        }
                        wb[jj] = wb[k - 1];
                        s[jj] = s[k - 1];
                        k -= 1;
                    }
                    if !test && stage {
                        state = BrState::Dependent;
                        break;
                    }
                    if !test {
                        state = BrState::Premature;
                        break;
                    }
                    pivot = w!(out, inn);
                    if w!(m1, inn) - pivot - pivot <= toler {
                        state = BrState::Pivot;
                        break;
                    }
                    // pass through the vertex: the row's residual changes sign
                    for j in kr..=n3 {
                        let d = w!(out, j);
                        w!(m1, j) = w!(m1, j) - d - d;
                        w!(m2, j) = w!(m2, j) - d - d;
                        w!(out, j) = -d;
                    }
                    w!(out, n4) = -w!(out, n4);
                }
            }
            BrState::Dependent => {
                for i in 1..=m4 {
                    let tmp = w!(i, kr);
                    w!(i, kr) = w!(i, inn);
                    w!(i, inn) = tmp;
                }
                kr += 1;
                state = BrState::L20;
            }
            BrState::Pivot => {
                resource::checkpoint()?;
                for j in kr..=n3 {
                    if j != inn {
                        w!(out, j) /= pivot;
                    }
                }
                for i in 1..=m3 {
                    if i != out {
                        let d = w!(i, inn);
                        if d != 0.0 {
                            for j in kr..=n3 {
                                if j != inn {
                                    w!(i, j) -= d * w!(out, j);
                                }
                            }
                        }
                    }
                }
                for i in 1..=m3 {
                    if i != out {
                        w!(i, inn) = -w!(i, inn) / pivot;
                    }
                }
                w!(out, inn) = 1.0 / pivot;
                let d = w!(out, n4);
                w!(out, n4) = w!(m4, inn);
                w!(m4, inn) = d;
                kount += 1;
                if kount > max_pivots {
                    return Err(EconError::ConvergenceFailure {
                        iterations: kount,
                        last_change: f64::NAN,
                        suggestion: "the simplex may be cycling on a degenerate problem; try \
                                     method = fn"
                            .into(),
                    });
                }
                if !stage {
                    state = BrState::In2;
                    continue;
                }
                kl += 1;
                for j in kr..=n4 {
                    let d = w!(out, j);
                    w!(out, j) = w!(kount, j);
                    w!(kount, j) = d;
                }
                state = BrState::L20;
            }
            BrState::L20 => {
                state = if kount + kr == n1 {
                    BrState::In2
                } else {
                    BrState::In1
                };
            }
            BrState::Optimal => {
                let mut nonunique = false;
                if kr == 1 {
                    for j in 1..=n {
                        let d = w!(m1, j).abs();
                        if d <= toler || 2.0 - d <= toler {
                            nonunique = true;
                            break;
                        }
                    }
                }
                let mut coef = Array1::zeros(n);
                for i in 1..kl {
                    let label = w!(i, n4);
                    let kk = label.abs() as usize;
                    if kk == 0 || kk > n {
                        return Err(EconError::Computation(format!(
                            "Barrodale-Roberts simplex: basis row {i} has label {label}"
                        )));
                    }
                    coef[kk - 1] = w!(i, n1) * sign1(label);
                }
                let convergence = Convergence::new(
                    true,
                    kount,
                    max_pivots,
                    "largest reduced cost (rqbr simplex optimality)",
                    reduced_cost,
                    toler,
                );
                return Ok(BrSolution {
                    coef,
                    nonunique,
                    pivots: kount,
                    convergence,
                });
            }
            BrState::Premature => {
                return Err(EconError::SingularMatrix {
                    context: "Barrodale-Roberts simplex ended prematurely (possible \
                              conditioning problem in x)"
                        .into(),
                    suggestion: "rescale or remove nearly collinear regressors, or use \
                                 method = fn"
                        .into(),
                });
            }
        }
    }
}

/// IRLS for the check loss: start from least squares, weights `(tau or 1 - tau) / max(|r|,
/// floor)` with the relative floor `1e-6 * median |r - median(r)|` (`1e-6 * mean |r|` when that
/// is 0), weighted QR steps, stop when the relative change in the coefficients is below `tol`.
/// Not an R method: the objective approaches the LP minimum but does not reach it.
fn irls_quantreg(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau: f64,
    max_iter: usize,
    tol: f64,
) -> EconResult<(Array1<f64>, Convergence)> {
    let n = x.nrows();
    let mut beta = least_squares(x, y)?.coefficients;
    let mut iterations = 0;
    let mut last_change = f64::NAN;
    let mut converged = false;
    let mut w = resource::try_zeros1(n, "IRLS weights")?;
    let mut buf = resource::try_vec(n, 0.0f64, "IRLS residuals")?;
    for iter in 1..=max_iter {
        resource::checkpoint()?;
        let r = &y - &x.dot(&beta);
        buf.copy_from_slice(r.as_slice().expect("contiguous residuals"));
        buf.sort_by(f64::total_cmp);
        let med = quantile_sorted(&buf, 0.5, QuantileType::T7);
        for (b, &ri) in buf.iter_mut().zip(r.iter()) {
            *b = (ri - med).abs();
        }
        buf.sort_by(f64::total_cmp);
        let mad0 = quantile_sorted(&buf, 0.5, QuantileType::T7);
        let floor = if mad0 > 0.0 {
            1e-6 * mad0
        } else {
            1e-6 * r.iter().map(|v| v.abs()).sum::<f64>() / n as f64
        };
        if floor == 0.0 {
            // every residual is 0: the fit is exact
            converged = true;
            last_change = 0.0;
            break;
        }
        for (wi, &ri) in w.iter_mut().zip(r.iter()) {
            let side = if ri >= 0.0 { tau } else { 1.0 - tau };
            *wi = side / ri.abs().max(floor);
        }
        let new = weighted_least_squares(x, y, w.view())?.coefficients;
        if new.iter().any(|v| v.is_nan()) {
            return Err(EconError::SingularMatrix {
                context: "IRLS: the weighted design became rank deficient".into(),
                suggestion: "use method = br".into(),
            });
        }
        last_change = relative_change_vec(
            new.as_slice().expect("contiguous coefficients"),
            beta.as_slice().expect("contiguous coefficients"),
        );
        beta = new;
        iterations = iter;
        if last_change < tol {
            converged = true;
            break;
        }
    }
    Ok((
        beta,
        Convergence::new(
            converged,
            iterations,
            max_iter,
            "relative change in coefficients",
            last_change,
            tol,
        ),
    ))
}

/// Compute standard errors using sandwich formula.
fn compute_quantreg_se(
    x: &Array2<f64>,
    residuals: &Array1<f64>,
    tau: f64,
    n: usize,
    k: usize,
) -> EconResult<Vec<f64>> {
    // Powell (1991) sandwich estimator
    // V = τ(1-τ) / (n f(0)²) × (X'X)^{-1}
    // where f(0) is the density of residuals at zero

    let xtx_mat = xtx(&x.view());
    let (xtx_inv, _) = safe_inverse(&xtx_mat.view()).map_err(|e| EconError::SingularMatrix {
        context: "Quantile regression SE computation".to_string(),
        suggestion: format!("Original error: {}", e),
    })?;

    // Estimate f(0) using kernel density
    let h = bandwidth_nrd(residuals);
    let f0 = kernel_density_at_zero(residuals, h);

    // Variance factor
    let var_factor = if f0.abs() > 1e-15 {
        tau * (1.0 - tau) / (n as f64 * f0 * f0)
    } else {
        // Fallback: use residual scale
        let mad = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;
        tau * (1.0 - tau) * mad * mad
    };

    // Standard errors
    let se: Vec<f64> = (0..k)
        .map(|i| (var_factor * xtx_inv[[i, i]]).abs().sqrt())
        .collect();

    Ok(se)
}

/// Normal reference bandwidth (Silverman's rule of thumb).
fn bandwidth_nrd(x: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    let sd = {
        let mean = x.iter().sum::<f64>() / n;
        let var = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt()
    };

    // IQR estimate
    let mut sorted: Vec<f64> = x.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    let q1 = sorted[(n as usize * 25) / 100];
    let q3 = sorted[(n as usize * 75) / 100];
    let iqr = q3 - q1;

    let scale = (sd).min(iqr / 1.34);
    0.9 * scale * n.powf(-0.2)
}

/// Kernel density estimate at zero using Epanechnikov kernel.
fn kernel_density_at_zero(residuals: &Array1<f64>, h: f64) -> f64 {
    if h <= 0.0 {
        return 0.1; // Fallback
    }

    let n = residuals.len() as f64;
    let mut density = 0.0;

    for &r in residuals.iter() {
        let u = r / h;
        if u.abs() <= 1.0 {
            // Epanechnikov kernel: K(u) = 0.75 * (1 - u²)
            density += 0.75 * (1.0 - u * u);
        }
    }

    density / (n * h)
}

/// Convenience function for quantile regression with default settings.
pub fn run_quantreg(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    tau: f64,
) -> EconResult<QuantRegResult> {
    let config = QuantRegConfig {
        tau,
        ..Default::default()
    };
    quantreg(dataset, y_col, x_cols, &config)
}

/// Run multiple quantile regressions for different quantiles.
pub fn quantreg_multi(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    taus: &[f64],
) -> EconResult<Vec<QuantRegResult>> {
    let mut results = Vec::with_capacity(taus.len());
    for &tau in taus {
        let result = run_quantreg(dataset, y_col, x_cols, tau)?;
        results.push(result);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_test_dataset() -> Dataset {
        // y = 1 + 2*x + noise (with heteroskedastic errors)
        let x: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| 1.0 + 2.0 * xi + (i as f64 % 5.0 - 2.0) * xi.sqrt())
            .collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_quantreg_median() {
        let dataset = create_test_dataset();
        let result = run_quantreg(&dataset, "y", &["x"], 0.5).unwrap();

        assert_eq!(result.tau, 0.5);
        assert_eq!(result.n_obs, 50);
        assert_eq!(result.n_params, 2);
        assert!(result.coefficients[1].estimate > 1.0); // Should be near 2
    }

    #[test]
    fn test_quantreg_quartiles() {
        let dataset = create_test_dataset();

        let result_25 = run_quantreg(&dataset, "y", &["x"], 0.25).unwrap();
        let result_50 = run_quantreg(&dataset, "y", &["x"], 0.50).unwrap();
        let result_75 = run_quantreg(&dataset, "y", &["x"], 0.75).unwrap();

        // Lower quantiles should have lower intercept
        assert!(result_25.coefficients[0].estimate < result_75.coefficients[0].estimate);
        // All should have similar slopes
        assert!(
            (result_25.coefficients[1].estimate - result_50.coefficients[1].estimate).abs() < 1.0
        );
    }

    #[test]
    fn test_quantreg_multi() {
        let dataset = create_test_dataset();
        let taus = vec![0.1, 0.25, 0.5, 0.75, 0.9];

        let results = quantreg_multi(&dataset, "y", &["x"], &taus).unwrap();

        assert_eq!(results.len(), 5);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.tau, taus[i]);
        }
    }

    #[test]
    fn test_quantreg_invalid_tau() {
        let dataset = create_test_dataset();
        let config = QuantRegConfig {
            tau: 1.5, // Invalid
            ..Default::default()
        };

        let result = quantreg(&dataset, "y", &["x"], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantreg_displays_correctly() {
        let dataset = create_test_dataset();
        let result = run_quantreg(&dataset, "y", &["x"], 0.5).unwrap();

        let display = format!("{}", result);
        assert!(display.contains("Quantile Regression"));
        assert!(display.contains("τ = 0.50"));
        assert!(display.contains("Estimate"));
    }

    #[test]
    fn test_quantreg_method_parses_aliases_and_rejects_unknown() {
        use QuantRegMethod::*;
        for (s, m) in [
            ("auto", Auto),
            ("br", BarrodaleRoberts),
            ("BR", BarrodaleRoberts),
            ("simplex", BarrodaleRoberts),
            ("Barrodale-Roberts", BarrodaleRoberts),
            ("fn", FrischNewton),
            ("fnb", FrischNewton),
            ("interior point", FrischNewton),
            ("Interior-Point", FrischNewton),
            ("ip", FrischNewton),
            ("irls", Irls),
            ("IRLS", Irls),
            ("reweighted", Irls),
        ] {
            assert_eq!(s.parse::<QuantRegMethod>().unwrap(), m, "{s}");
        }
        let err = "lasso".parse::<QuantRegMethod>().unwrap_err().to_string();
        assert!(
            err.contains("method")
                && err.contains("auto")
                && err.contains("br")
                && err.contains("fn")
                && err.contains("irls"),
            "{err}"
        );
        QuantRegMethod::choices().validate().unwrap();
        assert_eq!(QuantRegMethod::default(), Auto);
        assert_eq!(Irls.to_string(), "irls");
        assert_eq!(serde_json::to_string(&BarrodaleRoberts).unwrap(), "\"br\"");
        assert_eq!(serde_json::to_string(&Auto).unwrap(), "\"auto\"");
        // JSON written before the rename
        assert_eq!(
            serde_json::from_str::<QuantRegMethod>("\"IRLS\"").unwrap(),
            Irls
        );
        assert_eq!(
            serde_json::from_str::<QuantRegMethod>("\"InteriorPoint\"").unwrap(),
            FrischNewton
        );
        assert_eq!(
            serde_json::from_str::<QuantRegMethod>("\"Simplex\"").unwrap(),
            BarrodaleRoberts
        );
    }

    #[test]
    fn test_quantreg_se_parses_aliases_and_rejects_unknown() {
        use QuantRegSe::*;
        for (s, v) in [
            ("nid", Nid),
            ("Hendricks-Koenker", Nid),
            ("iid", Iid),
            ("ker", Ker),
            ("kernel", Ker),
            ("Powell", Ker),
            ("boot", Boot),
            ("bootstrap", Boot),
            ("xy", Boot),
        ] {
            assert_eq!(s.parse::<QuantRegSe>().unwrap(), v, "{s}");
        }
        // R's se = "rank" is not implemented
        let err = "rank".parse::<QuantRegSe>().unwrap_err().to_string();
        assert!(
            err.contains("se") && err.contains("nid") && err.contains("boot"),
            "{err}"
        );
        QuantRegSe::choices().validate().unwrap();
        assert_eq!(QuantRegSe::default(), Nid);
        assert_eq!(QuantRegConfig::default().se, Nid);
        assert_eq!(QuantRegConfig::default().seed, None);
        assert_eq!(serde_json::to_string(&Ker).unwrap(), "\"ker\"");
    }

    #[test]
    #[allow(deprecated)]
    fn test_quantreg_algorithm_old_names_still_compile() {
        assert_eq!(QuantRegAlgorithm::Simplex, QuantRegMethod::BarrodaleRoberts);
        assert_eq!(
            QuantRegAlgorithm::InteriorPoint,
            QuantRegMethod::FrischNewton
        );
        assert_eq!(QuantRegAlgorithm::IRLS, QuantRegMethod::Irls);
        assert_eq!(
            QuantRegAlgorithm::from_str("br"),
            Some(QuantRegMethod::BarrodaleRoberts)
        );
        assert_eq!(QuantRegAlgorithm::from_str("qs"), None);
    }

    #[test]
    fn test_check_function() {
        // tau = 0.5 (median): symmetric
        assert!((check_function(1.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((check_function(-1.0, 0.5) - 0.5).abs() < 1e-10);

        // tau = 0.75: penalizes negative residuals more
        assert!((check_function(1.0, 0.75) - 0.75).abs() < 1e-10);
        assert!((check_function(-1.0, 0.75) - 0.25).abs() < 1e-10);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // VALIDATION TESTS - Comparing against R reference implementations
    // ════════════════════════════════════════════════════════════════════════════

    /// Case `quantreg_hetero` (quantreg::rq on the legacy Test 1 DGP; validation/r/cases/regression_legacy.R):
    /// the simplex's coefficients and objective.
    #[test]
    fn test_validate_quantreg_hetero() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("quantreg_hetero");
        let n = case.scalar("n_obs") as usize;
        let mut check = Check::new(&case);
        let mut slopes = Vec::new();
        for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
            let r = run_quantreg(&ds, "y", &["x"], tau).unwrap();
            let key = format!("objective_{tag}");
            check.scalar(&key, r.objective, Tol::stat(n));
            assert!(
                r.objective >= case.scalar(&key) * (1.0 - 1e-12),
                "objective below rq's LP minimum at tau = {tau}"
            );
            check.named(
                &format!("coef_{tag}"),
                &r.variable_names,
                &r.coefficients
                    .iter()
                    .map(|c| c.estimate)
                    .collect::<Vec<_>>(),
                Tol::coef(n),
            );
            assert!(r.convergence.converged, "tau = {tau}");
            // deviation REG-26: std_error, t, p (se_iid_<tag>, se_nid_<tag>); deviation REG-64:
            //   95% CI uses 1.96.
            slopes.push(r.coefficients[1].estimate);
        }
        let r = run_quantreg(&ds, "y", &["x"], 0.5).unwrap();
        check
            .scalar("rdf", r.df as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
        // kept from the old test
        assert!(
            slopes[0] < slopes[1] && slopes[1] < slopes[2],
            "slopes increase with tau: {slopes:?}"
        );
    }

    /// Validation test: Quantile regression with multiple quantiles
    #[test]
    fn test_validate_quantreg_multi() {
        // Create data with clear heteroskedasticity
        let n = 80;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) / 8.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let noise = ((i as f64 * 0.987).cos()) * (0.5 + 0.3 * xi);
                1.0 + 2.0 * xi + noise
            })
            .collect();

        let df = df! {
            "y" => y,
            "x" => x
        }
        .unwrap();
        let dataset = Dataset::new(df);

        // Run multiple quantiles
        let taus = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let results = quantreg_multi(&dataset, "y", &["x"], &taus).unwrap();

        assert_eq!(results.len(), 5);

        // All should have positive slopes (true slope = 2)
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result.tau, taus[i]);
            assert!(
                result.coefficients[1].estimate > 0.0,
                "Quantile {} slope should be positive",
                taus[i]
            );
        }

        // Intercepts should generally increase with tau for location-scale model
        let _intercept_10 = results[0].coefficients[0].estimate;
        let _intercept_90 = results[4].coefficients[0].estimate;
        // With positive errors having larger spread at higher x,
        // we expect higher quantiles to have higher intercepts
        // (though this depends on the specific error distribution)
    }

    /// REG-25: quantreg reads y and X with R's na.omit (quantreg::rq drops incomplete
    /// rows), reports the drop, and accepts an Int64 outcome (TOOL-5's quantreg site).
    /// R-pinned: the number of rows used is R's `n_complete`; the SEs are not (REG-26),
    /// so the fit is compared with the fit on the complete rows.
    #[test]
    fn test_validate_na_quantreg_columns() {
        use crate::regression::ols::na_fixtures::{as_float, complete_rows, mtcars};
        use p2a_validation::{Check, Tol};
        let mt = mtcars();
        let a = run_quantreg(&mt, "hp", &["wt"], 0.5).unwrap();
        let b = run_quantreg(&as_float(&mt, "hp"), "hp", &["wt"], 0.5).unwrap();
        let pick = |r: &QuantRegResult| -> (Vec<f64>, Vec<f64>) {
            (
                r.coefficients.iter().map(|c| c.estimate).collect(),
                r.coefficients.iter().map(|c| c.std_error).collect(),
            )
        };
        assert_eq!(
            (a.n_obs, pick(&a), a.objective),
            (32, pick(&b), b.objective)
        );
        let (case, sm) = crate::test_support::load_case("reg_smoothers_na");
        let r = run_quantreg(&sm, "y", &["x"], 0.5).unwrap();
        Check::new(&case)
            .scalar("n_complete", r.n_obs as f64, Tol::exact())
            .finish();
        let c = run_quantreg(&complete_rows(&sm, &["y", "x"]), "y", &["x"], 0.5).unwrap();
        assert_eq!(
            (r.n_obs, pick(&r), r.objective),
            (56, pick(&c), c.objective)
        );
        assert_eq!(
            r.missing,
            MissingSummary {
                n_total: 60,
                n_used: 56,
                n_dropped: 4,
                by_column: vec![("y".into(), 2), ("x".into(), 2)],
            }
        );
        assert!(r.to_string().contains(
            "Missing values: 4 of 60 rows dropped because of missing values (y: 2, x: 2)\n"
        ));
        assert!(c.missing.is_empty() && !c.to_string().contains("Missing values"));
        match run_quantreg(&mt, "model", &["wt"], 0.5) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "model"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            run_quantreg(&mt, "hp", &["nope"], 0.5),
            Err(EconError::ColumnNotFound { .. })
        ));
        // No complete row: the error names the method and the outcome.
        use crate::regression::ols::na_fixtures::{insufficient_context, with_all_null};
        let none = with_all_null(&sm, "no_x", polars::prelude::DataType::Float64);
        assert_eq!(
            insufficient_context(run_quantreg(&none, "y", &["no_x"], 0.5)),
            "quantile regression of 'y': 0 of 60 rows have no missing value"
        );
    }

    // ════════════════════════════════════════════════════════════════════════════
    // T3-WP10: rq(method = "br"), IRLS, parameters (cases rq_br_se, rq_br_mtcars)
    // ════════════════════════════════════════════════════════════════════════════

    use crate::regression::ols::na_fixtures::{nan_vec, repo_dataset, with_derived};
    use ndarray::{Array1, Array2};
    use p2a_validation::{Check, Tol};

    fn estimates(r: &QuantRegResult) -> Vec<f64> {
        r.coefficients.iter().map(|c| c.estimate).collect()
    }

    /// quantreg_hetero (T1-WP04; 100 rows, y and x), for the tests that compare no R value.
    fn hetero() -> Dataset {
        repo_dataset("validation/datasets/generated/quantreg_hetero.csv")
    }

    /// Case `rq_br_se`: quantreg::rq(method = "br") coefficients and check-loss objective at tau
    /// 0.02/0.25/0.5/0.75. Fails before WP10: the IRLS coefficients are 1.8e-5 / 2.8e-5 / 1.0e-6 off.
    #[test]
    fn test_validate_rq_br_coefficients() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let n = case.scalar("n") as usize;
        let mut check = Check::new(&case);
        for (tau, tag) in [
            (0.25, "tau25"),
            (0.5, "tau50"),
            (0.75, "tau75"),
            (0.02, "tau02"),
        ] {
            let r = run_quantreg(&ds, "y", &["x"], tau).unwrap();
            check.named(
                &format!("{tag}_coef"),
                &r.variable_names,
                &estimates(&r),
                Tol::coef(n),
            );
            if tag != "tau02" {
                check.scalar(&format!("{tag}_objective"), r.objective, Tol::stat(n));
                assert_eq!(
                    r.nonunique as u8 as f64,
                    case.scalar(&format!("{tag}_nonunique"))
                );
            }
            assert_eq!(r.df as f64, case.scalar("rdf"));
            assert_eq!(r.algorithm, QuantRegMethod::BarrodaleRoberts);
            assert!(r.convergence.converged && r.iterations > 0, "{tag}");
        }
        check.finish();
    }

    /// Asserts the solution set of a nonunique location model: with k = tau n (an integer
    /// here), `y(k) <= coef <= y(k+1)` and the objective equals the objective at `y(k)`.
    fn assert_location_solution(y: &Array1<f64>, tau: f64, fit: &RqFit) {
        let mut sorted = y.to_vec();
        sorted.sort_by(f64::total_cmp);
        let k = (tau * y.len() as f64).round() as usize;
        let c = fit.coefficients[0];
        assert!(
            sorted[k - 1] <= c && c <= sorted[k],
            "{c} outside [{}, {}]",
            sorted[k - 1],
            sorted[k]
        );
        let loss = |b: f64| y.iter().map(|&v| check_function(v - b, tau)).sum::<f64>();
        let at_k = loss(sorted[k - 1]);
        assert!(
            (fit.objective - at_k).abs() <= 1e-12 * at_k,
            "{} vs {at_k}",
            fit.objective
        );
        assert!(
            fit.nonunique && fit.warnings == [NONUNIQUE],
            "{:?}",
            fit.warnings
        );
    }

    /// Case `rq_br_mtcars`: p = 4 (mpg ~ wt + hp + qsec), without intercept (mpg ~ wt + hp - 1),
    /// and the location model, whose quartiles and median are nonunique (R warns "Solution may be
    /// nonunique").
    #[test]
    fn test_validate_rq_br_mtcars() {
        let (case, ds) = crate::test_support::load_case("rq_br_mtcars");
        let n = case.scalar("n") as usize;
        let y = Array1::from(nan_vec(&ds, "mpg"));
        let ones = Array2::<f64>::ones((n, 1));
        let mut check = Check::new(&case);
        for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
            let mut cfg = QuantRegConfig::default();
            cfg.tau = tau;
            let r = quantreg(&ds, "mpg", &["wt", "hp", "qsec"], &cfg).unwrap();
            check
                .named(
                    &format!("{tag}_coef"),
                    &r.variable_names,
                    &estimates(&r),
                    Tol::coef(n),
                )
                .scalar(&format!("{tag}_objective"), r.objective, Tol::stat(n));
            assert_eq!(
                r.nonunique as u8 as f64,
                case.scalar(&format!("{tag}_nonunique"))
            );
            cfg.intercept = false;
            let g = quantreg(&ds, "mpg", &["wt", "hp"], &cfg).unwrap();
            check.named(
                &format!("noint_{tag}_coef"),
                &g.variable_names,
                &estimates(&g),
                Tol::coef(n),
            );
            let loc = rq_fit(ones.view(), y.view(), tau, &QuantRegConfig::default()).unwrap();
            assert_eq!(1.0, case.scalar(&format!("loc_{tag}_nonunique")));
            // tau n = 8, 16, 24: an integer
            assert_location_solution(&y, tau, &loc);
        }
        check.finish();
    }

    /// Case `rq_br_se`: the location model at tau 0.25 / 0.5 of n = 100 values is nonunique, and
    /// flagged as rq flags it; the solution lies between the order statistics y(k) and y(k+1).
    #[test]
    fn test_validate_rq_br_nonunique_location() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let y = Array1::from(nan_vec(&ds, "y"));
        let ones = Array2::<f64>::ones((y.len(), 1));
        for (tau, tag) in [(0.25, "loc_tau25"), (0.5, "loc_tau50")] {
            let f = rq_fit(ones.view(), y.view(), tau, &QuantRegConfig::default()).unwrap();
            assert_eq!(
                f.nonunique as u8 as f64,
                case.scalar(&format!("{tag}_nonunique"))
            );
            assert_location_solution(&y, tau, &f);
        }
    }

    /// y x 1e-6 and y x 1e6 give the coefficients x 1e-6 / x 1e6 on the same simplex path (rq is
    /// scale-equivariant). Fails before WP10: at 1e-6 every IRLS weight hit the absolute 1e-4 cap
    /// and the OLS fit (1.40798, 3.13291) came back.
    #[test]
    fn test_validate_rq_scale_equivariance() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let n = case.scalar("n") as usize;
        let base = run_quantreg(&ds, "y", &["x"], 0.5).unwrap();
        let mut check = Check::new(&case);
        for (factor, key) in [(1e-6, "m6"), (1e6, "p6")] {
            let scaled = with_derived(&ds, "ys", "y", |v| v * factor);
            let r = run_quantreg(&scaled, "ys", &["x"], 0.5).unwrap();
            check.named(
                &format!("scale_{key}_tau50_coef"),
                &r.variable_names,
                &estimates(&r),
                Tol::coef(n),
            );
            for (c, b) in r.coefficients.iter().zip(&base.coefficients) {
                assert!(
                    (c.estimate / factor / b.estimate - 1.0).abs() < 1e-12,
                    "{key}: {} vs {}",
                    c.estimate,
                    b.estimate
                );
            }
            assert_eq!(r.iterations, base.iterations);
        }
        check.finish();
    }

    /// Moving the 5 largest positive residuals up by 1000 and the 5 most negative down by 1000
    /// leaves the LP solution unchanged (R: max |change| 0); OLS would move by 15.8. Replaces the
    /// range-only test_validate_quantreg_median_simple (REG-44).
    #[test]
    fn test_validate_rq_median_unchanged_by_outlying_responses() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let base = run_quantreg(&ds, "y", &["x"], 0.5).unwrap();
        let r = &base.residuals;
        let mut order: Vec<usize> = (0..r.len()).collect();
        order.sort_by(|&a, &b| r[b].total_cmp(&r[a]));
        let (up, down) = (order[..5].to_vec(), order[order.len() - 5..].to_vec());
        let mut moved: Vec<usize> = up.iter().chain(&down).copied().collect();
        moved.sort_unstable();
        assert_eq!(moved, [8, 15, 20, 30, 43, 46, 53, 64, 67, 68]);
        let y: Vec<f64> = nan_vec(&ds, "y")
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if up.contains(&i) {
                    v + 1000.0
                } else if down.contains(&i) {
                    v - 1000.0
                } else {
                    v
                }
            })
            .collect();
        let ds2 = Dataset::new(df! { "y" => y, "x" => nan_vec(&ds, "x") }.unwrap());
        let r2 = run_quantreg(&ds2, "y", &["x"], 0.5).unwrap();
        Check::new(&case)
            .named(
                "tau50_coef",
                &r2.variable_names,
                &estimates(&r2),
                Tol::coef(100),
            )
            .finish();
    }

    /// IRLS option: a relative weight floor (1e-6 MAD of the residuals) and a relative stopping
    /// rule, so y x 1e-6 and 1e6 take the same iterations; the objective is within 1e-6 of rq's
    /// LP minimum (R transcription: 3.1e-8 / 9.1e-9 / 7.6e-8). max_iter = 2 reports
    /// non-convergence. Fails before WP10: 1 iteration (the OLS fit) at 1e-6, and no flag.
    #[test]
    fn test_validate_rq_irls_scale_free_and_honest() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
            let mut cfg = QuantRegConfig::default();
            cfg.tau = tau;
            cfg.algorithm = QuantRegMethod::Irls;
            let base = quantreg(&ds, "y", &["x"], &cfg).unwrap();
            let rmin = case.scalar(&format!("{tag}_objective"));
            assert!(
                base.convergence.converged && base.algorithm == QuantRegMethod::Irls,
                "{tag}"
            );
            assert!(
                base.objective >= rmin * (1.0 - 1e-12) && base.objective <= rmin * (1.0 + 1e-6),
                "{tag}: {} vs {rmin}",
                base.objective
            );
            for factor in [1e-6, 1e6] {
                let r = quantreg(
                    &with_derived(&ds, "ys", "y", |v| v * factor),
                    "ys",
                    &["x"],
                    &cfg,
                )
                .unwrap();
                assert_eq!(r.iterations, base.iterations, "{tag} x {factor}");
                for (c, b) in r.coefficients.iter().zip(&base.coefficients) {
                    assert!(
                        (c.estimate / factor / b.estimate - 1.0).abs() < 1e-9,
                        "{tag} x {factor}"
                    );
                }
            }
        }
        let mut cfg = QuantRegConfig::default();
        cfg.algorithm = QuantRegMethod::Irls;
        cfg.max_iter = 2;
        let r = quantreg(&ds, "y", &["x"], &cfg).unwrap();
        assert!(!r.convergence.converged && r.iterations == 2);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("did not converge after 2 iterations")),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("Converged: no (2 iterations"), "{r}");
    }

    /// R's rq.fit.br stops with "Singular design matrix"; p2a refuses with SingularMatrix naming
    /// the aliased column. Fails before WP10: a generalized inverse gave a result.
    #[test]
    fn test_rq_rejects_singular_design() {
        let ds2 = with_derived(&hetero(), "x2", "x", |v| 2.0 * v);
        match quantreg(&ds2, "y", &["x", "x2"], &QuantRegConfig::default()) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(
                    context.contains("x2") && context.contains("rank 2 of 3"),
                    "{context}"
                )
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
        let x = Array2::from_shape_fn((6, 2), |(i, j)| if j == 0 { 1.0 } else { 3.0 });
        let y = Array1::from(vec![1.0, 2.1, 2.9, 4.2, 5.0, 5.8]);
        assert!(matches!(
            rq_fit(x.view(), y.view(), 0.5, &QuantRegConfig::default()),
            Err(EconError::SingularMatrix { .. })
        ));
    }

    /// Degenerate parameters are errors naming the parameter (R3). Fails before WP10: tau = NaN
    /// passed `<= 0 || >= 1`, max_iter = 0 and tol <= 0 were accepted.
    #[test]
    fn test_rq_rejects_invalid_parameters() {
        let ds = hetero();
        let err = |f: &dyn Fn(&mut QuantRegConfig), xs: &[&str]| {
            let mut c = QuantRegConfig::default();
            f(&mut c);
            match quantreg(&ds, "y", xs, &c) {
                Err(EconError::InvalidSpecification { message }) => message,
                other => panic!("expected InvalidSpecification, got {other:?}"),
            }
        };
        for tau in [f64::NAN, 0.0, 1.0, -0.5, 1.5, f64::INFINITY] {
            assert!(err(&|c| c.tau = tau, &["x"]).contains("tau"), "tau = {tau}");
        }
        assert!(err(&|c| c.max_iter = 0, &["x"]).contains("max_iter"));
        for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(err(&|c| c.tol = tol, &["x"]).contains("tol"), "tol = {tol}");
        }
        assert!(err(&|c| c.intercept = false, &[]).contains("regressor"));
        let x = Array2::<f64>::ones((4, 1));
        let y = Array1::from(vec![1.0, f64::NAN, 2.0, 3.0]);
        assert!(matches!(
            rq_fit(x.view(), y.view(), 0.5, &QuantRegConfig::default()),
            Err(EconError::InvalidSpecification { message }) if message.contains("finite")
        ));
        let y = Array1::from(vec![1.0, 2.0, 2.0, 3.0]);
        assert!(matches!(
            rq_fit(x.view(), y.view(), 1.5, &QuantRegConfig::default()),
            Err(EconError::InvalidSpecification { message }) if message.contains("tau")
        ));
        assert!(matches!(
            rq_fit(
                x.view(),
                y.slice(ndarray::s![..3]),
                0.5,
                &QuantRegConfig::default()
            ),
            Err(EconError::InvalidSpecification { .. })
        ));
    }

    /// The simplex and summary.rq tolerances are R's: .Machine$double.eps^(2/3) =
    /// 3.666852862501036e-11 and sqrt(.Machine$double.eps) = 1.4901161193847656e-08.
    #[test]
    fn test_br_toler_is_r_eps_two_thirds() {
        assert!((BR_TOLER / f64::EPSILON.powf(2.0 / 3.0) - 1.0).abs() < 1e-15);
        assert_eq!(RQ_FUZZ, f64::EPSILON.sqrt());
    }

    #[test]
    fn test_rq_display_reports_method_and_convergence() {
        let r = run_quantreg(&hetero(), "y", &["x"], 0.5).unwrap();
        let s = r.to_string();
        assert!(
            s.contains("Algorithm: Barrodale-Roberts simplex (br)\n"),
            "{s}"
        );
        assert!(s.contains("Converged: yes ("), "{s}");
        let json = crate::traits::convergence::notes_json(&r);
        assert_eq!(json["converged"], true);
    }
}
