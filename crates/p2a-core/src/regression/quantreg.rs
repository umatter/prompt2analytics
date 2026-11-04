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
//! - **Barrodale-Roberts simplex** (`br`): a port of quantreg's `rqbr` (Koenker and d'Orey 1987,
//!   AS 229, a modification of Barrodale and Roberts 1974) for one τ. It returns a vertex of the LP,
//!   the solution `rq(method = "br")` returns, and flags a nonunique solution as R does ("Solution
//!   may be nonunique"). Its cost grows about quadratically in n.
//! - **Frisch-Newton interior point** (`fn`): a port of quantreg's `rqfnb` (Portnoy and Koenker 1997),
//!   a primal-dual log-barrier method with Mehrotra's predictor-corrector, linear in n. It runs on
//!   `y / s` with `s = mean |y - median(y)|`, so its duality-gap rule is scale-free; it stops within
//!   that gap of the LP optimum, not at a vertex.
//! - **Automatic** (`auto`, the default): `br` for n <= 5000 and `fn` above, quantreg's own advice
//!   for large problems. The result names the algorithm that ran.
//! - **IRLS** (`irls`): iteratively reweighted least squares with a relative weight floor
//!   (1e-6 times the residual MAD). An approximation that no R method computes; its objective is
//!   close to, but not at, the LP minimum.
//!
//! # Standard errors ([`QuantRegSe`], `summary.rq(se = )`)
//!
//! - `nid` (default, at every n): the Hendricks-Koenker sandwich, with the density from fits at
//!   τ ± h (Hall-Sheather bandwidth). R's default for n < 1001 is `se = "rank"` (rank-inversion
//!   intervals), which is not computed here.
//! - `iid`: the Koenker-Bassett sparsity estimate.
//! - `ker`: Powell's kernel sandwich.
//! - `boot`: the xy-pair bootstrap (`boot.rq(bsmethod = "xy")`), seeded.
//!
//! summary.rq's absolute fuzz `sqrt(.Machine$double.eps)` is multiplied by `mean |y - median(y)|`
//! here, so every standard error is scale-equivariant. t, p and the 95% interval use t(n - k).
//! `validation/regression/quantreg.md` lists every difference from R.
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
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::cholesky;
use crate::linalg::solve::{LM_QR_TOL, Qr, least_squares, weighted_least_squares};
use crate::options::Choices;
use crate::resource;
use crate::stats::robust::{QuantileType, quantile_sorted};
use crate::traits::convergence::{Convergence, ResultNotes, notes_text, relative_change_vec};
use crate::traits::estimator::{normal_quantile, t_quantile, t_test_p_value, validate_level};

/// `.Machine$double.eps^(2/3)`: the pivot tolerance of quantreg's `rq.fit.br`.
const BR_TOLER: f64 = 3.666852862501036e-11;
/// `sqrt(.Machine$double.eps)`: summary.rq's fuzz `eps` (multiplied here by the outcome's
/// dispersion, see [`rq_se`]).
const RQ_FUZZ: f64 = 1.4901161193847656e-8;
/// `QuantRegMethod::Auto` uses the simplex up to this many observations.
const AUTO_BR_MAX_N: usize = 5000;
/// Iteration limit of the Frisch-Newton method (quantreg's `rq.fit.fnb`).
const FN_MAX_ITER: usize = 500;
/// Step-length factor of the Frisch-Newton method (quantreg's `beta`).
const FN_BETA: f64 = 0.99995;
/// summary.rq(se = "boot")'s default number of replications.
const BOOT_DEFAULT_REPLICATIONS: usize = 200;
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
    if config.se != QuantRegSe::Boot && config.bootstrap_samples > 0 {
        return Err(spec(format!(
            "bootstrap_samples = {} is used only with se = boot",
            config.bootstrap_samples
        )));
    }
    if config.se == QuantRegSe::Boot && config.bootstrap_samples == 1 {
        return Err(spec(
            "bootstrap_samples must be at least 2 (or 0 for R's default 200)".into(),
        ));
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
    let mut warnings = fit.warnings.clone();

    let se = match rq_se(x.view(), y.view(), config.tau, &fit, config) {
        Ok(se) => se,
        Err(e @ (EconError::InsufficientData { .. } | EconError::SingularMatrix { .. })) => {
            warnings.push(format!(
                "standard errors (se = {}) could not be computed: {e}",
                config.se
            ));
            RqSe::unavailable(config.se, k)
        }
        Err(e) => return Err(e),
    };
    warnings.extend(se.warnings.iter().cloned());

    let q = t_quantile(0.975, df as f64)?;
    let coefficients: Vec<QuantRegCoefficient> = (0..k)
        .map(|i| {
            let b = fit.coefficients[i];
            let s = se.std_errors[i];
            let t = b / s;
            QuantRegCoefficient {
                name: var_names[i].clone(),
                estimate: b,
                std_error: s,
                t_value: t,
                p_value: t_test_p_value(t, df as f64),
                ci_lower_95: b - q * s,
                ci_upper_95: b + q * s,
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
        se_type: se.se_type,
        se_method: se.method,
        se_bandwidth: se.bandwidth,
        sparsity: se.sparsity,
        kernel_bandwidth: se.kernel_bandwidth,
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
fn resolve_method(method: QuantRegMethod, n: usize, tau: f64, tol: f64) -> QuantRegMethod {
    match method {
        QuantRegMethod::Auto if n > AUTO_BR_MAX_N && tau > tol && tau < 1.0 - tol => {
            QuantRegMethod::FrischNewton
        }
        QuantRegMethod::Auto => QuantRegMethod::BarrodaleRoberts,
        m => m,
    }
}

/// The algorithm of an auxiliary fit at `t` (nid's fits at tau -/+ h, the bootstrap's replicates)
/// of a fit that ran `fit_algorithm`: Frisch-Newton only where it can run (`tol < t < 1 - tol`),
/// the simplex otherwise (and for a simplex or IRLS fit).
fn aux_method(fit_algorithm: QuantRegMethod, t: f64, tol: f64) -> QuantRegMethod {
    match fit_algorithm {
        QuantRegMethod::FrischNewton if t > tol && t < 1.0 - tol => QuantRegMethod::FrischNewton,
        _ => QuantRegMethod::BarrodaleRoberts,
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
            let (b, c) = frisch_newton(x, y, tau, config.tol)?;
            warnings.extend(c.note("Frisch-Newton quantile regression"));
            (b, c.iterations, false, c)
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

/// Solve `L L' v = b` in place (`l` lower triangular).
fn chol_solve(l: &Array2<f64>, b: &mut Array1<f64>) {
    let p = b.len();
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[[i, k]] * b[k];
        }
        b[i] = s / l[[i, i]];
    }
    for i in (0..p).rev() {
        let mut s = b[i];
        for k in i + 1..p {
            s -= l[[k, i]] * b[k];
        }
        b[i] = s / l[[i, i]];
    }
}

/// The Cholesky factor of `X' D X`.
fn weighted_gram_cholesky(x: ArrayView2<'_, f64>, d: &Array1<f64>) -> EconResult<Array2<f64>> {
    let p = x.ncols();
    let mut ada = Array2::<f64>::zeros((p, p));
    for (row, &di) in x.rows().into_iter().zip(d.iter()) {
        for j in 0..p {
            let v = di * row[j];
            for k in 0..=j {
                ada[[j, k]] += v * row[k];
            }
        }
    }
    for j in 0..p {
        for k in 0..j {
            ada[[k, j]] = ada[[j, k]];
        }
    }
    cholesky(&ada.view()).map_err(|_| EconError::SingularMatrix {
        context: "Frisch-Newton: X'DX is not positive definite (possibly singular design)".into(),
        suggestion: "use method = br".into(),
    })
}

/// The largest step in (0, 1e20] keeping `v + step * dv` positive (`min(-v / dv)` over `dv <
/// 0`), for two vectors at once.
fn max_step(v1: &Array1<f64>, d1: &Array1<f64>, v2: &Array1<f64>, d2: &Array1<f64>) -> f64 {
    let mut m = 1e20f64;
    for (v, d) in [(v1, d1), (v2, d2)] {
        for (&vi, &di) in v.iter().zip(d.iter()) {
            if di < 0.0 {
                m = m.min(-vi / di);
            }
        }
    }
    m
}

/// quantreg's `rqfnb` (Portnoy and Koenker 1997): the primal-dual log-barrier method with
/// Mehrotra's predictor-corrector for the dual of the quantile-regression LP, run on `y / s`
/// with `s = mean |y - median(y)|` so that the duality-gap rule `gap <= tol` is scale-free
/// (quantreg's absolute rule stops early on small-scale outcomes). The coefficients are
/// rescaled by `s`.
fn frisch_newton(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau: f64,
    tol: f64,
) -> EconResult<(Array1<f64>, Convergence)> {
    if !(tau > tol && tau < 1.0 - tol) {
        return Err(spec(format!(
            "method fn needs tol < tau < 1 - tol (quantreg: \"No parametric Frisch-Newton \
             method\"), got tau = {tau}"
        )));
    }
    let n = x.nrows();
    let scale = dispersion(y);
    let eps = tol;
    let xt = x.t();
    // LP data: A = X', c = -y / s, b = (1 - tau) X'1, upper bound u = 1
    let c: Array1<f64> = y.mapv(|v| -v / scale);
    let b: Array1<f64> = xt.dot(&Array1::<f64>::ones(n)) * (1.0 - tau);
    let mut xv = Array1::from_elem(n, 1.0 - tau);
    let mut d = Array1::<f64>::ones(n);
    let l0 = weighted_gram_cholesky(x, &d)?;
    let mut yd = xt.dot(&c);
    chol_solve(&l0, &mut yd);
    let r0 = &c - &x.dot(&yd);
    let mut z = r0.mapv(|v| {
        if v.abs() < eps {
            v.max(0.0) + eps
        } else {
            v.max(0.0)
        }
    });
    let mut w = r0.mapv(|v| {
        if v.abs() < eps {
            (-v).max(0.0) + eps
        } else {
            (-v).max(0.0)
        }
    });
    let mut sv = xv.mapv(|v| 1.0 - v);
    let gap_of = |z: &Array1<f64>, xv: &Array1<f64>, w: &Array1<f64>, sv: &Array1<f64>| {
        z.dot(xv) + w.dot(sv)
    };
    let mut gap = gap_of(&z, &xv, &w, &sv);
    let mut it = 0usize;
    while gap > eps && it < FN_MAX_ITER {
        resource::checkpoint()?;
        it += 1;
        for i in 0..n {
            d[i] = 1.0 / (z[i] / xv[i] + w[i] / sv[i]);
        }
        let mut ds = &z - &w;
        let dz = &d * &ds;
        let mut dy = &b - &xt.dot(&(&xv - &dz));
        let mut rhs = dy.clone();
        let l = weighted_gram_cholesky(x, &d)?;
        chol_solve(&l, &mut dy);
        ds = &x.dot(&dy) - &ds;
        let mut dx = &d * &ds;
        let mut ds = -&dx;
        let mut dz: Array1<f64> = Array1::from_iter((0..n).map(|i| -z[i] * (dx[i] / xv[i] + 1.0)));
        let mut dw: Array1<f64> = Array1::from_iter((0..n).map(|i| -w[i] * (ds[i] / sv[i] + 1.0)));
        let mut deltap = (FN_BETA * max_step(&xv, &dx, &sv, &ds)).min(1.0);
        let mut deltad = (FN_BETA * max_step(&z, &dz, &w, &dw)).min(1.0);
        if deltap.min(deltad) < 1.0 {
            // Mehrotra's corrector
            let mut mu = xv.dot(&z) + sv.dot(&w);
            let g = mu
                + deltap * dx.dot(&z)
                + deltad * dz.dot(&xv)
                + deltap * deltad * dz.dot(&dx)
                + deltap * ds.dot(&w)
                + deltad * dw.dot(&sv)
                + deltap * deltad * ds.dot(&dw);
            mu = mu * (g / mu).powi(3) / (2.0 * n as f64);
            let dr: Array1<f64> = Array1::from_iter((0..n).map(|i| {
                d[i] * (mu * (1.0 / sv[i] - 1.0 / xv[i]) + dx[i] * dz[i] / xv[i]
                    - ds[i] * dw[i] / sv[i])
            }));
            std::mem::swap(&mut rhs, &mut dy);
            dy = &dy + &xt.dot(&dr);
            chol_solve(&l, &mut dy);
            let uu = x.dot(&dy);
            let dxdz = &dx * &dz;
            let dsdw = &ds * &dw;
            dx = Array1::from_iter((0..n).map(|i| d[i] * (uu[i] - z[i] + w[i]) - dr[i]));
            ds = -&dx;
            dz = Array1::from_iter((0..n).map(|i| -z[i] + (mu - z[i] * dx[i] - dxdz[i]) / xv[i]));
            dw = Array1::from_iter((0..n).map(|i| -w[i] + (mu - w[i] * ds[i] - dsdw[i]) / sv[i]));
            deltap = (FN_BETA * max_step(&xv, &dx, &sv, &ds)).min(1.0);
            deltad = (FN_BETA * max_step(&z, &dz, &w, &dw)).min(1.0);
        }
        xv.scaled_add(deltap, &dx);
        sv.scaled_add(deltap, &ds);
        yd.scaled_add(deltad, &dy);
        z.scaled_add(deltad, &dz);
        w.scaled_add(deltad, &dw);
        gap = gap_of(&z, &xv, &w, &sv);
    }
    let coef = yd.mapv(|v| -v * scale);
    let convergence = Convergence::new(
        gap <= eps,
        it,
        FN_MAX_ITER,
        "duality gap of the standardised LP (rqfnb)",
        gap,
        eps,
    );
    Ok((coef, convergence))
}

/// The standard normal density (R `dnorm`).
fn dnorm(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// quantreg::bandwidth.rq(tau, n, hs = TRUE, alpha = 0.05): the Hall-Sheather bandwidth
/// `n^(-1/3) qnorm(0.975)^(2/3) (1.5 phi(x0)^2 / (2 x0^2 + 1))^(1/3)`, `x0 = qnorm(tau)`.
pub fn hall_sheather_bandwidth(tau: f64, n: usize) -> EconResult<f64> {
    validate_level(tau, "tau")?;
    if n == 0 {
        return Err(spec(
            "the Hall-Sheather bandwidth needs at least one observation".into(),
        ));
    }
    let x0 = normal_quantile(tau)?;
    let f0 = dnorm(x0);
    let z = normal_quantile(0.975)?;
    Ok((n as f64).powf(-1.0 / 3.0)
        * z.powf(2.0 / 3.0)
        * ((1.5 * f0 * f0) / (2.0 * x0 * x0 + 1.0)).powf(1.0 / 3.0))
}

/// summary.rq's bandwidth for nid and ker: [`hall_sheather_bandwidth`] halved until
/// tau -/+ h lies in [0, 1].
fn halved_bandwidth(tau: f64, n: usize) -> EconResult<f64> {
    let mut h = hall_sheather_bandwidth(tau, n)?;
    while tau - h < 0.0 || tau + h > 1.0 {
        h /= 2.0;
    }
    Ok(h)
}

/// Standard errors of a quantile-regression fit; see [`rq_se`].
#[derive(Debug, Clone)]
pub struct RqSe {
    /// The type computed
    pub se_type: QuantRegSe,
    /// `sqrt(diag(vcov))`
    pub std_errors: Array1<f64>,
    /// Covariance of the coefficients
    pub vcov: Array2<f64>,
    /// Hall-Sheather bandwidth (iid: unhalved; nid, ker: halved; boot: `None`)
    pub bandwidth: Option<f64>,
    /// iid: the sparsity estimate (R's `1 / scale`)
    pub sparsity: Option<f64>,
    /// ker: Powell's kernel bandwidth
    pub kernel_bandwidth: Option<f64>,
    /// boot: replications kept
    pub replications: Option<usize>,
    /// How the standard errors were computed
    pub method: String,
    /// Warnings (R's "non-positive fis", nonunique auxiliary fits, dropped replications)
    pub warnings: Vec<String>,
}

impl RqSe {
    /// NaN standard errors of `se_type` (they could not be computed).
    fn unavailable(se_type: QuantRegSe, p: usize) -> Self {
        Self {
            se_type,
            std_errors: Array1::from_elem(p, f64::NAN),
            vcov: Array2::from_elem((p, p), f64::NAN),
            bandwidth: None,
            sparsity: None,
            kernel_bandwidth: None,
            replications: None,
            method: format!("{se_type} (not available)"),
            warnings: Vec::new(),
        }
    }
}

/// `tau (1 - tau) H^-1 X'X H^-1` with `H = (sqrt(f) X)'(sqrt(f) X)`: the sandwich of nid and ker.
fn density_sandwich(
    x: ArrayView2<'_, f64>,
    f: &Array1<f64>,
    tau: f64,
    what: &str,
) -> EconResult<Array2<f64>> {
    let p = x.ncols();
    let mut fx = x.to_owned();
    for (mut row, &fi) in fx.rows_mut().into_iter().zip(f.iter()) {
        row *= fi.sqrt();
    }
    let qr = Qr::decompose(fx.view(), LM_QR_TOL)?;
    if qr.rank() < p {
        return Err(EconError::SingularMatrix {
            context: format!(
                "{what} sandwich: sqrt(f) X has rank {} of {p} (too few observations with a \
                 positive density estimate)",
                qr.rank()
            ),
            suggestion: "use se = iid or se = boot".into(),
        });
    }
    let hinv = qr.unscaled_covariance()?;
    let xx = x.t().dot(&x);
    Ok(hinv.dot(&xx).dot(&hinv) * (tau * (1.0 - tau)))
}

/// The standard errors of `fit` as `quantreg::summary.rq(se = config.se)`:
///
/// - `Iid`: `sparsity^2 tau (1 - tau) (X'X)^-1`, the sparsity being the slope of the simplex
///   fit of the ordered residuals (h + 1 of them around the zero residuals, h from the
///   Hall-Sheather bandwidth) on their ranks / (n - p);
/// - `Nid`: the Hendricks-Koenker sandwich, the density `2h / (x (b(tau + h) - b(tau - h)) -
///   eps)` from fits with the fit's algorithm (the simplex for an IRLS fit); "k non-positive
///   fis" as R;
/// - `Ker`: Powell's sandwich with the kernel bandwidth `(qnorm(tau + h) - qnorm(tau - h))
///   min(sd(u), IQR(u) / 1.34)`;
/// - `Boot`: the xy-pair bootstrap with `config.bootstrap_samples` replications (0: 200),
///   resampled with ChaCha8 from `config.seed` (stream = replication index).
///
/// summary.rq's fuzz `eps = sqrt(.Machine$double.eps)` is multiplied by `mean |y -
/// median(y)|` so that the iid and nid standard errors are scale-equivariant. Errors:
/// `InsufficientData` (iid: fewer than h + pz + 1 observations, where R stops), `SingularMatrix`
/// (a singular density-weighted design).
pub fn rq_se(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau: f64,
    fit: &RqFit,
    config: &QuantRegConfig,
) -> EconResult<RqSe> {
    let (n, p) = x.dim();
    if y.len() != n || fit.coefficients.len() != p || fit.residuals.len() != n {
        return Err(spec(format!(
            "rq_se: x is {n} x {p}, y has {} rows and the fit {} coefficients and {} residuals",
            y.len(),
            fit.coefficients.len(),
            fit.residuals.len()
        )));
    }
    validate_level(tau, "tau")?;
    validate_config(config)?;
    let eps = RQ_FUZZ * dispersion(y);
    let mut warnings = Vec::new();
    let se_type = config.se;
    let mut out = RqSe::unavailable(se_type, p);
    let vcov = match se_type {
        QuantRegSe::Iid => {
            let xxinv = Qr::decompose(x, LM_QR_TOL)?.unscaled_covariance()?;
            let resid = &fit.residuals;
            let pz = resid.iter().filter(|r| r.abs() < eps).count();
            let hs = hall_sheather_bandwidth(tau, n)?;
            let h = ((n as f64 * hs).ceil() as usize).max(p + 1);
            if h + pz + 1 > n {
                return Err(EconError::InsufficientData {
                    required: h + pz + 1,
                    provided: n,
                    context: "iid sparsity estimate (summary.rq)".into(),
                });
            }
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| resid[a].abs().total_cmp(&resid[b].abs()));
            let mut ord_resid: Vec<f64> = order[pz..=h + pz].iter().map(|&i| resid[i]).collect();
            ord_resid.sort_by(f64::total_cmp);
            let m = ord_resid.len();
            let design = Array2::from_shape_fn((m, 2), |(j, c)| {
                if c == 0 {
                    1.0
                } else {
                    (pz + 1 + j) as f64 / (n - p) as f64
                }
            });
            let sp = br_simplex(design.view(), ArrayView1::from(&ord_resid[..]), 0.5)?;
            if sp.nonunique {
                warnings.push(format!("{NONUNIQUE} (iid sparsity fit)"));
            }
            let sparsity = sp.coef[1];
            out.bandwidth = Some(hs);
            out.sparsity = Some(sparsity);
            out.method = format!("iid (Koenker-Bassett sparsity, Hall-Sheather bandwidth {hs:.4})");
            xxinv * (sparsity * sparsity) * tau * (1.0 - tau)
        }
        QuantRegSe::Nid => {
            let h = halved_bandwidth(tau, n)?;
            let hi = fit_full_rank(
                x,
                y,
                tau + h,
                aux_method(fit.algorithm, tau + h, config.tol),
                config,
            )?;
            let lo = fit_full_rank(
                x,
                y,
                tau - h,
                aux_method(fit.algorithm, tau - h, config.tol),
                config,
            )?;
            for (f, t) in [(&hi, tau + h), (&lo, tau - h)] {
                if f.nonunique {
                    warnings.push(format!("{NONUNIQUE} (nid fit at tau = {t:.4})"));
                }
                warnings.extend(f.warnings.iter().filter(|w| *w != NONUNIQUE).cloned());
            }
            let dyhat = x.dot(&(&hi.coefficients - &lo.coefficients));
            let k = dyhat.iter().filter(|&&v| v <= 0.0).count();
            if k > 0 {
                warnings.push(format!("{k} non-positive fis"));
            }
            let f = dyhat.mapv(|v| {
                let den = v - eps;
                if den > 0.0 { 2.0 * h / den } else { 0.0 }
            });
            out.bandwidth = Some(h);
            out.method =
                format!("nid (Hendricks-Koenker sandwich, Hall-Sheather bandwidth {h:.4})");
            density_sandwich(x, &f, tau, "nid")?
        }
        QuantRegSe::Ker => {
            let h = halved_bandwidth(tau, n)?;
            let u = &fit.residuals;
            let mean = u.sum() / n as f64;
            let sd = (u.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
            let mut sorted = u.to_vec();
            sorted.sort_by(f64::total_cmp);
            let iqr = quantile_sorted(&sorted, 0.75, QuantileType::T7)
                - quantile_sorted(&sorted, 0.25, QuantileType::T7);
            let hk = (normal_quantile(tau + h)? - normal_quantile(tau - h)?) * sd.min(iqr / 1.34);
            if !(hk.is_finite() && hk > 0.0) {
                return Err(EconError::SingularMatrix {
                    context: format!(
                        "ker standard errors: Powell's kernel bandwidth is {hk} because the \
                         residual scale min(sd, IQR / 1.34) is 0 (at least half the residuals \
                         are tied)"
                    ),
                    suggestion: "use se = nid, se = iid or se = boot".into(),
                });
            }
            let f = u.mapv(|v| dnorm(v / hk) / hk);
            out.bandwidth = Some(h);
            out.kernel_bandwidth = Some(hk);
            out.method = format!("ker (Powell kernel sandwich, bandwidth {hk:.4})");
            density_sandwich(x, &f, tau, "ker")?
        }
        QuantRegSe::Boot => {
            let reps = if config.bootstrap_samples == 0 {
                BOOT_DEFAULT_REPLICATIONS
            } else {
                config.bootstrap_samples
            };
            let seed = config.seed.unwrap_or(QUANTREG_BOOT_DEFAULT_SEED);
            let method = aux_method(fit.algorithm, tau, config.tol);
            let draws = bootstrap_xy_coefficients(x, y, tau, reps, method, config, |r, idx| {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                rng.set_stream(r as u64);
                idx.clear();
                idx.extend((0..n).map(|_| rng.gen_range(0..n)));
            })?;
            if draws.dropped > 0 {
                warnings.push(format!(
                    "{} of {reps} bootstrap replications had a singular design and were dropped",
                    draws.dropped
                ));
            }
            if draws.nonunique > 0 {
                warnings.push(format!(
                    "{} of {reps} bootstrap fits may be nonunique",
                    draws.nonunique
                ));
            }
            out.replications = Some(draws.coefficients.nrows());
            out.method = format!("boot (xy-pair bootstrap, R = {reps}, seed {seed})");
            bootstrap_se(&draws.coefficients).1
        }
    };
    out.std_errors = vcov.diag().mapv(f64::sqrt);
    out.vcov = vcov;
    out.warnings = warnings;
    Ok(out)
}

/// Bootstrap replicate coefficients (kept replications by row).
pub(crate) struct BootstrapDraws {
    /// kept replications x p
    pub coefficients: Array2<f64>,
    /// replications with a singular resampled design
    pub dropped: usize,
    /// kept simplex fits flagged nonunique
    pub nonunique: usize,
}

/// The xy-pair bootstrap of `boot.rq(bsmethod = "xy")`: replication `r` refits on the rows
/// `draw(r, idx)` writes into `idx` (one reused buffer). A resample whose design is singular is
/// skipped and counted (R keeps rq0's output and warns).
pub(crate) fn bootstrap_xy_coefficients(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    tau: f64,
    replications: usize,
    method: QuantRegMethod,
    config: &QuantRegConfig,
    mut draw: impl FnMut(usize, &mut Vec<usize>),
) -> EconResult<BootstrapDraws> {
    let (n, p) = x.dim();
    let mut b = resource::try_zeros2(replications, p, "quantreg bootstrap draws")?;
    let mut xb = resource::try_zeros2(n, p, "quantreg bootstrap resample")?;
    let mut yb = resource::try_zeros1(n, "quantreg bootstrap resample")?;
    let mut idx: Vec<usize> = Vec::with_capacity(n);
    let (mut kept, mut dropped, mut nonunique) = (0usize, 0usize, 0usize);
    for r in 0..replications {
        resource::checkpoint()?;
        draw(r, &mut idx);
        if idx.len() != n || idx.iter().any(|&i| i >= n) {
            return Err(EconError::Computation(format!(
                "bootstrap replication {r}: {} indices for {n} rows",
                idx.len()
            )));
        }
        for (k, &i) in idx.iter().enumerate() {
            xb.row_mut(k).assign(&x.row(i));
            yb[k] = y[i];
        }
        if Qr::decompose(xb.view(), LM_QR_TOL)?.rank() < p {
            dropped += 1;
            continue;
        }
        let fit = fit_full_rank(xb.view(), yb.view(), tau, method, config)?;
        nonunique += fit.nonunique as usize;
        b.row_mut(kept).assign(&fit.coefficients);
        kept += 1;
    }
    if kept < 2 {
        return Err(EconError::InsufficientData {
            required: 2,
            provided: kept,
            context: "bootstrap replications with a non-singular design".into(),
        });
    }
    Ok(BootstrapDraws {
        coefficients: b.slice(ndarray::s![..kept, ..]).to_owned(),
        dropped,
        nonunique,
    })
}

/// `sqrt(diag(cov(draws)))` and `cov(draws)` (denominator R - 1), as summary.rq(se = "boot").
pub(crate) fn bootstrap_se(draws: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let (r, p) = draws.dim();
    let mean: Array1<f64> = (0..p).map(|j| draws.column(j).sum() / r as f64).collect();
    let mut cov = Array2::<f64>::zeros((p, p));
    for row in draws.rows() {
        for j in 0..p {
            let dj = row[j] - mean[j];
            for k in 0..p {
                cov[[j, k]] += dj * (row[k] - mean[k]);
            }
        }
    }
    cov /= (r as f64 - 1.0).max(1.0);
    (cov.diag().mapv(f64::sqrt), cov)
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

/// Quantile regressions at several quantiles with default settings; see
/// [`quantreg_multi_with`].
pub fn quantreg_multi(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    taus: &[f64],
) -> EconResult<Vec<QuantRegResult>> {
    quantreg_multi_with(dataset, y_col, x_cols, taus, &QuantRegConfig::default())
}

/// One fit per tau with `config` (its `tau` replaced); an empty `taus` is an error.
pub fn quantreg_multi_with(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    taus: &[f64],
    config: &QuantRegConfig,
) -> EconResult<Vec<QuantRegResult>> {
    if taus.is_empty() {
        return Err(spec("taus must contain at least one quantile".into()));
    }
    let mut results = Vec::with_capacity(taus.len());
    for &tau in taus {
        let mut cfg = config.clone();
        cfg.tau = tau;
        results.push(quantreg(dataset, y_col, x_cols, &cfg)?);
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
            let se = |r: &QuantRegResult| {
                r.coefficients
                    .iter()
                    .map(|c| c.std_error)
                    .collect::<Vec<_>>()
            };
            check.named(
                &format!("se_nid_{tag}"),
                &r.variable_names,
                &se(&r),
                Tol::se(n),
            );
            let mut cfg = QuantRegConfig::default();
            cfg.tau = tau;
            cfg.se = QuantRegSe::Iid;
            let ri = quantreg(&ds, "y", &["x"], &cfg).unwrap();
            check.named(
                &format!("se_iid_{tag}"),
                &ri.variable_names,
                &se(&ri),
                Tol::se(n),
            );
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

    // ── summary.rq standard errors (REG-26) and t(n - k) intervals (REG-64) ──

    fn table(r: &QuantRegResult) -> [Vec<f64>; 5] {
        let col =
            |f: fn(&QuantRegCoefficient) -> f64| r.coefficients.iter().map(f).collect::<Vec<_>>();
        [
            col(|c| c.std_error),
            col(|c| c.t_value),
            col(|c| c.p_value),
            col(|c| c.ci_lower_95),
            col(|c| c.ci_upper_95),
        ]
    }

    /// summary.rq(se = <se>) on rq_br_se (n = 100) and rq_br_mtcars (n = 32) at tau
    /// 0.25/0.5/0.75: SE, t, p and the t(n - p) 95% interval. Returns the fits per case id and tag.
    fn check_summary_se(se: QuantRegSe) -> Vec<(&'static str, &'static str, QuantRegResult)> {
        let key = se.to_string();
        let mut out = Vec::new();
        for (id, y, xs) in [
            ("rq_br_se", "y", &["x"][..]),
            ("rq_br_mtcars", "mpg", &["wt", "hp", "qsec"][..]),
        ] {
            let (case, ds) = crate::test_support::load_case(id);
            let n = case.scalar("n") as usize;
            let mut check = Check::new(&case);
            for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
                let mut cfg = QuantRegConfig::default();
                cfg.tau = tau;
                cfg.se = se;
                let r = quantreg(&ds, y, xs, &cfg).unwrap();
                let [s, t, p, lo, hi] = table(&r);
                let names = &r.variable_names;
                check
                    .named(&format!("{tag}_se_{key}"), names, &s, Tol::se(n))
                    .named(&format!("{tag}_t_{key}"), names, &t, Tol::stat(n))
                    .named(&format!("{tag}_ci_lower_{key}"), names, &lo, Tol::stat(n))
                    .named(&format!("{tag}_ci_upper_{key}"), names, &hi, Tol::stat(n));
                if (id, tag, se) == ("rq_br_se", "tau25", QuantRegSe::Nid) {
                    // deviation REG-26: the scale-free fuzz moves this intercept p-value 3.4e-6
                    // relative (outside Tol::p_value(100)); the slope's p-value is pinned
                    check.named_subset(
                        &format!("{tag}_p_{key}"),
                        &["x"],
                        names,
                        &p,
                        Tol::p_value(n),
                    );
                    let rp = case.named(&format!("{tag}_p_{key}"), "(Intercept)");
                    assert!((p[0] / rp - 1.0).abs() < 1e-5, "{} vs {rp}", p[0]);
                } else {
                    check.named(&format!("{tag}_p_{key}"), names, &p, Tol::p_value(n));
                }
                assert_eq!(r.se_type, se);
                out.push((id, tag, r));
            }
            check.finish();
        }
        out
    }

    /// summary.rq(se = "iid"): Koenker-Bassett sparsity with the Hall-Sheather bandwidth; the tau
    /// 0.75 sparsity fit is nonunique in R too (its vertex defines the estimator). Fails before
    /// WP10: SEs sqrt(n) too small (tau 0.5: 0.0503 / 0.00832 vs 0.4807 / 0.0795).
    #[test]
    fn test_validate_rq_summary_se_iid() {
        for (id, tag, r) in check_summary_se(QuantRegSe::Iid) {
            let (case, _) = crate::test_support::load_case(id);
            let n = case.scalar("n") as usize;
            let mut check = Check::new(&case);
            check.scalar(
                &format!("{tag}_scale_iid"),
                1.0 / r.sparsity.unwrap(),
                Tol::stat(n),
            );
            if id == "rq_br_se" {
                check.scalar(
                    &format!("{tag}_hs_bandwidth"),
                    r.se_bandwidth.unwrap(),
                    Tol::stat(n),
                );
            }
            check.finish();
            let k = r
                .warnings
                .iter()
                .filter(|w| *w == "Solution may be nonunique (iid sparsity fit)")
                .count();
            assert_eq!(
                k as f64,
                case.scalar(&format!("{tag}_nonunique_iid")),
                "{id} {tag}"
            );
            assert!(
                r.se_method.starts_with("iid (Koenker-Bassett sparsity"),
                "{}",
                r.se_method
            );
        }
    }

    /// summary.rq(se = "nid") (p2a's default at every n): the Hendricks-Koenker sandwich with
    /// tau -/+ h fits; mtcars tau 0.25 has R's "1 non-positive fis"; the no-intercept fit.
    #[test]
    fn test_validate_rq_summary_se_nid() {
        for (id, tag, r) in check_summary_se(QuantRegSe::Nid) {
            let (case, _) = crate::test_support::load_case(id);
            let fis: Vec<String> = r
                .warnings
                .iter()
                .filter(|w| w.ends_with(" non-positive fis"))
                .cloned()
                .collect();
            let expected = case.scalar(&format!("{tag}_nonpositive_fis"));
            let want = if expected > 0.0 {
                vec![format!("{expected} non-positive fis")]
            } else {
                vec![]
            };
            assert_eq!(fis, want, "{id} {tag}");
            if id == "rq_br_se" {
                Check::new(&case)
                    .scalar(
                        &format!("{tag}_hs_bandwidth"),
                        r.se_bandwidth.unwrap(),
                        Tol::stat(100),
                    )
                    .finish();
            }
            assert!(
                r.se_method.starts_with("nid (Hendricks-Koenker sandwich"),
                "{}",
                r.se_method
            );
        }
        let (case, ds) = crate::test_support::load_case("rq_br_mtcars");
        let mut check = Check::new(&case);
        for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
            let mut cfg = QuantRegConfig::default();
            cfg.tau = tau;
            cfg.intercept = false;
            let g = quantreg(&ds, "mpg", &["wt", "hp"], &cfg).unwrap();
            check.named(
                &format!("noint_{tag}_se_nid"),
                &g.variable_names,
                &table(&g)[0],
                Tol::se(32),
            );
        }
        check.finish();
    }

    /// summary.rq(se = "ker"): Powell's kernel sandwich with the residual-scale bandwidth.
    #[test]
    fn test_validate_rq_summary_se_ker() {
        for (id, tag, r) in check_summary_se(QuantRegSe::Ker) {
            assert!(
                r.se_method.starts_with("ker (Powell kernel sandwich"),
                "{}",
                r.se_method
            );
            if id != "rq_br_se" {
                continue;
            }
            let (case, _) = crate::test_support::load_case(id);
            Check::new(&case)
                .scalar(
                    &format!("{tag}_ker_bandwidth"),
                    r.kernel_bandwidth.unwrap(),
                    Tol::stat(100),
                )
                .scalar(
                    &format!("{tag}_hs_bandwidth"),
                    r.se_bandwidth.unwrap(),
                    Tol::stat(100),
                )
                .finish();
        }
    }

    /// tau = 0.02, n = 100: bandwidth.rq = 0.0243 exceeds tau, so summary.rq halves it to 0.01214
    /// (nid and ker).
    #[test]
    fn test_validate_rq_bandwidth_halving() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let mut check = Check::new(&case);
        check.scalar(
            "tau02_hs_bandwidth_unhalved",
            hall_sheather_bandwidth(0.02, 100).unwrap(),
            Tol::stat(100),
        );
        for se in [QuantRegSe::Nid, QuantRegSe::Ker] {
            let mut cfg = QuantRegConfig::default();
            cfg.tau = 0.02;
            cfg.se = se;
            let r = quantreg(&ds, "y", &["x"], &cfg).unwrap();
            let key = format!("tau02_se_{se}");
            let s = table(&r)[0].clone();
            if se == QuantRegSe::Nid {
                // deviation REG-26: the scale-free fuzz (sqrt(eps) * mean |y - median(y)|) moves the
                // intercept's nid SE 1.2e-6 relative here, where tau - h is small; the slope is pinned
                check.named_subset(&key, &["x"], &r.variable_names, &s, Tol::se(100));
                let rs = case.named(&key, "(Intercept)");
                assert!((s[0] / rs - 1.0).abs() < 2e-6, "{} vs {rs}", s[0]);
            } else {
                check.named(&key, &r.variable_names, &s, Tol::se(100));
            }
            check.scalar(
                "tau02_hs_bandwidth",
                r.se_bandwidth.unwrap(),
                Tol::stat(100),
            );
        }
        check.finish();
        assert!(
            hall_sheather_bandwidth(0.0, 100).is_err() && hall_sheather_bandwidth(0.5, 0).is_err()
        );
    }

    /// REG-64: both interval ends are estimate -/+ qt(0.975, n - p) * se, so p < 0.05 exactly when
    /// 0 is outside the interval. qt(0.975, 98) = 1.9844674545084815, qt(0.975, 28) =
    /// 2.0484071417952445 (R). Fails before WP10: 1.96.
    #[test]
    fn test_rq_ci_uses_t_quantile() {
        let mt = crate::regression::ols::na_fixtures::mtcars();
        for (ds, y, xs, q) in [
            (hetero(), "y", &["x"][..], 1.9844674545084815),
            (mt, "mpg", &["wt", "hp", "qsec"][..], 2.0484071417952445),
        ] {
            for se in [QuantRegSe::Iid, QuantRegSe::Nid, QuantRegSe::Ker] {
                for tau in [0.25, 0.5, 0.75] {
                    let mut cfg = QuantRegConfig::default();
                    cfg.tau = tau;
                    cfg.se = se;
                    for c in &quantreg(&ds, y, xs, &cfg).unwrap().coefficients {
                        assert!(
                            ((c.ci_upper_95 - c.estimate) / (q * c.std_error) - 1.0).abs() < 1e-9,
                            "{y} {se} {tau} {}",
                            c.name
                        );
                        assert!(
                            ((c.estimate - c.ci_lower_95) / (q * c.std_error) - 1.0).abs() < 1e-9,
                            "{y} {se} {tau} {}",
                            c.name
                        );
                        assert_eq!(
                            c.p_value < 0.05,
                            c.ci_lower_95 > 0.0 || c.ci_upper_95 < 0.0,
                            "{y} {se} {tau} {}",
                            c.name
                        );
                    }
                }
            }
        }
    }

    /// Rule 4: the iid, nid and ker SEs are scale-equivariant (summary.rq's fuzz is multiplied by
    /// mean |y - median(y)|). R's own iid and nid SEs move by 7.7% and 4.0% at y x 1e-6
    /// (scale_m6_tau50_se_* in rq_br_se, recorded, not pinned).
    #[test]
    fn test_rq_se_scale_equivariance() {
        let ds = hetero();
        for se in [QuantRegSe::Iid, QuantRegSe::Nid, QuantRegSe::Ker] {
            let mut cfg = QuantRegConfig::default();
            cfg.se = se;
            let base = quantreg(&ds, "y", &["x"], &cfg).unwrap();
            for factor in [1e-6, 1e6] {
                let r = quantreg(
                    &with_derived(&ds, "ys", "y", |v| v * factor),
                    "ys",
                    &["x"],
                    &cfg,
                )
                .unwrap();
                for (c, b) in r.coefficients.iter().zip(&base.coefficients) {
                    assert!(
                        (c.std_error / factor / b.std_error - 1.0).abs() < 1e-12,
                        "{se} x {factor}: {} vs {}",
                        c.std_error,
                        b.std_error
                    );
                    assert!(
                        (c.p_value / b.p_value - 1.0).abs() < 1e-9 || c.p_value == b.p_value,
                        "{se} x {factor}"
                    );
                }
                assert_eq!(r.warnings, base.warnings, "{se} x {factor}");
            }
        }
    }

    /// n = 5 (rows 1-5): R's summary.rq(se = "iid") stops ("variable lengths differ"; h + pz + 1 = 6
    /// > 5), while rq() itself works. p2a keeps the fit, reports NaN SEs and says why. nid works and
    /// matches R, with "1 non-positive fis".
    #[test]
    fn test_validate_rq_small_sample_se() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let d5 = Dataset::new(ds.df().head(Some(5)));
        let mut cfg = QuantRegConfig::default();
        cfg.se = QuantRegSe::Iid;
        let r = quantreg(&d5, "y", &["x"], &cfg).unwrap();
        let mut check = Check::new(&case);
        check.named(
            "small5_coef",
            &r.variable_names,
            &estimates(&r),
            Tol::coef(5),
        );
        assert!(
            r.coefficients
                .iter()
                .all(|c| c.std_error.is_nan() && c.p_value.is_nan() && c.ci_lower_95.is_nan())
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("standard errors (se = iid) could not be computed")),
            "{:?}",
            r.warnings
        );
        cfg.se = QuantRegSe::Nid;
        let r = quantreg(&d5, "y", &["x"], &cfg).unwrap();
        check
            .named(
                "small5_se_nid",
                &r.variable_names,
                &table(&r)[0],
                Tol::se(5),
            )
            .finish();
        assert!(
            r.warnings.iter().any(|w| w == "1 non-positive fis"),
            "{:?}",
            r.warnings
        );
        assert_eq!(case.scalar("small5_iid_fails"), 1.0);
        assert_eq!(case.scalar("small5_nonpositive_fis"), 1.0);
    }

    /// quantreg_multi / quantreg_multi_with give the single fits (nid default; iid on request); an
    /// empty taus is an error (TOOL-14's core part). Replaces the range-only
    /// test_validate_quantreg_multi (REG-44).
    #[test]
    fn test_validate_rq_multi_hetero() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let mut check = Check::new(&case);
        let rs = quantreg_multi(&ds, "y", &["x"], &[0.25, 0.5, 0.75]).unwrap();
        for (r, tag) in rs.iter().zip(["tau25", "tau50", "tau75"]) {
            check
                .named(
                    &format!("{tag}_coef"),
                    &r.variable_names,
                    &estimates(r),
                    Tol::coef(100),
                )
                .named(
                    &format!("{tag}_se_nid"),
                    &r.variable_names,
                    &table(r)[0],
                    Tol::se(100),
                );
        }
        let mut cfg = QuantRegConfig::default();
        cfg.se = QuantRegSe::Iid;
        let ri = quantreg_multi_with(&ds, "y", &["x"], &[0.75], &cfg).unwrap();
        check
            .named(
                "tau75_se_iid",
                &ri[0].variable_names,
                &table(&ri[0])[0],
                Tol::se(100),
            )
            .finish();
        assert!(matches!(
            quantreg_multi(&ds, "y", &["x"], &[]),
            Err(EconError::InvalidSpecification { message }) if message.contains("taus")
        ));
    }

    // ── Frisch-Newton interior point and the automatic choice ──

    /// method = fn (rqfnb on y / mean |y - median(y)|): converged, and within the duality-gap
    /// tolerance of the exact LP solution rq(method = "br") (Tol::iterative(1e-6); R's rq.fit.fnb
    /// on the standardised outcome is 1.7e-10 / 3.0e-9 / 8.3e-8 away). The iterates are not pinned
    /// (fn_*_coef and fn_*_nit stay in the case as R facts). Fails before WP10: InteriorPoint ran
    /// IRLS.
    #[test]
    fn test_validate_rq_fn_coefficients() {
        let (case, ds) = crate::test_support::load_case("rq_br_se");
        let mut check = Check::new(&case);
        for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
            let mut cfg = QuantRegConfig::default();
            cfg.tau = tau;
            cfg.algorithm = QuantRegMethod::FrischNewton;
            let r = quantreg(&ds, "y", &["x"], &cfg).unwrap();
            check.named(
                &format!("{tag}_coef"),
                &r.variable_names,
                &estimates(&r),
                Tol::iterative(1e-6),
            );
            assert!(
                r.convergence.converged && r.algorithm == QuantRegMethod::FrischNewton,
                "{tag}"
            );
            assert!(
                r.convergence.final_change <= cfg.tol,
                "{tag}: gap {}",
                r.convergence.final_change
            );
            assert!(!r.nonunique);
            assert!(
                r.to_string()
                    .contains("Algorithm: Frisch-Newton interior point (fn)\n")
            );
        }
        check.finish();
    }

    /// Scale-free stopping: y x 1e-6 and 1e6 take the same iterations and give the factor times the
    /// unscaled coefficients. R's own rq(method = "fn") at 1e-6 stops after 3 iterations, 0.93% off
    /// (scale_m6_fn_r_tau50_coef, recorded, not pinned).
    #[test]
    fn test_rq_fn_scale_free() {
        let ds = hetero();
        let mut cfg = QuantRegConfig::default();
        cfg.algorithm = QuantRegMethod::FrischNewton;
        let base = quantreg(&ds, "y", &["x"], &cfg).unwrap();
        for factor in [1e-6, 1e6] {
            let r = quantreg(
                &with_derived(&ds, "ys", "y", |v| v * factor),
                "ys",
                &["x"],
                &cfg,
            )
            .unwrap();
            assert_eq!(r.iterations, base.iterations, "x {factor}");
            for (c, b) in r.coefficients.iter().zip(&base.coefficients) {
                assert!(
                    (c.estimate / factor / b.estimate - 1.0).abs() < 1e-9,
                    "x {factor}"
                );
                assert!(
                    (c.std_error / factor / b.std_error - 1.0).abs() < 1e-9,
                    "x {factor}"
                );
            }
        }
    }

    /// R: rq.fit.fnb stops with "No parametric Frisch-Newton method.  Set tau in (0,1)" when tau <
    /// eps or > 1 - eps.
    #[test]
    fn test_rq_fn_rejects_extreme_tau() {
        let mut cfg = QuantRegConfig::default();
        cfg.algorithm = QuantRegMethod::FrischNewton;
        for tau in [5e-7, 1.0 - 5e-7] {
            cfg.tau = tau;
            assert!(matches!(
                quantreg(&hetero(), "y", &["x"], &cfg),
                Err(EconError::InvalidSpecification { message }) if message.contains("fn") && message.contains("tau")
            ));
        }
        // the automatic choice falls back to the simplex there
        assert_eq!(
            resolve_method(QuantRegMethod::Auto, 10_000, 5e-7, 1e-6),
            QuantRegMethod::BarrodaleRoberts
        );
    }

    /// Auto: the simplex up to 5000 rows, Frisch-Newton above (the result names the one that ran);
    /// on a unique problem the two agree within Tol::iterative(1e-6).
    #[test]
    fn test_rq_auto_selects_by_n() {
        use QuantRegMethod::*;
        assert_eq!(resolve_method(Auto, 5000, 0.5, 1e-6), BarrodaleRoberts);
        assert_eq!(resolve_method(Auto, 5001, 0.5, 1e-6), FrischNewton);
        assert_eq!(resolve_method(Irls, 5001, 0.5, 1e-6), Irls);
        let n = 5001;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                ((i * 37) % 101) as f64 / 10.0
            }
        });
        let y = Array1::from_iter((0..n).map(|i| {
            1.0 + 0.5 * x[[i, 1]] + (((i * 7919) % 1009) as f64 / 1009.0 - 0.5) * (1.0 + x[[i, 1]])
        }));
        let auto = rq_fit(x.view(), y.view(), 0.5, &QuantRegConfig::default()).unwrap();
        assert_eq!(auto.algorithm, FrischNewton);
        let mut cfg = QuantRegConfig::default();
        cfg.algorithm = BarrodaleRoberts;
        let br = rq_fit(x.view(), y.view(), 0.5, &cfg).unwrap();
        assert_eq!(br.algorithm, BarrodaleRoberts);
        for (a, b) in auto.coefficients.iter().zip(&br.coefficients) {
            assert!(Tol::iterative(1e-6).allows(*a, *b), "{a} vs {b}");
        }
        assert!(
            auto.objective >= br.objective * (1.0 - 1e-12)
                && auto.objective <= br.objective * (1.0 + 1e-6)
        );
    }

    // ── xy-pair bootstrap ──

    /// boot.rq(bsmethod = "xy") on the committed index matrix: each replicate is the simplex fit on
    /// rows U[, b]; the SE is sqrt(diag(cov(B))) (summary.rq(se = "boot")). R's boot.rq.xy uses
    /// quantreg's rq0; on this matrix all 150 replicate fits equal rqbr's exactly.
    #[test]
    fn test_validate_rq_boot_xy_indices() {
        let (case, idx) = crate::test_support::load_case("rq_boot_xy");
        let ds = hetero();
        let n = case.scalar("n") as usize;
        let reps = case.scalar("replications") as usize;
        let y = Array1::from(nan_vec(&ds, "y"));
        let xv = nan_vec(&ds, "x");
        let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { 1.0 } else { xv[i] });
        let cols: Vec<Vec<usize>> = (1..=reps)
            .map(|b| {
                nan_vec(&idx, &format!("b{b:02}"))
                    .iter()
                    .map(|&v| v as usize - 1)
                    .collect()
            })
            .collect();
        let names = ["(Intercept)", "x"];
        let mut check = Check::new(&case);
        let cfg = QuantRegConfig::default();
        for (tau, tag) in [(0.25, "tau25"), (0.5, "tau50"), (0.75, "tau75")] {
            let d = bootstrap_xy_coefficients(
                x.view(),
                y.view(),
                tau,
                reps,
                QuantRegMethod::BarrodaleRoberts,
                &cfg,
                |r, out| {
                    out.clear();
                    out.extend_from_slice(&cols[r]);
                },
            )
            .unwrap();
            assert_eq!((d.dropped, d.coefficients.nrows()), (0, reps));
            let (se, _) = bootstrap_se(&d.coefficients);
            let mean: Vec<f64> = (0..2)
                .map(|j| d.coefficients.column(j).mean().unwrap())
                .collect();
            check
                .named(
                    &format!("{tag}_se_boot"),
                    &names,
                    se.as_slice().unwrap(),
                    Tol::se(n),
                )
                .named(&format!("{tag}_boot_mean"), &names, &mean, Tol::coef(n));
        }
        check.finish();
    }

    /// Public path: seeded ChaCha8 resampling (stream = replicate index); the same seed gives the
    /// same SEs, another seed different ones; bootstrap_samples = 0 means R's 200.
    #[test]
    fn test_rq_boot_seeded_reproducible() {
        let ds = hetero();
        let mut cfg = QuantRegConfig::default();
        cfg.se = QuantRegSe::Boot;
        cfg.bootstrap_samples = 60;
        cfg.seed = Some(7);
        let a = quantreg(&ds, "y", &["x"], &cfg).unwrap();
        let b = quantreg(&ds, "y", &["x"], &cfg).unwrap();
        assert_eq!(table(&a)[0], table(&b)[0]);
        assert!(table(&a)[0].iter().all(|s| s.is_finite() && *s > 0.0));
        assert!(
            a.se_method.contains("R = 60") && a.se_method.contains("seed 7"),
            "{}",
            a.se_method
        );
        assert_eq!(a.se_type, QuantRegSe::Boot);
        cfg.seed = Some(8);
        assert_ne!(
            table(&quantreg(&ds, "y", &["x"], &cfg).unwrap())[0],
            table(&a)[0]
        );
        cfg.bootstrap_samples = 0;
        cfg.seed = None;
        assert!(
            quantreg(&ds, "y", &["x"], &cfg)
                .unwrap()
                .se_method
                .contains("R = 200, seed 42")
        );
    }

    /// bootstrap_samples is used only with se = boot (R2: a setting that would be ignored is an
    /// error), and needs >= 2. Fails before WP10: accepted and ignored (a tracing warning only).
    #[test]
    fn test_rq_bootstrap_samples_requires_boot_se() {
        let ds = hetero();
        let mut cfg = QuantRegConfig::default();
        cfg.bootstrap_samples = 10;
        assert!(matches!(
            quantreg(&ds, "y", &["x"], &cfg),
            Err(EconError::InvalidSpecification { message }) if message.contains("bootstrap_samples")
        ));
        cfg.se = QuantRegSe::Boot;
        cfg.bootstrap_samples = 1;
        assert!(matches!(
            quantreg(&ds, "y", &["x"], &cfg),
            Err(EconError::InvalidSpecification { message }) if message.contains("at least 2")
        ));
    }

    /// ker on heavily tied outcomes: y in {0, 1, 2} with one binary regressor fits more than half
    /// the rows exactly, so IQR(u) = 0 and Powell's bandwidth is 0. The fit is kept, the standard
    /// errors are NaN and a warning names the cause (before: the whole call failed with "x
    /// contains a non-finite value").
    #[test]
    fn test_rq_ker_zero_residual_scale_keeps_fit() {
        let n = 60;
        let x: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        // mostly y = 1 + x, with a few 0s and 2s as noise
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let base = 1.0 + (i % 2) as f64;
                match i % 7 {
                    0 => base - 1.0,
                    3 if base < 2.0 => base + 1.0,
                    _ => base,
                }
            })
            .collect();
        let ds = Dataset::new(df! { "y" => y, "x" => x }.unwrap());
        let mut cfg = QuantRegConfig::default();
        cfg.se = QuantRegSe::Ker;
        let r = quantreg(&ds, "y", &["x"], &cfg).unwrap();
        assert!(r.coefficients.iter().all(|c| c.estimate.is_finite()));
        assert!(r.coefficients.iter().all(|c| c.std_error.is_nan()));
        assert!(
            r.warnings.iter().any(|w| w.contains("se = ker")
                && w.contains("kernel bandwidth")
                && w.contains("tied")),
            "{:?}",
            r.warnings
        );
    }

    /// nid after a Frisch-Newton fit: an auxiliary tau -/+ h outside (tol, 1 - tol) runs the
    /// simplex instead of failing (Frisch-Newton cannot run there), as `Auto` does for the main fit.
    #[test]
    fn test_rq_nid_aux_fit_falls_back_to_simplex_outside_fn_range() {
        let ds = hetero();
        let tau = 0.1;
        let n = ds.df().height();
        let h = halved_bandwidth(tau, n).unwrap();
        assert!(tau - h > 0.0 && tau - h < tau / 2.0, "h = {h}");
        let mut cfg = QuantRegConfig::default();
        cfg.tau = tau;
        cfg.algorithm = QuantRegMethod::FrischNewton;
        cfg.se = QuantRegSe::Nid;
        // tau - h <= tol < tau: the main fit can run Frisch-Newton, the lower auxiliary fit cannot
        cfg.tol = (tau - h + tau) / 2.0;
        let r = quantreg(&ds, "y", &["x"], &cfg).unwrap();
        assert_eq!(r.algorithm, QuantRegMethod::FrischNewton);
        assert!(r.coefficients.iter().all(|c| c.std_error.is_finite()));
        use QuantRegMethod::*;
        assert_eq!(aux_method(FrischNewton, tau - h, cfg.tol), BarrodaleRoberts);
        assert_eq!(aux_method(FrischNewton, tau + h, cfg.tol), FrischNewton);
        assert_eq!(aux_method(BarrodaleRoberts, 0.5, 1e-6), BarrodaleRoberts);
        assert_eq!(aux_method(Irls, 0.5, 1e-6), BarrodaleRoberts);
    }
}
