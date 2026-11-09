//! Arellano-Bond difference GMM and Blundell-Bond system GMM for dynamic panel data, as
//! R's `plm::pgmm(transformation = "d")` and `plm::pgmm(transformation = "ld")`.
//!
//! # Model
//!
//! ```text
//! y_it = α_1 y_{i,t-1} + ... + α_p y_{i,t-p} + x_it' β + η_i + ε_it
//! ```
//!
//! First differences remove η_i; the lagged levels y_{i,t-2}, y_{i,t-3}, ... are valid
//! instruments for the differenced equation when ε_it is serially uncorrelated, and the
//! strictly exogenous x enter in differences as their own instruments.
//!
//! # Conventions (those of `plm::pgmm`)
//!
//! - **Equations.** The differenced equation of (entity, t) exists only where y is
//!   observed at t, t-1, ..., t-p-1. Time runs in integer steps of 1, as plm's time-aware
//!   `lag()` and Stata without `delta()`: recode biennial years or dates as consecutive
//!   period numbers. Nothing is zero-filled: gaps and late entry drop the equations that
//!   need a missing period. A repeated (entity, time) row is an error. `n_obs` is the
//!   number of equations formed.
//! - **Instruments.** The observed levels y_s, s in [t - max_lag, t - min_lag], one
//!   column per (equation period, s), block-diagonal by period, or one per lag depth with
//!   `collapse`; then the differenced x. A column that no equation observes is not
//!   created, so with a period no entity observes the instrument count is pgmm's minus
//!   its all-zero columns.
//! - **Weights.** One-step: (Σ_i Z_i' H_i Z_i)^-1, H tridiagonal with 2 on the diagonal
//!   and -1 between equations one period apart. Two-step: (Σ_i Z_i' e_i e_i' Z_i)^-1 from
//!   the one-step residuals. A weighting matrix that is numerically singular gets
//!   `MASS::ginv`'s generalized inverse, as pgmm, and a warning; the singularity decision
//!   is made on the matrix scaled to unit diagonal, so it does not depend on the units of
//!   y and x.
//! - **Standard errors.** One-step: Arellano-Bond (1991) classical σ² (X'Z A Z'X)^-1, σ²
//!   = Σ e² / (2 (n_obs - k)) from the differenced residuals (Stata `xtabond`; pgmm's own
//!   non-robust one-step matrix omits σ²), or robust as `plm::vcovHC.pgmm`. Two-step:
//!   uncorrected (X'Z A Z'X)^-1, or the Windmeijer (2005) correction of `vcovHC.pgmm`.
//! - **Tests.** The Sargan/Hansen statistic as `plm::sargan` (two-step weights, for one-
//!   and two-step fits), with df = instruments - parameters; one-step fits also report the
//!   Arellano-Bond (1991) Sargan statistic under homoskedastic errors. The AR(1)/AR(2)
//!   m-tests as `plm::mtest` with the robust covariance; a test with no pair of equations
//!   that many periods apart is NaN, with a warning.
//! - **Missing values.** Rows with a missing value in y, x, the entity or the time are
//!   dropped (`na.omit`) and reported in `missing` and `warnings`.
//!
//! # System GMM (`GmmTransform::System`; `pgmm` "ld", `fsm = "full"`)
//!
//! - **Equations.** Each entity's difference equations (as above) are stacked with its
//!   level equations: y_t on y_{t-1}, ..., y_{t-p} and x_t, for each t with y observed at
//!   t, ..., t-p and t - t_min >= max(p + 1, min_lag) - 1. `n_obs` counts both kinds
//!   (`n_obs_diff`, `n_obs_level`).
//! - **Instruments.** The level equation at t is instrumented by Δy_{t-1} = y_{t-1} -
//!   y_{t-2} where both are observed, one column per period (one column with `collapse`),
//!   whatever `min_lag`; x enters as its own instrument in differences (difference rows)
//!   and in levels (level rows).
//! - **No intercept.** The level equations have no intercept (plm::pgmm `effect =
//!   "individual"`); with x in levels as its own instrument they are valid only when
//!   E[eta_i] = 0 or x has mean zero; Stata's xtabond2 adds `_cons`.
//! - **Weights.** One-step: (Σ_i Z_i' H_i Z_i)^-1 with pgmm's "full" H: the tridiagonal
//!   difference block, the identity level block, and +1 between the difference and the
//!   level equation of the same period, -1 between the difference equation at t and the
//!   level equation at t - 1. Two-step as above.
//! - **Standard errors and tests.** One-step robust (`vcovHC.pgmm`), two-step uncorrected
//!   or Windmeijer; one-step classical standard errors are refused, because H is the error
//!   covariance of the level equations only when var(eta_i) = 0, and so the homoskedastic
//!   Sargan statistic is not reported. σ² comes from the difference equations. Sargan as
//!   `plm::sargan`; the AR(1)/AR(2) m-tests use the differenced residuals, as `plm::mtest`.
//! - **`lags >= 3` or `min_lag >= 4`.** plm 2.6.7 fills its level-instrument matrix from a
//!   vector of the wrong length in these settings; p2a applies the rule pgmm applies
//!   otherwise (Δy_{t-1} at t) and is pinned there to the estimator written out in R.
//!
//! # References
//!
//! - Arellano, M. & Bond, S. (1991). Some Tests of Specification for Panel Data:
//!   Monte Carlo Evidence and an Application to Employment Equations.
//!   *Review of Economic Studies*, 58(2), 277-297.
//!   https://doi.org/10.2307/2297968
//!
//! - Windmeijer, F. (2005). A finite sample correction for the variance of linear
//!   efficient two-step GMM estimators. *Journal of Econometrics*, 126(1), 25-51.
//!   https://doi.org/10.1016/j.jeconom.2004.02.005
//!
//! - Blundell, R. & Bond, S. (1998). Initial Conditions and Moment Restrictions
//!   in Dynamic Panel Data Models. *Journal of Econometrics*, 87(1), 115-143.
//!   https://doi.org/10.1016/S0304-4076(98)00009-8
//!
//! R equivalent: `plm::pgmm()`, `plm::vcovHC.pgmm()`, `plm::sargan()`, `plm::mtest()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky_inverse, eig_symmetric};
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::SignificanceLevel;

use super::utils::{EqKind, GmmEquations, GmmLagSpec, gmm_equations, h_entry};

/// The inversions of the L x L weighting matrices hold about this many L x L matrices at
/// their peak: the moment matrix and its symmetrised and scaled copies, the
/// eigendecomposition's vectors and workspace (most of the total), the Cholesky factor and
/// the solution or the generalized inverse's factors, and the one-step inverse while the
/// two-step weight is inverted. Measured with a counting allocator on one- and two-step
/// fits, near the instrument count at which the hosted per-allocation ceiling binds. In
/// these units the peak is higher for small L, where a fixed part of the
/// eigendecomposition's workspace dominates, and falls towards 11 as L grows. Check the
/// working set against the allocation ceiling before building it.
const DENSE_INVERSE_COPIES: usize = 16;

fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
}

/// GMM transformation type for dynamic panel models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmTransform {
    /// Difference GMM (Arellano-Bond 1991)
    /// First-differences to eliminate fixed effects
    #[default]
    Difference,
    /// System GMM (Blundell-Bond 1998), as `plm::pgmm(transformation = "ld")`: the
    /// difference equations stacked with level equations instrumented by y_{t-1} - y_{t-2}
    System,
}

impl fmt::Display for GmmTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GmmTransform::Difference => write!(f, "Difference GMM (Arellano-Bond)"),
            GmmTransform::System => write!(f, "System GMM (Blundell-Bond)"),
        }
    }
}

crate::option_enum!(
    GmmTransform,
    crate::options::Choices {
        option: "transform",
        values: &[
            (
                &["difference", "diff", "d", "arellano_bond"],
                GmmTransform::Difference
            ),
            (
                &["system", "sys", "ld", "blundell_bond"],
                GmmTransform::System
            ),
        ],
    },
    no_display
);

/// GMM estimation step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GmmStep {
    /// One-step GMM, weighted by (Σ_i Z_i' H Z_i)^-1
    #[default]
    OneStep,
    /// Two-step GMM, weighted by (Σ_i Z_i' e_i e_i' Z_i)^-1 from the one-step residuals
    TwoStep,
}

impl fmt::Display for GmmStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GmmStep::OneStep => write!(f, "One-step"),
            GmmStep::TwoStep => write!(f, "Two-step"),
        }
    }
}

crate::option_enum!(
    GmmStep,
    crate::options::Choices {
        option: "step",
        values: &[
            (&["onestep", "one_step", "one", "1"], GmmStep::OneStep),
            (
                &["twostep", "two_step", "twosteps", "two", "2"],
                GmmStep::TwoStep
            ),
        ],
    },
    no_display
);

/// Configuration for GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmConfig {
    /// Transformation: `Difference` (Arellano-Bond, `pgmm` "d") or `System`
    /// (Blundell-Bond, `pgmm` "ld" with `fsm = "full"`). The level equations of `System`
    /// have no intercept (plm::pgmm `effect = "individual"`); with x in levels as its own
    /// instrument they are valid only when E[eta_i] = 0 or x has mean zero; Stata's
    /// xtabond2 adds `_cons`.
    pub transform: GmmTransform,
    /// Estimation step (one-step or two-step; default two-step)
    pub step: GmmStep,
    /// Deepest lag of y used as instrument; `None`: every available lag
    pub max_lag: Option<usize>,
    /// Shallowest lag of y used as instrument (default 2; 0 is an error, 1 warns)
    pub min_lag: usize,
    /// One instrument column per lag depth instead of one per (period, lag)
    pub collapse: bool,
    /// One-step: `plm::vcovHC` robust (true) or Arellano-Bond (1991) classical (false)
    /// standard errors; system GMM has no classical one-step standard errors (false is an
    /// error). Two-step: Windmeijer (2005)-corrected (true) or uncorrected.
    pub robust: bool,
}

impl Default for GmmConfig {
    fn default() -> Self {
        Self {
            transform: GmmTransform::Difference,
            step: GmmStep::TwoStep,
            max_lag: None,
            min_lag: 2,
            collapse: false,
            robust: true,
        }
    }
}

/// Result from dynamic-panel GMM estimation (difference or system).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmResult {
    /// Estimator run
    pub transform: GmmTransform,
    /// Estimation step
    pub step: GmmStep,
    /// Dependent variable name ("D.y" for difference GMM, "y" for system GMM)
    pub dep_var: String,
    /// Variable names: L1.y, ..., Lp.y, then the x columns
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors of type `se_type`
    pub std_errors: Vec<f64>,
    /// z-statistics (NaN when a standard error is 0)
    pub z_stats: Vec<f64>,
    /// Two-sided normal p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Number of equations used: differenced plus level equations (pgmm's "Number of
    /// Observations Used")
    pub n_obs: usize,
    /// Number of entities with at least one equation of either kind
    pub n_groups: usize,
    /// Number of instrument columns used (columns no equation observes are not created):
    /// difference GMM, level GMM (system), D.x, x (system)
    pub n_instruments: usize,
    /// Sargan/Hansen statistic as `plm::sargan` (two-step weights, for one- and two-step
    /// fits)
    pub sargan_statistic: f64,
    /// Sargan/Hansen p-value (NaN when `sargan_df` is 0)
    pub sargan_p_value: f64,
    /// Sargan/Hansen degrees of freedom: `n_instruments` - parameters
    pub sargan_df: usize,
    /// Arellano-Bond AR(1) m-test (`plm::mtest`, robust covariance; NaN when undefined)
    pub ar1_statistic: f64,
    /// AR(1) test p-value
    pub ar1_p_value: f64,
    /// Arellano-Bond AR(2) m-test (`plm::mtest`, robust covariance; NaN when undefined)
    pub ar2_statistic: f64,
    /// AR(2) test p-value
    pub ar2_p_value: f64,
    /// Entity variable name
    pub entity_var: String,
    /// Time variable name
    pub time_var: String,
    /// Warnings (generalized inverses, undefined tests, dropped rows, ...)
    pub warnings: Vec<String>,
    /// Standard-error type of `std_errors`.
    #[serde(default)]
    pub se_type: String,
    /// σ² = Σ (one-step differenced residual)² / (2 (n_obs_diff - k)) over the difference
    /// equations (Arellano-Bond 1991); NaN when n_obs_diff <= k.
    #[serde(default)]
    pub sigma2: f64,
    /// Arellano-Bond (1991) Sargan statistic of a one-step fit under homoskedastic errors:
    /// (Z'e)' (Σ_i Z_i' H Z_i)^-1 (Z'e) / σ², df = `sargan_df`. `None` for two-step fits
    /// and for system GMM (its H is not the error covariance of the level equations).
    #[serde(default)]
    pub sargan_homoskedastic: Option<f64>,
    /// p-value of `sargan_homoskedastic`.
    #[serde(default)]
    pub sargan_homoskedastic_p_value: Option<f64>,
    /// Numerical rank of Σ_i Z_i' H Z_i; below `n_instruments`, a generalized inverse was
    /// used (see `warnings`).
    #[serde(default)]
    pub one_step_weight_rank: usize,
    /// Numerical rank of Σ_i Z_i' e_i e_i' Z_i; below `n_instruments`, a generalized
    /// inverse was used (see `warnings`).
    #[serde(default)]
    pub two_step_weight_rank: usize,
    /// First-step coefficients (equal to `coefficients` for a one-step fit).
    #[serde(default)]
    pub one_step_coefficients: Vec<f64>,
    /// Lags of the dependent variable among the regressors.
    #[serde(default)]
    pub lags: usize,
    /// Shallowest instrument lag.
    #[serde(default)]
    pub min_lag: usize,
    /// Deepest instrument lag as requested (`None`: every available lag).
    #[serde(default)]
    pub max_lag: Option<usize>,
    /// Whether the instruments were collapsed (one column per lag depth).
    #[serde(default)]
    pub collapse: bool,
    /// Rows dropped because of missing values.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
    /// Differenced equations used (equals `n_obs` for difference GMM).
    #[serde(default)]
    pub n_obs_diff: usize,
    /// Level equations used (system GMM; 0 for difference GMM).
    #[serde(default)]
    pub n_obs_level: usize,

    // ── Trait-backing storage (LinearEstimator). Skipped from JSON to keep the
    //    public surface stable.
    /// Coefficients as `Array1`, used by `LinearEstimator::coefficients`.
    #[serde(skip, default)]
    pub coef_arr: ndarray::Array1<f64>,
    /// Standard errors as `Array1`, used by `LinearEstimator::std_errors`.
    #[serde(skip, default)]
    pub se_arr: ndarray::Array1<f64>,
    /// Residuals of the reported step: per entity its difference rows, then (system GMM)
    /// its level rows.
    #[serde(skip, default)]
    pub residuals: ndarray::Array1<f64>,
    /// The reported coefficient covariance matrix (of type `se_type`).
    #[serde(skip, default)]
    pub vcov: ndarray::Array2<f64>,
}

impl crate::traits::estimator::LinearEstimator for GmmResult {
    fn coefficients(&self) -> &ndarray::Array1<f64> {
        &self.coef_arr
    }
    fn std_errors(&self) -> &ndarray::Array1<f64> {
        &self.se_arr
    }
    fn residuals(&self) -> &ndarray::Array1<f64> {
        &self.residuals
    }
    fn vcov_matrix(&self) -> &ndarray::Array2<f64> {
        &self.vcov
    }
    fn variable_names(&self) -> &[String] {
        &self.variables
    }
    fn degrees_of_freedom(&self) -> usize {
        self.n_obs.saturating_sub(self.coefficients.len())
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

impl GmmResult {
    /// "y lags 2..all, one column per period and lag; D.x": what the instruments are; for
    /// system GMM the differenced and the level parts.
    fn instrument_description(&self) -> String {
        let y = self.dep_var.strip_prefix("D.").unwrap_or(&self.dep_var);
        if self.transform == GmmTransform::System {
            let difference = Self::difference_instruments(
                y,
                self.min_lag,
                self.max_lag,
                self.collapse,
                self.x_names(),
            );
            let layout = if self.collapse {
                "collapsed: one column"
            } else {
                "one column per period"
            };
            let mut level = format!("D.{y} lag 1, {layout}");
            for v in self.x_names() {
                level.push_str("; ");
                level.push_str(v);
            }
            return format!("differenced equations: {difference} | level equations: {level}");
        }
        Self::difference_instruments(y, self.min_lag, self.max_lag, self.collapse, self.x_names())
    }

    /// The names of the x columns (the variables after the lags of y).
    fn x_names(&self) -> &[String] {
        self.variables.get(self.lags..).unwrap_or_default()
    }

    /// The instruments of the differenced equations.
    fn difference_instruments(
        y: &str,
        min_lag: usize,
        max_lag: Option<usize>,
        collapse: bool,
        x: &[String],
    ) -> String {
        let max = max_lag.map_or_else(|| "all".to_string(), |m| m.to_string());
        let layout = if collapse {
            "collapsed: one column per lag depth"
        } else {
            "one column per period and lag"
        };
        let mut text = format!("{y} lags {min_lag}..{max}, {layout}");
        let dx: Vec<String> = x.iter().map(|v| format!("D.{v}")).collect();
        if !dx.is_empty() {
            text.push_str("; ");
            text.push_str(&dx.join(", "));
        }
        text
    }
}

impl fmt::Display for GmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let system = self.transform == GmmTransform::System;
        writeln!(f, "{} ({}) Panel GMM Results", self.transform, self.step)?;
        writeln!(f, "===========================================")?;
        if system {
            writeln!(
                f,
                "Dep. Variable: {} (differenced and level equations)",
                self.dep_var
            )?;
        } else {
            writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        }
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "Time: {}", self.time_var)?;
        if system {
            writeln!(
                f,
                "No. Observations: {} ({} differenced, {} level equations)",
                self.n_obs, self.n_obs_diff, self.n_obs_level
            )?;
        } else {
            writeln!(
                f,
                "No. Observations: {} (differenced equations)",
                self.n_obs
            )?;
        }
        writeln!(f, "No. Groups: {}", self.n_groups)?;
        writeln!(
            f,
            "No. Instruments: {} ({})",
            self.n_instruments,
            self.instrument_description()
        )?;
        if !self.se_type.is_empty() {
            writeln!(f, "Std. errors: {}", self.se_type)?;
        }
        writeln!(f)?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coef", "Std Err", "z", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f)?;
        writeln!(
            f,
            "Sargan/Hansen test: chi2({}) = {:.4}, p-value = {:.4}",
            self.sargan_df, self.sargan_statistic, self.sargan_p_value
        )?;
        writeln!(
            f,
            "  H0: overidentifying restrictions are valid (two-step weights, as plm::sargan)"
        )?;
        if let (Some(stat), Some(p)) =
            (self.sargan_homoskedastic, self.sargan_homoskedastic_p_value)
        {
            writeln!(
                f,
                "Sargan test (homoskedastic, one-step weights): chi2({}) = {stat:.4}, \
                 p-value = {p:.4}",
                self.sargan_df
            )?;
        }
        writeln!(
            f,
            "Arellano-Bond Test for AR(1): z = {:.4}, p-value = {:.4}",
            self.ar1_statistic, self.ar1_p_value
        )?;
        writeln!(
            f,
            "Arellano-Bond Test for AR(2): z = {:.4}, p-value = {:.4}",
            self.ar2_statistic, self.ar2_p_value
        )?;
        writeln!(
            f,
            "  H0: No autocorrelation (robust m-tests, as plm::mtest)"
        )?;

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        // The warnings, then the missing-value note as the last line (not a warning line).
        f.write_str(&notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

impl ResultNotes for GmmResult {
    /// The warnings without the missing-value note, which Display prints as its last line.
    fn notes_warnings(&self) -> Vec<String> {
        let note = self.missing.note();
        self.warnings
            .iter()
            .filter(|w| Some(w.as_str()) != note.as_deref())
            .cloned()
            .collect()
    }
}

/// Inverse of a symmetric weighting matrix as `plm::pgmm` inverts it.
struct WeightInverse {
    /// The inverse, or the generalized inverse when `rank` < L.
    inv: Array2<f64>,
    /// Numerical rank.
    rank: usize,
    /// The matrix restricted to the kept eigen-directions (`MASS::ginv(MASS::ginv(S))`);
    /// the matrix itself at full rank. Only when asked for: the one-step weight never
    /// needs it, so its inversion does not hold it.
    kept: Option<Array2<f64>>,
}

/// `(m + m') / 2`.
fn symmetrized(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) * 0.5
}

/// The inverse of a symmetric positive definite matrix scaled to unit diagonal, scaled
/// back, when its smallest eigenvalue exceeds `L * eps` times its largest; `None`
/// otherwise. Equilibrating makes the decision and the inverse invariant to the units
/// of the variables, as R's `solve()`.
fn equilibrated_inverse(s: &Array2<f64>) -> Option<Array2<f64>> {
    let l = s.nrows();
    let diag = s.diag();
    if diag.iter().any(|&d| !(d > 0.0 && d.is_finite())) {
        return None;
    }
    let scale: Array1<f64> = diag.mapv(|d| 1.0 / d.sqrt());
    let mut st = s.clone();
    for ((i, j), v) in st.indexed_iter_mut() {
        *v *= scale[i] * scale[j];
    }
    let (values, _) = eig_symmetric(&st.view()).ok()?;
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let positive_definite = min > l as f64 * f64::EPSILON * max;
    if !positive_definite {
        return None;
    }
    let mut inv = cholesky_inverse(&st.view()).ok()?;
    for ((i, j), v) in inv.indexed_iter_mut() {
        *v *= scale[i] * scale[j];
    }
    Some(symmetrized(&inv))
}

/// The inverse of a symmetric L x L weighting matrix as `plm::pgmm`: the inverse when the
/// matrix scaled to unit diagonal is numerically positive definite, otherwise
/// `MASS::ginv`'s generalized inverse (eigenvalues with |λ| <= sqrt(eps) max|λ| dropped).
/// `with_kept` also returns the matrix restricted to the kept eigen-directions.
fn weight_inverse(s: &Array2<f64>, with_kept: bool) -> EconResult<WeightInverse> {
    let s = symmetrized(s);
    let l = s.nrows();
    if let Some(inv) = equilibrated_inverse(&s) {
        return Ok(WeightInverse {
            inv,
            rank: l,
            kept: with_kept.then_some(s),
        });
    }
    let singular = || EconError::SingularMatrix {
        context: "GMM weighting matrix".to_string(),
        suggestion: "Every instrument column is zero or the residuals are all zero; check \
                     the panel and the lag specification."
            .to_string(),
    };
    let (values, vectors) = eig_symmetric(&s.view()).map_err(|_| singular())?;
    let top = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if !(top > 0.0 && top.is_finite()) {
        return Err(singular());
    }
    let cut = f64::EPSILON.sqrt() * top;
    let keep: Vec<usize> = (0..l).filter(|&j| values[j].abs() > cut).collect();
    let mut vk = resource::try_zeros2(l, keep.len(), "GMM generalized inverse")?;
    let mut vk_inv = resource::try_zeros2(l, keep.len(), "GMM generalized inverse")?;
    for (c, &j) in keep.iter().enumerate() {
        vk.column_mut(c).assign(&vectors.column(j));
        vk_inv
            .column_mut(c)
            .assign(&(&vectors.column(j) / values[j]));
    }
    let inv = symmetrized(&vk_inv.dot(&vk.t()));
    drop(vk_inv);
    let kept = if with_kept {
        let mut vk_kept = resource::try_zeros2(l, keep.len(), "GMM generalized inverse")?;
        for (c, &j) in keep.iter().enumerate() {
            vk_kept
                .column_mut(c)
                .assign(&(&vectors.column(j) * values[j]));
        }
        Some(symmetrized(&vk_kept.dot(&vk.t())))
    } else {
        None
    };
    Ok(WeightInverse {
        inv,
        rank: keep.len(),
        kept,
    })
}

/// k x k inverse of a bread matrix X'Z A Z'X (`linalg::solve::sym_inverse`); singular is
/// `EconError::SingularMatrix` with the context "<what>: rank r of k".
fn small_inverse(m: &Array2<f64>, what: &str) -> EconResult<Array2<f64>> {
    match sym_inverse(symmetrized(m).view(), SingularPolicy::Error, what) {
        Ok(inv) => Ok(inv.inverse),
        Err(EconError::SingularMatrix { context, .. }) => Err(EconError::SingularMatrix {
            context,
            suggestion: "The regressors are collinear given the instruments (for example a \
                         time-invariant x, whose difference is zero); drop or recode them."
                .to_string(),
        }),
        Err(e) => Err(e),
    }
}

/// Two-sided standard-normal p-value of `z`; NaN for NaN.
fn normal_two_sided(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    Normal::new(0.0, 1.0).map_or(f64::NAN, |n| 2.0 * n.sf(z.abs()))
}

/// Upper-tail chi-squared p-value; NaN for df = 0 or a NaN statistic.
fn chi_squared_upper(stat: f64, df: usize) -> f64 {
    if df == 0 || stat.is_nan() {
        return f64::NAN;
    }
    crate::traits::estimator::chi_squared_p_value(stat.max(0.0), df as f64)
}

/// The non-zero cells of each row of Z, row by row.
struct SparseRows {
    ptr: Vec<usize>,
    col: Vec<usize>,
    val: Vec<f64>,
}

impl SparseRows {
    fn of(z: &Array2<f64>) -> EconResult<SparseRows> {
        let nnz = z.iter().filter(|v| **v != 0.0).count();
        let mut ptr = resource::try_with_capacity(z.nrows() + 1, "GMM instrument rows")?;
        let mut col = resource::try_with_capacity(nnz, "GMM instrument rows")?;
        let mut val = resource::try_with_capacity(nnz, "GMM instrument rows")?;
        ptr.push(0);
        for row in z.rows() {
            for (c, &v) in row.iter().enumerate() {
                if v != 0.0 {
                    col.push(c);
                    val.push(v);
                }
            }
            ptr.push(col.len());
        }
        Ok(SparseRows { ptr, col, val })
    }

    fn row(&self, r: usize) -> (&[usize], &[f64]) {
        let (a, b) = (self.ptr[r], self.ptr[r + 1]);
        (&self.col[a..b], &self.val[a..b])
    }
}

/// Σ_a z_a w_a over the rows of one entity block: a dense L vector, with the columns any
/// row of the block touches listed.
struct BlockSum {
    dense: Vec<f64>,
    mark: Vec<bool>,
    touched: Vec<usize>,
}

impl BlockSum {
    fn new(l: usize) -> EconResult<BlockSum> {
        Ok(BlockSum {
            dense: resource::try_vec(l, 0.0, "GMM moment vector")?,
            mark: resource::try_vec(l, false, "GMM moment vector")?,
            touched: resource::try_with_capacity(l, "GMM moment vector")?,
        })
    }

    /// Add `w` times a row; its columns count as touched even where `w` is 0.
    fn add(&mut self, cols: &[usize], vals: &[f64], w: f64) {
        for (&c, &v) in cols.iter().zip(vals) {
            if !self.mark[c] {
                self.mark[c] = true;
                self.touched.push(c);
            }
            self.dense[c] += v * w;
        }
    }

    fn clear(&mut self) {
        for &c in &self.touched {
            self.dense[c] = 0.0;
            self.mark[c] = false;
        }
        self.touched.clear();
    }
}

/// `s1 += h z_a z_b'` (and its transpose when a != b) over the non-zero cells of the rows.
fn add_moment_pair(s1: &mut Array2<f64>, sp: &SparseRows, a: usize, b: usize, h: f64) {
    let (ca, va) = sp.row(a);
    if a == b {
        for (&c, &v) in ca.iter().zip(va) {
            for (&d, &w) in ca.iter().zip(va) {
                s1[[c, d]] += h * v * w;
            }
        }
        return;
    }
    let (cb, vb) = sp.row(b);
    for (&c, &v) in ca.iter().zip(va) {
        for (&d, &w) in cb.iter().zip(vb) {
            s1[[c, d]] += h * v * w;
            s1[[d, c]] += h * v * w;
        }
    }
}

/// Σ_i Z_i' H_i Z_i with `plm:::FSM(., "full")`'s H ([`h_entry`]): 2 on the diagonal of
/// the difference rows and -1 between difference rows one period apart; for system GMM
/// also 1 on the diagonal of the level rows and the cross terms between the difference
/// row at t and the level rows at t (+1) and t - 1 (-1). H is sparse: each pair with a
/// non-zero entry is added once, as outer products of the sparse instrument rows.
fn one_step_moment_matrix(eq: &GmmEquations, sp: &SparseRows) -> EconResult<Array2<f64>> {
    let l = eq.z.ncols();
    let mut s1 = resource::try_zeros2(l, l, "GMM one-step weighting matrix")?;
    let row = |r: usize| (eq.kind[r], eq.time[r]);
    for block in &eq.blocks {
        resource::checkpoint()?;
        let diff = eq.difference_rows(block);
        let level = diff.end..block.end;
        let level_times = &eq.time[level.clone()];
        for a in diff.clone() {
            add_moment_pair(&mut s1, sp, a, a, h_entry(row(a), row(a)));
            let b = a + 1;
            if b < diff.end && eq.time[b] == eq.time[a] + 1 {
                add_moment_pair(&mut s1, sp, a, b, h_entry(row(a), row(b)));
            }
            // The level rows at t - 1 and t (ascending times).
            let t = eq.time[a];
            for s in [t - 1, t] {
                if let Ok(q) = level_times.binary_search(&s) {
                    let b = level.start + q;
                    add_moment_pair(&mut s1, sp, a, b, h_entry(row(a), row(b)));
                }
            }
        }
        for a in level {
            add_moment_pair(&mut s1, sp, a, a, h_entry(row(a), row(a)));
        }
    }
    Ok(s1)
}

/// Σ_i g_i g_i', g_i = Z_i' e_i.
fn two_step_moment_matrix(
    eq: &GmmEquations,
    sp: &SparseRows,
    e: &Array1<f64>,
    acc: &mut BlockSum,
) -> EconResult<Array2<f64>> {
    let l = eq.z.ncols();
    let mut s2 = resource::try_zeros2(l, l, "GMM two-step weighting matrix")?;
    for block in &eq.blocks {
        resource::checkpoint()?;
        for a in block.clone() {
            let (c, v) = sp.row(a);
            acc.add(c, v, e[a]);
        }
        for &c in &acc.touched {
            let gc = acc.dense[c];
            if gc == 0.0 {
                continue;
            }
            for &d in &acc.touched {
                s2[[c, d]] += gc * acc.dense[d];
            }
        }
        acc.clear();
    }
    Ok(s2)
}

/// Windmeijer's (2005) D (k x k), as `plm::vcovHC.pgmm`'s Dk: column j is
/// -B2 ZX' W2 q_j, q_j = -Σ_i [a_ij (g_i·u) + g_i (a_ij·u)], a_ij = Z_i' X_i[:, j],
/// g_i = Z_i' e1_i, u = W2 Z' e2, and `r` = B2 ZX' W2.
fn windmeijer_d(
    eq: &GmmEquations,
    sp: &SparseRows,
    e1: &Array1<f64>,
    u: &Array1<f64>,
    r: &Array2<f64>,
    acc: &mut BlockSum,
) -> EconResult<Array2<f64>> {
    let (l, k) = (eq.z.ncols(), eq.x.ncols());
    let mut q = resource::try_zeros2(l, k, "GMM Windmeijer correction")?;
    let mut a = resource::try_zeros2(l, k, "GMM Windmeijer correction")?;
    let mut au = vec![0.0; k];
    for block in &eq.blocks {
        resource::checkpoint()?;
        for row in block.clone() {
            let (c, v) = sp.row(row);
            acc.add(c, v, e1[row]);
            for j in 0..k {
                let xj = eq.x[[row, j]];
                for (&cc, &vv) in c.iter().zip(v) {
                    a[[cc, j]] += vv * xj;
                }
            }
        }
        let gu: f64 = acc.touched.iter().map(|&c| acc.dense[c] * u[c]).sum();
        for (j, auj) in au.iter_mut().enumerate() {
            *auj = acc.touched.iter().map(|&c| a[[c, j]] * u[c]).sum();
        }
        for &c in &acc.touched {
            for j in 0..k {
                q[[c, j]] -= a[[c, j]] * gu + acc.dense[c] * au[j];
                a[[c, j]] = 0.0;
            }
        }
        acc.clear();
    }
    Ok(-r.dot(&q))
}

/// Whether some entity has two equations `order` periods apart.
fn has_lag_pair(eq: &GmmEquations, order: i64) -> bool {
    eq.blocks.iter().any(|block| {
        let times = &eq.time[eq.difference_rows(block)];
        times
            .iter()
            .enumerate()
            .any(|(p, &t)| times[..p].binary_search(&(t - order)).is_ok())
    })
}

/// The Arellano-Bond m-test of order `order`, as `plm::mtest`, on the differenced
/// residuals (for system GMM the level rows do not enter): (statistic, p-value), NaN
/// when the variance estimate is not positive (no entity has two equations `order`
/// periods apart, or the estimate is negative). `b` is the bread (X'Z W Z'X)^-1, `w` the
/// weighting matrix and `v_robust` the robust coefficient covariance of the fit whose
/// residuals are `e`.
fn ab_mtest(
    e: &[f64],
    eq: &GmmEquations,
    order: i64,
    b: &Array2<f64>,
    w: &Array2<f64>,
    v_robust: &Array2<f64>,
    zx: &Array2<f64>,
) -> (f64, f64) {
    let (k, l) = (eq.x.ncols(), eq.z.ncols());
    let mut num = 0.0;
    let mut eve = 0.0;
    let mut ex = Array1::<f64>::zeros(k);
    let mut zve = Array1::<f64>::zeros(l);
    for block in &eq.blocks {
        let diff = eq.difference_rows(block);
        let times = &eq.time[diff.clone()];
        let mut s = 0.0;
        for (p, a) in diff.clone().enumerate() {
            if let Ok(q) = times[..p].binary_search(&(times[p] - order)) {
                let lagged = e[diff.start + q];
                s += e[a] * lagged;
                ex.scaled_add(lagged, &eq.x.row(a));
            }
        }
        num += s;
        eve += s * s;
        if s != 0.0 {
            for a in diff {
                zve.scaled_add(e[a] * s, &eq.z.row(a));
            }
        }
    }
    let denom = eve - 2.0 * ex.dot(&b.dot(&zx.t().dot(&w.dot(&zve)))) + ex.dot(&v_robust.dot(&ex));
    if !(denom > 0.0 && denom.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let stat = num / denom.sqrt();
    (stat, normal_two_sided(stat))
}

/// "'a', 'b', 'c' (and 4 more)": at most five labels.
fn label_list(labels: &[String]) -> String {
    let shown: Vec<String> = labels.iter().take(5).map(|l| format!("'{l}'")).collect();
    let mut text = shown.join(", ");
    if labels.len() > 5 {
        text.push_str(&format!(" (and {} more)", labels.len() - 5));
    }
    text
}

/// Arellano-Bond difference GMM, as `plm::pgmm(transformation = "d")`, or Blundell-Bond
/// system GMM with `config.transform = System`, as `plm::pgmm(transformation = "ld")`
/// (see the module docs for the conventions).
///
/// # Arguments
///
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the strictly exogenous regressors (not the lagged y: `lags`
///   adds it); they enter in differences and serve as their own instruments
/// * `entity_col` - Column name for entity/individual identifier (numbers or text)
/// * `time_col` - Column name for the time period: whole numbers (a Date column counts
///   days, so recode dates as period numbers); lags are steps of 1
/// * `lags` - Number of lags of the dependent variable among the regressors
/// * `config` - GMM configuration options (default: two-step, Windmeijer SEs)
///
/// # Returns
///
/// A `GmmResult` containing the estimated coefficients, standard errors,
/// test statistics, and diagnostics.
pub fn run_gmm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
    lags: usize,
    config: Option<GmmConfig>,
) -> EconResult<GmmResult> {
    let config = config.unwrap_or_default();
    let mut warnings = Vec::new();
    let system = config.transform == GmmTransform::System;

    if lags == 0 && x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "GMM has no regressors: set lags to at least 1 or name x columns".to_string(),
        });
    }
    if system && config.step == GmmStep::OneStep && !config.robust {
        return Err(EconError::InvalidSpecification {
            message: "System GMM has no classical one-step standard errors: the one-step \
                      weight H is the error covariance of the level equations only when the \
                      individual effect has zero variance (Blundell and Bond 1998). Use \
                      robust = true for one-step robust standard errors, or step = two-step."
                .to_string(),
        });
    }
    let cols = ColumnRequest::new(dataset.df())
        .factor(entity_col)
        .time_index(time_col)
        .numeric(y_col)
        .numerics(x_cols)
        .na(NaPolicy::Omit)
        .extract()?;
    let spec = GmmLagSpec {
        lags,
        min_lag: config.min_lag,
        max_lag: config.max_lag,
        collapse: config.collapse,
        system,
    };
    let mut eq = gmm_equations(&cols, y_col, x_cols, entity_col, time_col, &spec)?;
    let (n_obs, k) = eq.x.dim();
    let l = eq.z.ncols();
    let (n_obs_diff, n_obs_level) = (eq.n_diff, eq.n_level);
    if config.min_lag == 1 {
        warnings.push(
            "min_lag = 1 uses y_{t-1} as an instrument, which is correlated with the \
             differenced error when the errors are serially uncorrelated; the default is 2."
                .to_string(),
        );
    }
    if !eq.entities_without_equations.is_empty() {
        let m = eq.entities_without_equations.len();
        let (noun, verb) = if m == 1 {
            ("entity", "contributes")
        } else {
            ("entities", "contribute")
        };
        warnings.push(format!(
            "{m} {noun} {verb} no equation (no {} consecutive periods): {}",
            if system { lags + 1 } else { lags + 2 },
            label_list(&eq.entities_without_equations)
        ));
    }

    ensure_dense_inverse_fits(l, "GMM weighting matrix")?;
    let sp = SparseRows::of(&eq.z)?;
    let mut acc = BlockSum::new(l)?;
    let lhs = Array1::from_vec(std::mem::take(&mut eq.lhs));
    let zx = eq.z.t().dot(&eq.x);
    let zy = eq.z.t().dot(&lhs);

    // One step.
    let w1 = weight_inverse(&one_step_moment_matrix(&eq, &sp)?, false)?;
    let w1zx = w1.inv.dot(&zx);
    let b1 = small_inverse(&zx.t().dot(&w1zx), "X'Z A Z'X in one-step GMM")?;
    let beta1 = b1.dot(&w1zx.t().dot(&zy));
    let e1 = &lhs - &eq.x.dot(&beta1);

    // Two-step weights from the one-step residuals: the weights of the Sargan/Hansen test
    // of either step, and of the two-step estimates.
    let w2 = weight_inverse(&two_step_moment_matrix(&eq, &sp, &e1, &mut acc)?, true)?;
    let two_step = match config.step {
        GmmStep::OneStep => None,
        GmmStep::TwoStep => {
            let w2zx = w2.inv.dot(&zx);
            let b2 = small_inverse(&zx.t().dot(&w2zx), "X'Z A Z'X in two-step GMM")?;
            let beta2 = b2.dot(&w2zx.t().dot(&zy));
            let e2 = &lhs - &eq.x.dot(&beta2);
            Some((w2zx, b2, beta2, e2))
        }
    };

    if w1.rank < l {
        warnings.push(format!(
            "The one-step weighting matrix (sum of Z_i' H Z_i) has numerical rank {} of {l} \
             instruments; a generalized inverse is used, as plm::pgmm does. Reduce the \
             instruments with max_lag or collapse.",
            w1.rank
        ));
    }
    if w2.rank < l {
        warnings.push(match config.step {
            GmmStep::OneStep => format!(
                "The Sargan/Hansen weighting matrix (sum of Z_i' e_i e_i' Z_i) has numerical \
                 rank {} of {l} instruments; a generalized inverse is used, as plm::sargan \
                 does. Reduce the instruments with max_lag or collapse.",
                w2.rank
            ),
            GmmStep::TwoStep => format!(
                "The two-step weighting matrix (sum of Z_i' e_i e_i' Z_i) has numerical rank \
                 {} of {l} instruments; a generalized inverse is used, as plm::pgmm does. \
                 Two-step GMM needs at least as many entities as instruments: reduce them \
                 with max_lag or collapse.",
                w2.rank
            ),
        });
    }

    // Covariances. σ² from the one-step residuals of the difference equations.
    let ss_diff = if system {
        eq.kind
            .iter()
            .zip(&e1)
            .filter(|(kind, _)| **kind == EqKind::Difference)
            .map(|(_, &e)| e * e)
            .sum()
    } else {
        e1.dot(&e1)
    };
    let sigma2 = match n_obs_diff.checked_sub(k) {
        Some(df) if df > 0 => ss_diff / (2.0 * df as f64),
        _ => f64::NAN,
    };
    let w2_kept = w2.kept.as_ref().ok_or_else(|| {
        EconError::Computation("GMM: the two-step weighting matrix has no kept part".to_string())
    })?;
    let v1r = b1.dot(&w1zx.t().dot(w2_kept).dot(&w1zx)).dot(&b1);
    // The reported step, and the robust covariance its m-tests use (pgmm's summary
    // default): one-step vcovHC, two-step Windmeijer.
    let (beta, resid, bread, weight, v_robust) = match &two_step {
        None => (&beta1, &e1, &b1, &w1.inv, v1r.clone()),
        Some((w2zx, b2, beta2, e2)) => {
            let u = w2.inv.dot(&eq.z.t().dot(e2));
            let r = b2.dot(&w2zx.t());
            let d = windmeijer_d(&eq, &sp, &e1, &u, &r, &mut acc)?;
            let db2 = d.dot(b2);
            let mut v2r = d.dot(&v1r).dot(&d.t());
            v2r += b2;
            v2r += &db2;
            v2r += &db2.t();
            (beta2, e2, b2, &w2.inv, v2r)
        }
    };
    let (vcov, se_type) = match (config.step, config.robust) {
        (GmmStep::OneStep, false) => (&b1 * sigma2, "one-step, homoskedastic (Arellano-Bond 1991)"),
        (GmmStep::OneStep, true) => (
            v_robust.clone(),
            "one-step, robust (plm vcovHC: heteroskedasticity and within-entity correlation)",
        ),
        (GmmStep::TwoStep, false) => (
            bread.clone(),
            "two-step, uncorrected (biased downward in finite samples)",
        ),
        (GmmStep::TwoStep, true) => (v_robust.clone(), "two-step, Windmeijer (2005) corrected"),
    };

    let diag = vcov.diag();
    if diag.iter().any(|&v| v < 0.0) {
        warnings.push("A coefficient variance is negative; its standard error is NaN.".to_string());
    }
    let std_errors: Vec<f64> = diag
        .iter()
        .map(|&v| if v >= 0.0 { v.sqrt() } else { f64::NAN })
        .collect();
    let coefficients = beta.to_vec();
    let z_stats: Vec<f64> = coefficients
        .iter()
        .zip(&std_errors)
        .map(|(&b, &se)| if se > 0.0 { b / se } else { f64::NAN })
        .collect();
    let p_values: Vec<f64> = z_stats.iter().map(|&z| normal_two_sided(z)).collect();
    let significance = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Sargan/Hansen (plm::sargan: two-step weights, the reported step's residuals).
    let sargan_df = l - k;
    let ze = eq.z.t().dot(resid);
    let sargan_statistic = ze.dot(&w2.inv.dot(&ze));
    let sargan_p_value = chi_squared_upper(sargan_statistic, sargan_df);
    if sargan_df == 0 {
        warnings.push(
            "The model is exactly identified (as many instruments as parameters): the \
             Sargan/Hansen test has no degrees of freedom and its p-value is NaN."
                .to_string(),
        );
    }
    let (sargan_homoskedastic, sargan_homoskedastic_p_value) = match config.step {
        GmmStep::OneStep if !system => {
            let z1 = eq.z.t().dot(&e1);
            let stat = z1.dot(&w1.inv.dot(&z1)) / sigma2;
            (Some(stat), Some(chi_squared_upper(stat, sargan_df)))
        }
        _ => (None, None),
    };

    // Arellano-Bond m-tests.
    let resid_values = resid.to_vec();
    let mut m_test = |order: i64| {
        let (stat, p) = ab_mtest(&resid_values, &eq, order, bread, weight, &v_robust, &zx);
        if stat.is_nan() {
            warnings.push(if has_lag_pair(&eq, order) {
                format!("AR({order}) test is undefined: its variance estimate is not positive.")
            } else {
                format!(
                    "AR({order}) test is undefined: no entity has two equations {order} \
                     periods apart."
                )
            });
        }
        (stat, p)
    };
    let (ar1_statistic, ar1_p_value) = m_test(1);
    let (ar2_statistic, ar2_p_value) = m_test(2);

    let missing = cols.missing().clone();
    if let Some(note) = missing.note() {
        warnings.push(note);
    }

    let mut variables = Vec::with_capacity(k);
    for lag in 1..=lags {
        variables.push(format!("L{}.{}", lag, y_col));
    }
    variables.extend(x_cols.iter().map(|x| x.to_string()));

    Ok(GmmResult {
        transform: config.transform,
        step: config.step,
        dep_var: if system {
            y_col.to_string()
        } else {
            format!("D.{}", y_col)
        },
        variables,
        coef_arr: Array1::from_vec(coefficients.clone()),
        se_arr: Array1::from_vec(std_errors.clone()),
        coefficients,
        std_errors,
        z_stats,
        p_values,
        significance,
        n_obs,
        n_groups: eq.blocks.len(),
        n_instruments: l,
        sargan_statistic,
        sargan_p_value,
        sargan_df,
        ar1_statistic,
        ar1_p_value,
        ar2_statistic,
        ar2_p_value,
        entity_var: entity_col.to_string(),
        time_var: time_col.to_string(),
        warnings,
        se_type: se_type.to_string(),
        sigma2,
        sargan_homoskedastic,
        sargan_homoskedastic_p_value,
        one_step_weight_rank: w1.rank,
        two_step_weight_rank: w2.rank,
        one_step_coefficients: beta1.to_vec(),
        lags,
        min_lag: config.min_lag,
        max_lag: config.max_lag,
        collapse: config.collapse,
        missing,
        n_obs_diff,
        n_obs_level,
        residuals: resid.clone(),
        vcov,
    })
}

/// Arellano-Bond difference GMM with defaults: one lag of the dependent variable,
/// two-step estimates with Windmeijer-corrected standard errors.
pub fn run_arellano_bond(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<GmmResult> {
    run_gmm(dataset, y_col, x_cols, entity_col, time_col, 1, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    const GRUNFELD_CSV: &str = include_str!("../../../../../validation/datasets/grunfeld.csv");

    fn grunfeld() -> Dataset {
        Dataset::new(crate::data::DataLoader::from_csv_string(GRUNFELD_CSV).unwrap())
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// A balanced synthetic panel of `n_ent` entities and `n_t` periods (the
    /// same generator as `gls_models::tests::synthetic_panel`).
    fn synthetic_panel(n_ent: usize, n_t: usize) -> Dataset {
        let n = n_ent * n_t;
        let entity: Vec<i64> = (0..n).map(|i| (i / n_t) as i64).collect();
        let time: Vec<i64> = (0..n).map(|i| (i % n_t) as i64 + 2000).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 101) as f64 / 10.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                1.0 + 0.5 * x[i]
                    + ((i / n_t) % 17) as f64 * 0.1
                    + ((i * 7919) % 1009) as f64 / 1009.0
                    - 0.5
            })
            .collect();
        Dataset::new(
            polars::prelude::df! { "entity" => entity, "time" => time, "x" => x, "y" => y }
                .unwrap(),
        )
    }

    #[test]
    fn test_gmm_transform_parses_aliases_and_rejects_unknown() {
        for s in [
            "difference",
            "Difference",
            " diff ",
            "d",
            "Arellano-Bond",
            "arellano bond",
        ] {
            assert_eq!(
                s.parse::<GmmTransform>().unwrap(),
                GmmTransform::Difference,
                "{s}"
            );
        }
        for s in [
            "system",
            "System",
            "SYS",
            "ld",
            "Blundell-Bond",
            "blundell_bond",
        ] {
            assert_eq!(
                s.parse::<GmmTransform>().unwrap(),
                GmmTransform::System,
                "{s}"
            );
        }
        match "levels".parse::<GmmTransform>() {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "unknown transform 'levels'; expected one of: difference, system"
            ),
            other => panic!("{other:?}"),
        }
        assert!("".parse::<GmmTransform>().is_err());
        assert_eq!(GmmTransform::choices().validate(), Ok(()));
        assert_eq!(
            GmmTransform::System.to_string(),
            "System GMM (Blundell-Bond)"
        );
    }

    #[test]
    fn test_gmm_step_parses_aliases_and_rejects_unknown() {
        for s in ["onestep", "OneStep", "one-step", "One Step", "one", "1"] {
            assert_eq!(s.parse::<GmmStep>().unwrap(), GmmStep::OneStep, "{s}");
        }
        for s in ["twostep", "two_step", "twosteps", "Two", "2"] {
            assert_eq!(s.parse::<GmmStep>().unwrap(), GmmStep::TwoStep, "{s}");
        }
        match "three".parse::<GmmStep>() {
            Err(EconError::InvalidSpecification { message }) => assert_eq!(
                message,
                "unknown step 'three'; expected one of: onestep, twostep"
            ),
            other => panic!("{other:?}"),
        }
        assert_eq!(GmmStep::choices().validate(), Ok(()));
        assert_eq!(GmmStep::TwoStep.to_string(), "Two-step");
    }

    #[test]
    fn test_gmm_rejects_max_lag_below_min_lag() {
        let cfg = GmmConfig {
            min_lag: 3,
            max_lag: Some(2),
            ..Default::default()
        };
        let e = run_gmm(
            &grunfeld(),
            "inv",
            &["value", "capital"],
            "firm",
            "year",
            1,
            Some(cfg),
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("max_lag")),
            "{e:?}"
        );
    }

    #[test]
    fn test_gmm_huge_lags_is_an_error_not_a_panic() {
        let e = run_gmm(
            &grunfeld(),
            "inv",
            &["value"],
            "firm",
            "year",
            usize::MAX,
            None,
        )
        .unwrap_err();
        assert!(matches!(e, EconError::InsufficientData { .. }), "{e:?}");
    }

    #[test]
    fn test_gmm_instrument_matrix_respects_alloc_ceiling() {
        // Grunfeld's full instrument matrix is larger than the 64 KiB ceiling set here.
        let e = resource::with_alloc_ceiling(Some(64 << 10), || {
            run_gmm(
                &grunfeld(),
                "inv",
                &["value", "capital"],
                "firm",
                "year",
                1,
                None,
            )
        })
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_gmm_stops_when_cancelled() {
        let e = resource::with_budget(cancelled(), || {
            run_arellano_bond(&grunfeld(), "inv", &["value", "capital"], "firm", "year")
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_gmm_scaling_many_entities() {
        if cfg!(debug_assertions) {
            return; // release-only timing check
        }
        let ds = synthetic_panel(50_000, 5);
        let t0 = std::time::Instant::now();
        let r = run_arellano_bond(&ds, "y", &["x"], "entity", "time").unwrap();
        assert_eq!(r.n_groups, 50_000);
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(10),
            "took {:?}",
            t0.elapsed()
        );
    }

    // ── pgmm references (validation/r/cases/econ_dynamic_panel.R) ──────────────────────

    use p2a_validation::{Check, Tol};
    use polars::prelude::Column;

    /// A committed CSV loaded as `p2a data load` loads it; for the property tests, which
    /// compare no R value and so load no case.
    fn repo_dataset(rel: &str) -> Dataset {
        Dataset::new(crate::data::DataLoader::load_csv(p2a_validation::repo_path(rel)).unwrap())
    }

    const AB_BALANCED: &str = "validation/datasets/generated/ab_balanced.csv";
    const AB_COMMON_GAP: &str = "validation/datasets/generated/ab_common_gap.csv";

    /// `ds` with column `column` added or replaced.
    fn with_column(ds: &Dataset, column: Column) -> Dataset {
        let mut df = ds.df().clone();
        df.with_column(column).unwrap();
        Dataset::new(df)
    }

    #[allow(clippy::too_many_arguments)]
    fn fit(
        ds: &Dataset,
        y: &str,
        x: &[&str],
        e: &str,
        t: &str,
        lags: usize,
        step: GmmStep,
        robust: bool,
        min_lag: usize,
        max_lag: Option<usize>,
        collapse: bool,
    ) -> GmmResult {
        let cfg = GmmConfig {
            transform: GmmTransform::Difference,
            step,
            max_lag,
            min_lag,
            collapse,
            robust,
        };
        run_gmm(ds, y, x, e, t, lags, Some(cfg)).unwrap_or_else(|err| panic!("{err}"))
    }

    /// The tolerances of one case.
    #[derive(Clone, Copy)]
    struct Tols {
        coef: Tol,
        se: Tol,
        stat: Tol,
        p: Tol,
    }

    impl Tols {
        /// validation/README.md for `n` equations.
        fn n(n: usize) -> Tols {
            Tols {
                coef: Tol::coef(n),
                se: Tol::se(n),
                stat: Tol::stat(n),
                p: Tol::p_value(n),
            }
        }
    }

    /// The quantities of `{p}_*` that do not depend on the SE type.
    fn check_common(c: &mut Check, p: &str, names: &[&str], r: &GmmResult, t: Tols) {
        let exact = Tol::exact();
        c.named(&format!("{p}_coef"), names, &r.coefficients, t.coef)
            .scalar(&format!("{p}_sargan"), r.sargan_statistic, t.stat)
            .scalar(&format!("{p}_sargan_df_nonzero"), r.sargan_df as f64, exact)
            .scalar(&format!("{p}_sargan_p_nonzero"), r.sargan_p_value, t.p)
            .scalar(&format!("{p}_ar1"), r.ar1_statistic, t.stat)
            .scalar(&format!("{p}_ar1_p"), r.ar1_p_value, t.p)
            .scalar(&format!("{p}_ar2"), r.ar2_statistic, t.stat)
            .scalar(&format!("{p}_ar2_p"), r.ar2_p_value, t.p)
            .scalar(&format!("{p}_n_obs"), r.n_obs as f64, exact)
            .scalar(
                &format!("{p}_n_instruments_nonzero"),
                r.n_instruments as f64,
                exact,
            );
    }

    /// A one-step fit with Arellano-Bond (1991) classical SEs (`r_cl`) and with robust
    /// SEs (`r_rb`) against `{p}_*`.
    fn check_onestep(
        c: &mut Check,
        p: &str,
        names: &[&str],
        r_cl: &GmmResult,
        r_rb: &GmmResult,
        t: Tols,
    ) {
        check_common(c, p, names, r_cl, t);
        check_common(c, p, names, r_rb, t);
        c.named(&format!("{p}_se_textbook"), names, &r_cl.std_errors, t.se)
            .named(&format!("{p}_se_robust"), names, &r_rb.std_errors, t.se)
            .scalar(&format!("{p}_sigma2"), r_cl.sigma2, t.se)
            .scalar(
                &format!("{p}_sargan_homoskedastic"),
                r_cl.sargan_homoskedastic.unwrap_or(f64::NAN),
                t.stat,
            )
            .scalar(
                &format!("{p}_sargan_homoskedastic_p"),
                r_cl.sargan_homoskedastic_p_value.unwrap_or(f64::NAN),
                t.p,
            );
        assert_eq!(r_cl.step, GmmStep::OneStep);
        assert!(
            r_cl.se_type.contains("Arellano-Bond 1991"),
            "{}",
            r_cl.se_type
        );
        assert!(r_rb.se_type.contains("robust"), "{}", r_rb.se_type);
    }

    /// A two-step fit with uncorrected (`r_cl`) and Windmeijer SEs (`r_rb`) against
    /// `{p}_*`.
    fn check_twostep(
        c: &mut Check,
        p: &str,
        names: &[&str],
        r_cl: &GmmResult,
        r_rb: &GmmResult,
        t: Tols,
    ) {
        check_common(c, p, names, r_cl, t);
        check_common(c, p, names, r_rb, t);
        c.named(&format!("{p}_se"), names, &r_cl.std_errors, t.se)
            .named(&format!("{p}_se_robust"), names, &r_rb.std_errors, t.se);
        assert_eq!(r_cl.step, GmmStep::TwoStep);
        assert_eq!(r_cl.sargan_homoskedastic, None);
        assert_eq!(r_rb.se_type, "two-step, Windmeijer (2005) corrected");
    }

    /// The four fits of `onestep_*` and `twostep_*`: (one-step classical, one-step
    /// robust, two-step uncorrected, two-step Windmeijer).
    fn four_fits(ds: &Dataset, y: &str, x: &[&str], e: &str, t: &str) -> [GmmResult; 4] {
        [
            fit(ds, y, x, e, t, 1, GmmStep::OneStep, false, 2, None, false),
            fit(ds, y, x, e, t, 1, GmmStep::OneStep, true, 2, None, false),
            fit(ds, y, x, e, t, 1, GmmStep::TwoStep, false, 2, None, false),
            fit(ds, y, x, e, t, 1, GmmStep::TwoStep, true, 2, None, false),
        ]
    }

    fn check_four(c: &mut Check, names: &[&str], fits: &[GmmResult; 4], t: Tols) {
        check_onestep(c, "onestep", names, &fits[0], &fits[1], t);
        check_twostep(c, "twostep", names, &fits[2], &fits[3], t);
    }

    #[test]
    fn test_validate_gmm_ab_balanced() {
        let (case, ds) = crate::test_support::load_case("dpd_ab_balanced");
        let fits = four_fits(&ds, "y", &["x"], "id", "year");
        let mut c = Check::new(&case);
        check_four(&mut c, &["lag(y, 1)", "x"], &fits, Tols::n(600));
        c.finish();
        for r in &fits {
            assert_eq!(r.variables, ["L1.y", "x"]);
            assert_eq!(r.n_groups, 100);
            assert_eq!((r.one_step_weight_rank, r.two_step_weight_rank), (22, 22));
            assert!(r.warnings.is_empty(), "{:?}", r.warnings);
            assert!(r.missing.is_empty());
            assert_eq!(r.transform, GmmTransform::Difference);
        }
        assert_eq!(fits[2].one_step_coefficients, fits[0].coefficients);
    }

    #[test]
    fn test_validate_gmm_ab_balanced_variants() {
        let (case, ds) = crate::test_support::load_case("dpd_ab_balanced");
        let mut c = Check::new(&case);
        let two = |lags, max_lag, collapse| {
            [false, true].map(|robust| {
                fit(
                    &ds,
                    "y",
                    &["x"],
                    "id",
                    "year",
                    lags,
                    GmmStep::TwoStep,
                    robust,
                    2,
                    max_lag,
                    collapse,
                )
            })
        };
        let [cl, rb] = two(2, None, false);
        check_twostep(
            &mut c,
            "lags2_twostep",
            &["lag(y, 1:2)1", "lag(y, 1:2)2", "x"],
            &cl,
            &rb,
            Tols::n(500),
        );
        assert_eq!(rb.variables, ["L1.y", "L2.y", "x"]);
        let [cl, rb] = two(1, Some(3), false);
        check_twostep(
            &mut c,
            "maxlag3_twostep",
            &["lag(y, 1)", "x"],
            &cl,
            &rb,
            Tols::n(600),
        );
        let [cl, rb] = two(1, None, true);
        check_twostep(
            &mut c,
            "collapse_twostep",
            &["lag(y, 1)", "x"],
            &cl,
            &rb,
            Tols::n(600),
        );
        assert!(
            rb.collapse
                && rb
                    .to_string()
                    .contains("collapsed: one column per lag depth")
        );

        let one = |x: &[&str], min_lag| {
            [false, true].map(|robust| {
                fit(
                    &ds,
                    "y",
                    x,
                    "id",
                    "year",
                    1,
                    GmmStep::OneStep,
                    robust,
                    min_lag,
                    None,
                    false,
                )
            })
        };
        let [cl, rb] = one(&["x"], 3);
        check_onestep(
            &mut c,
            "minlag3_onestep",
            &["lag(y, 1)", "x"],
            &cl,
            &rb,
            Tols::n(500),
        );
        let [cl, rb] = one(&[], 2);
        check_onestep(
            &mut c,
            "nox_onestep",
            &["lag(y, 1)"],
            &cl,
            &rb,
            Tols::n(600),
        );
        assert_eq!(rb.variables, ["L1.y"]);
        c.finish();
    }

    #[test]
    fn test_validate_gmm_ab_gaps() {
        let (case, ds) = crate::test_support::load_case("dpd_ab_gaps");
        let fits = four_fits(&ds, "y", &["x"], "id", "year");
        let mut c = Check::new(&case);
        check_four(&mut c, &["lag(y, 1)", "x"], &fits, Tols::n(553));
        let two = |lags, collapse| {
            [false, true].map(|robust| {
                fit(
                    &ds,
                    "y",
                    &["x"],
                    "id",
                    "year",
                    lags,
                    GmmStep::TwoStep,
                    robust,
                    2,
                    None,
                    collapse,
                )
            })
        };
        let [cl, rb] = two(2, false);
        check_twostep(
            &mut c,
            "lags2_twostep",
            &["lag(y, 1:2)1", "lag(y, 1:2)2", "x"],
            &cl,
            &rb,
            Tols::n(442),
        );
        let [cl, rb] = two(1, true);
        check_twostep(
            &mut c,
            "collapse_twostep",
            &["lag(y, 1)", "x"],
            &cl,
            &rb,
            Tols::n(553),
        );
        c.finish();
        for r in &fits {
            assert_eq!(r.n_groups, 100);
            assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        }
    }

    /// A year no entity observes (2003), missing y and x cells, text entity ids.
    #[test]
    fn test_validate_gmm_ab_common_gap() {
        let (case, ds) = crate::test_support::load_case("dpd_ab_common_gap");
        let mut c = Check::new(&case);
        let mut unit_fits = Vec::new();
        for entity in ["id", "unit"] {
            let fits = four_fits(&ds, "y", &["x"], entity, "year");
            check_four(&mut c, &["lag(y, 1)", "x"], &fits, Tols::n(260));
            for r in &fits {
                c.scalar("n_rows_dropped", r.missing.n_dropped as f64, Tol::exact());
                assert_eq!(
                    r.missing.by_column,
                    [("y".to_string(), 10), ("x".to_string(), 10)]
                );
                let note = r.missing.note().unwrap();
                assert!(r.warnings.contains(&note), "{:?}", r.warnings);
                assert!(r.ar2_statistic.is_nan() && r.ar2_p_value.is_nan());
                assert!(
                    r.warnings
                        .iter()
                        .any(|w| w.contains("AR(2) test is undefined")),
                    "{:?}",
                    r.warnings
                );
                assert_eq!((r.n_instruments, r.sargan_df), (11, 9));
                // The note is the last line of the text, and only there.
                let text = r.to_string();
                assert!(
                    text.trim_end()
                        .ends_with(&format!("Missing values: {note}")),
                    "{text}"
                );
                assert_eq!(text.matches(note.as_str()).count(), 1, "{text}");
            }
            unit_fits.push(fits);
        }
        c.finish();
        // Text ids (u001..u100, C order) are the numeric ids in the same order.
        for (a, b) in unit_fits[0].iter().zip(&unit_fits[1]) {
            assert_eq!(a.coefficients, b.coefficients);
            assert_eq!(a.std_errors, b.std_errors);
        }
    }

    /// Grunfeld: 173 instruments for 10 firms, so both weighting matrices are singular
    /// and get pgmm's generalized inverse.
    #[test]
    fn test_validate_gmm_ab_grunfeld() {
        let (case, ds) = crate::test_support::load_case("dpd_ab_grunfeld");
        let fits = four_fits(&ds, "inv", &["value", "capital"], "firm", "year");
        let mut c = Check::new(&case);
        // Generalized inverses of rank-deficient weighting matrices (eigen- vs SVD-based
        // truncation): the coefficients agree with pgmm to 3e-12, within validation/README.md's
        // Tol::coef(180); the SEs agree to 1.6e-10 and the statistics and p-values to 6e-11, so
        // their relative bounds are 1e-7, ten times tighter than README's for n = 180 (README's
        // absolute parts).
        let t = Tols::n(180);
        let t = Tols {
            se: Tol::new(1e-7, t.se.abs),
            stat: Tol::new(1e-7, t.stat.abs),
            p: Tol::new(1e-7, t.p.abs),
            ..t
        };
        check_four(&mut c, &["lag(inv, 1)", "value", "capital"], &fits, t);
        for r in &fits {
            c.scalar(
                "rank_one_step_weight",
                r.one_step_weight_rank as f64,
                Tol::exact(),
            )
            .scalar(
                "rank_two_step_weight",
                r.two_step_weight_rank as f64,
                Tol::exact(),
            );
            assert_eq!((r.n_instruments, r.sargan_df), (173, 170));
            for rank in ["rank 133 of 173", "rank 9 of 173"] {
                assert!(
                    r.warnings.iter().any(|w| w.contains(rank)),
                    "{:?}",
                    r.warnings
                );
            }
        }
        c.finish();
    }

    /// The check before the weight inversions counts their whole working set: a panel
    /// whose instrument count passes a count of six L x L matrices but not the measured
    /// peak is refused under a per-allocation ceiling between the two, and runs without one.
    #[test]
    fn test_gmm_dense_inverse_check_counts_the_measured_peak() {
        let ds = synthetic_panel(4, 20);
        let fit = || run_gmm(&ds, "y", &["x"], "entity", "time", 1, None);
        let r = fit().unwrap();
        let l = r.n_instruments;
        let ceiling = 2usize << 20;
        assert!(
            6 * 8 * l * l < ceiling && 16 * 8 * l * l > ceiling,
            "L = {l}"
        );
        let e = resource::with_alloc_ceiling(Some(ceiling), fit).unwrap_err();
        assert!(
            matches!(&e, EconError::ResourceLimit(m) if m.contains("GMM weighting matrix")),
            "{e:?}"
        );
    }

    /// A three-period panel with one lag is the smallest that pgmm fits: one
    /// differenced equation per entity, instrumented by the first period's level.
    #[test]
    fn test_gmm_fits_a_three_period_panel() {
        let ds = repo_dataset(AB_BALANCED);
        let keep: polars::prelude::BooleanChunked = ds
            .df()
            .column("year")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|y| y.map(|y| y <= 2002))
            .collect();
        let three = Dataset::new(ds.df().filter(&keep).unwrap());
        for step in [GmmStep::OneStep, GmmStep::TwoStep] {
            let cfg = GmmConfig {
                step,
                ..Default::default()
            };
            let r = run_gmm(&three, "y", &["x"], "id", "year", 1, Some(cfg)).unwrap();
            assert_eq!((r.n_obs, r.n_groups), (100, 100), "{step:?}");
            assert!(
                r.coefficients
                    .iter()
                    .chain(&r.std_errors)
                    .all(|v| v.is_finite()),
                "{step:?}: {:?} {:?}",
                r.coefficients,
                r.std_errors
            );
        }
    }

    #[test]
    fn test_gmm_duplicate_entity_time_is_an_error() {
        let ds = repo_dataset(AB_BALANCED);
        let mut df = ds.df().clone();
        df.vstack_mut(&ds.df().slice(2, 1)).unwrap();
        let e = run_arellano_bond(&Dataset::new(df), "y", &["x"], "id", "year").unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("Entity '1' has 2 rows for time 2002")),
            "{e:?}"
        );
    }

    #[test]
    fn test_gmm_lags_use_time_steps_of_one() {
        let ds = repo_dataset(AB_BALANCED);
        let years: Vec<i64> = ds
            .df()
            .column("year")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|y| 2000 + 2 * (y.unwrap() - 2000))
            .collect();
        let biennial = with_column(&ds, Column::new("year".into(), years));
        let e = run_arellano_bond(&biennial, "y", &["x"], "id", "year").unwrap_err();
        assert!(
            matches!(&e, EconError::InsufficientData { context, .. }
                if context.contains("recode them as consecutive period numbers")),
            "{e:?}"
        );
    }

    #[test]
    fn test_gmm_fractional_time_is_an_error() {
        let ds = repo_dataset(AB_BALANCED);
        let mut years: Vec<f64> = ds
            .df()
            .column("year")
            .unwrap()
            .i64()
            .unwrap()
            .iter()
            .map(|y| y.unwrap() as f64)
            .collect();
        years[3] = 2002.5;
        let ds = with_column(&ds, Column::new("year".into(), years));
        let e = run_arellano_bond(&ds, "y", &["x"], "id", "year").unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("whole numbers")),
            "{e:?}"
        );
    }

    /// x in millions: the same estimates, scaled. The singularity decision is made on the
    /// equilibrated weighting matrix, so the x direction is not dropped.
    #[test]
    fn test_gmm_scale_equivariance() {
        let ds = repo_dataset(AB_BALANCED);
        let x: Vec<f64> = ds
            .df()
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| 1e6 * v.unwrap())
            .collect();
        let scaled = with_column(&ds, Column::new("x".into(), x));
        let tol = Tol::new(1e-9, 0.0);
        let a = run_arellano_bond(&ds, "y", &["x"], "id", "year").unwrap();
        let b = run_arellano_bond(&scaled, "y", &["x"], "id", "year").unwrap();
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
        assert_eq!(b.two_step_weight_rank, 22);
        let pairs = [
            (b.coefficients[0], a.coefficients[0]),
            (1e6 * b.coefficients[1], a.coefficients[1]),
            (b.std_errors[0], a.std_errors[0]),
            (1e6 * b.std_errors[1], a.std_errors[1]),
        ];
        for (got, want) in pairs {
            assert!(tol.allows(got, want), "{got} vs {want}");
        }
    }

    // ── System GMM: pgmm(transformation = "ld") (validation/r/cases/econ_sysgmm.R) ─────

    const SYS_FIT_LAGS1: &[&str] = &["L1.y", "x"];

    #[allow(clippy::too_many_arguments)]
    fn sys_fit(
        ds: &Dataset,
        y: &str,
        x: &[&str],
        e: &str,
        t: &str,
        lags: usize,
        step: GmmStep,
        robust: bool,
        min_lag: usize,
        max_lag: Option<usize>,
        collapse: bool,
    ) -> GmmResult {
        let mut cfg = GmmConfig::default();
        cfg.transform = GmmTransform::System;
        cfg.step = step;
        cfg.robust = robust;
        cfg.min_lag = min_lag;
        cfg.max_lag = max_lag;
        cfg.collapse = collapse;
        run_gmm(ds, y, x, e, t, lags, Some(cfg)).unwrap_or_else(|err| panic!("{err}"))
    }

    /// A system fit on `ab_balanced`/`ab_gaps` (y on x, entity id, time year).
    #[allow(clippy::too_many_arguments)]
    fn sys_ab(
        ds: &Dataset,
        x: &[&str],
        lags: usize,
        step: GmmStep,
        robust: bool,
        min_lag: usize,
        max_lag: Option<usize>,
        collapse: bool,
    ) -> GmmResult {
        sys_fit(
            ds, "y", x, "id", "year", lags, step, robust, min_lag, max_lag, collapse,
        )
    }

    /// The properties every pinned system fit has.
    fn assert_sys(r: &GmmResult, variables: &[&str]) {
        assert_eq!(r.transform, GmmTransform::System);
        assert_eq!(r.variables, variables);
        assert_eq!(
            (r.one_step_weight_rank, r.two_step_weight_rank),
            (r.n_instruments, r.n_instruments)
        );
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        assert_eq!(r.sargan_homoskedastic, None);
        assert_eq!(r.sargan_homoskedastic_p_value, None);
        assert_eq!(r.n_obs, r.n_obs_diff + r.n_obs_level);
    }

    /// `{p}_*` of a robust fit (one-step vcovHC or two-step Windmeijer) for `n` equations.
    fn check_sys(c: &mut Check<'_>, p: &str, names: &[&str], r: &GmmResult, n: usize) {
        let t = Tols::n(n);
        let exact = Tol::exact();
        c.named(&format!("{p}_coef"), names, &r.coefficients, t.coef)
            .named(&format!("{p}_se_robust"), names, &r.std_errors, t.se)
            .scalar(&format!("{p}_sargan"), r.sargan_statistic, t.stat)
            .scalar(&format!("{p}_sargan_p"), r.sargan_p_value, t.p)
            .scalar(&format!("{p}_ar1"), r.ar1_statistic, t.stat)
            .scalar(&format!("{p}_ar1_p"), r.ar1_p_value, t.p)
            .scalar(&format!("{p}_ar2"), r.ar2_statistic, t.stat)
            .scalar(&format!("{p}_ar2_p"), r.ar2_p_value, t.p)
            .scalar(&format!("{p}_sargan_df"), r.sargan_df as f64, exact)
            .scalar(&format!("{p}_n_instruments"), r.n_instruments as f64, exact)
            .scalar(&format!("{p}_n_obs"), r.n_obs as f64, exact)
            .scalar(&format!("{p}_n_obs_diff"), r.n_obs_diff as f64, exact)
            .scalar(&format!("{p}_n_obs_level"), r.n_obs_level as f64, exact)
            .scalar(&format!("{p}_n_groups"), r.n_groups as f64, exact);
        assert!(
            r.se_type.contains("robust") || r.se_type.contains("Windmeijer"),
            "{}",
            r.se_type
        );
    }

    /// A two-step fit's uncorrected SEs (`{p}_se`); its coefficients equal the Windmeijer
    /// fit's bit for bit.
    fn check_sys_uncorrected(
        c: &mut Check<'_>,
        p: &str,
        names: &[&str],
        uncorrected: &GmmResult,
        windmeijer: &GmmResult,
        n: usize,
    ) {
        c.named(
            &format!("{p}_se"),
            names,
            &uncorrected.std_errors,
            Tols::n(n).se,
        );
        assert_eq!(uncorrected.coefficients, windmeijer.coefficients);
        assert_eq!(
            uncorrected.se_type,
            "two-step, uncorrected (biased downward in finite samples)"
        );
    }

    #[test]
    fn test_validate_sysgmm_balanced_onestep() {
        let (case, ds) = crate::test_support::load_case("dpd_sys_balanced");
        let r1 = sys_ab(&ds, &["x"], 1, GmmStep::OneStep, true, 2, None, false);
        let mut c = Check::new(&case);
        check_sys(&mut c, "onestep", &["lag(y, 1)", "x"], &r1, 1300);
        c.scalar("onestep_sigma2", r1.sigma2, Tol::stat(1300));
        c.finish();
        assert_sys(&r1, SYS_FIT_LAGS1);
        assert_eq!(
            r1.se_type,
            "one-step, robust (plm vcovHC: heteroskedasticity and within-entity correlation)"
        );
        assert_eq!(r1.one_step_coefficients, r1.coefficients);
    }

    #[test]
    fn test_validate_sysgmm_balanced_twostep() {
        let (case, ds) = crate::test_support::load_case("dpd_sys_balanced");
        let rb = sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, true, 2, None, false);
        let cl = sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, false, 2, None, false);
        let names = ["lag(y, 1)", "x"];
        let mut c = Check::new(&case);
        check_sys(&mut c, "twostep", &names, &rb, 1300);
        check_sys_uncorrected(&mut c, "twostep", &names, &cl, &rb, 1300);
        c.finish();
        for r in [&rb, &cl] {
            assert_sys(r, SYS_FIT_LAGS1);
        }
        assert_eq!(rb.se_type, "two-step, Windmeijer (2005) corrected");
    }

    #[test]
    fn test_validate_sysgmm_balanced_variants() {
        let (case, ds) = crate::test_support::load_case("dpd_sys_balanced");
        let mut c = Check::new(&case);
        let two = |lags, max_lag, collapse| {
            [true, false].map(|robust| {
                sys_ab(
                    &ds,
                    &["x"],
                    lags,
                    GmmStep::TwoStep,
                    robust,
                    2,
                    max_lag,
                    collapse,
                )
            })
        };

        let [rb, cl] = two(2, None, false);
        let names = ["lag(y, 1:2)1", "lag(y, 1:2)2", "x"];
        check_sys(&mut c, "lags2_twostep", &names, &rb, 1100);
        check_sys_uncorrected(&mut c, "lags2_twostep", &names, &cl, &rb, 1100);
        assert_sys(&rb, &["L1.y", "L2.y", "x"]);

        let [rb, cl] = two(1, Some(3), false);
        let names = ["lag(y, 1)", "x"];
        check_sys(&mut c, "maxlag3_twostep", &names, &rb, 1300);
        check_sys_uncorrected(&mut c, "maxlag3_twostep", &names, &cl, &rb, 1300);
        assert_sys(&rb, SYS_FIT_LAGS1);

        let [rb, cl] = two(1, None, true);
        check_sys(&mut c, "collapse_twostep", &names, &rb, 1300);
        check_sys_uncorrected(&mut c, "collapse_twostep", &names, &cl, &rb, 1300);
        assert_sys(&rb, SYS_FIT_LAGS1);
        assert!(
            rb.to_string().contains("D.y lag 1, collapsed: one column"),
            "{rb}"
        );

        let r = sys_ab(&ds, &["x"], 1, GmmStep::OneStep, true, 3, None, false);
        check_sys(&mut c, "minlag3_onestep", &names, &r, 1100);
        c.scalar("minlag3_onestep_sigma2", r.sigma2, Tol::stat(1100));
        assert_sys(&r, SYS_FIT_LAGS1);

        let r = sys_ab(&ds, &[], 1, GmmStep::OneStep, true, 2, None, false);
        check_sys(&mut c, "nox_onestep", &["lag(y, 1)"], &r, 1300);
        c.scalar("nox_onestep_sigma2", r.sigma2, Tol::stat(1300));
        assert_sys(&r, &["L1.y"]);
        let text = r.to_string();
        assert!(!text.contains("D.x") && !text.contains("; x)"), "{text}");
        c.finish();
    }

    /// pgmm 2.6.7 mis-builds its level instruments for TL2 >= 3; reference = the estimator
    /// written out in R (econ_sysgmm.R .sys_spec), which reproduces pgmm on every other
    /// variant.
    #[test]
    fn test_validate_sysgmm_spec_lags3_minlag4() {
        let (case, ds) = crate::test_support::load_case("dpd_sys_balanced");
        let mut c = Check::new(&case);
        let t = Tols::n(900);
        let exact = Tol::exact();
        for (p, lags, min_lag, variables) in [
            ("lags3_spec", 3, 2, ["L1.y", "L2.y", "L3.y", "x"].as_slice()),
            ("minlag4_spec", 1, 4, ["L1.y", "x"].as_slice()),
        ] {
            let one = sys_ab(
                &ds,
                &["x"],
                lags,
                GmmStep::OneStep,
                true,
                min_lag,
                None,
                false,
            );
            let two = sys_ab(
                &ds,
                &["x"],
                lags,
                GmmStep::TwoStep,
                true,
                min_lag,
                None,
                false,
            );
            c.vector(&format!("{p}_onestep_coef"), &one.coefficients, t.coef)
                .vector(&format!("{p}_twostep_coef"), &two.coefficients, t.coef)
                .vector(&format!("{p}_twostep_se_robust"), &two.std_errors, t.se)
                .scalar(&format!("{p}_twostep_sargan"), two.sargan_statistic, t.stat)
                .scalar(&format!("{p}_n_obs"), two.n_obs as f64, exact)
                .scalar(&format!("{p}_n_obs_diff"), two.n_obs_diff as f64, exact)
                .scalar(&format!("{p}_n_obs_level"), two.n_obs_level as f64, exact)
                .scalar(
                    &format!("{p}_n_instruments"),
                    two.n_instruments as f64,
                    exact,
                );
            for r in [&one, &two] {
                assert_sys(r, variables);
            }
        }
        c.finish();
    }

    #[test]
    fn test_validate_sysgmm_gaps() {
        let (case, ds) = crate::test_support::load_case("dpd_sys_gaps");
        let mut c = Check::new(&case);
        let names = ["lag(y, 1)", "x"];
        let r1 = sys_ab(&ds, &["x"], 1, GmmStep::OneStep, true, 2, None, false);
        check_sys(&mut c, "onestep", &names, &r1, 1217);
        c.scalar("onestep_sigma2", r1.sigma2, Tol::stat(1217));
        let rb = sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, true, 2, None, false);
        let cl = sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, false, 2, None, false);
        check_sys(&mut c, "twostep", &names, &rb, 1217);
        check_sys_uncorrected(&mut c, "twostep", &names, &cl, &rb, 1217);
        for r in [&r1, &rb, &cl] {
            assert_sys(r, SYS_FIT_LAGS1);
        }
        assert_eq!(
            (rb.n_obs, rb.n_obs_diff, rb.n_obs_level, rb.n_groups),
            (1217, 553, 664, 100)
        );

        let lags2 = [true, false]
            .map(|robust| sys_ab(&ds, &["x"], 2, GmmStep::TwoStep, robust, 2, None, false));
        let l2 = ["lag(y, 1:2)1", "lag(y, 1:2)2", "x"];
        check_sys(&mut c, "lags2_twostep", &l2, &lags2[0], 995);
        check_sys_uncorrected(&mut c, "lags2_twostep", &l2, &lags2[1], &lags2[0], 995);
        assert_sys(&lags2[0], &["L1.y", "L2.y", "x"]);
        let col = [true, false]
            .map(|robust| sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, robust, 2, None, true));
        check_sys(&mut c, "collapse_twostep", &names, &col[0], 1217);
        check_sys_uncorrected(&mut c, "collapse_twostep", &names, &col[1], &col[0], 1217);
        assert_sys(&col[0], SYS_FIT_LAGS1);
        c.finish();
    }

    #[test]
    fn test_validate_sysgmm_empluk() {
        let (case, ds) = crate::test_support::load_case("dpd_sys_empluk");
        let fit = |step, robust| {
            sys_fit(
                &ds,
                "lemp",
                &["lwage", "lcap"],
                "firm",
                "year",
                1,
                step,
                robust,
                2,
                None,
                false,
            )
        };
        let names = ["lag(lemp, 1)", "lwage", "lcap"];
        let variables = ["L1.lemp", "lwage", "lcap"];
        let mut c = Check::new(&case);
        let r1 = fit(GmmStep::OneStep, true);
        check_sys(&mut c, "onestep", &names, &r1, 1642);
        c.scalar("onestep_sigma2", r1.sigma2, Tol::stat(1642));
        let rb = fit(GmmStep::TwoStep, true);
        let cl = fit(GmmStep::TwoStep, false);
        check_sys(&mut c, "twostep", &names, &rb, 1642);
        check_sys_uncorrected(&mut c, "twostep", &names, &cl, &rb, 1642);
        c.finish();
        for r in [&r1, &rb, &cl] {
            assert_sys(r, &variables);
            assert_eq!((r.n_groups, r.n_instruments), (140, 39));
        }
    }

    #[test]
    fn test_sysgmm_differs_from_difference() {
        let ds = repo_dataset(AB_BALANCED);
        let s = sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, true, 2, None, false);
        let d = fit(
            &ds,
            "y",
            &["x"],
            "id",
            "year",
            1,
            GmmStep::TwoStep,
            true,
            2,
            None,
            false,
        );
        assert_eq!(s.transform, GmmTransform::System);
        assert!(
            !s.warnings.iter().any(|w| w.contains("ECON-30")),
            "{:?}",
            s.warnings
        );
        assert!(
            (s.coefficients[0] - d.coefficients[0]).abs() > 0.3,
            "system {} vs difference {}",
            s.coefficients[0],
            d.coefficients[0]
        );
        assert_eq!((s.n_obs, d.n_obs), (1300, 600));
        assert_eq!((s.n_obs_diff, s.n_obs_level), (600, 700));
        assert_eq!((d.n_obs_diff, d.n_obs_level), (600, 0));
        assert_eq!((s.n_instruments, d.n_instruments), (29, 22));
        let text = s.to_string();
        assert!(
            text.starts_with("System GMM (Blundell-Bond) (Two-step)"),
            "{text}"
        );
        for needle in [
            "Dep. Variable: y (differenced and level equations)",
            "No. Observations: 1300 (600 differenced, 700 level equations)",
            "No. Instruments: 29 (differenced equations: y lags 2..all, one column per \
             period and lag; D.x | level equations: D.y lag 1, one column per period; x)",
        ] {
            assert!(text.contains(needle), "{needle:?} not in:\n{text}");
        }
        assert!(
            d.to_string()
                .starts_with("Difference GMM (Arellano-Bond) (Two-step)")
        );
    }

    #[test]
    fn test_sysgmm_onestep_classical_se_is_refused() {
        let ds = repo_dataset(AB_BALANCED);
        let mut cfg = GmmConfig::default();
        cfg.transform = GmmTransform::System;
        cfg.step = GmmStep::OneStep;
        cfg.robust = false;
        match run_gmm(&ds, "y", &["x"], "id", "year", 1, Some(cfg.clone())) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("System GMM has no classical one-step standard errors")
                    && message.contains("robust = true"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
        cfg.step = GmmStep::TwoStep;
        assert!(run_gmm(&ds, "y", &["x"], "id", "year", 1, Some(cfg)).is_ok());
    }

    /// x in millions: the same system estimates, scaled.
    #[test]
    fn test_sysgmm_scale_equivariance() {
        let ds = repo_dataset(AB_BALANCED);
        let x: Vec<f64> = ds
            .df()
            .column("x")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| 1e6 * v.unwrap())
            .collect();
        let scaled = with_column(&ds, Column::new("x".into(), x));
        let a = sys_ab(&ds, &["x"], 1, GmmStep::TwoStep, true, 2, None, false);
        let b = sys_ab(&scaled, &["x"], 1, GmmStep::TwoStep, true, 2, None, false);
        assert!(b.warnings.is_empty(), "{:?}", b.warnings);
        assert_eq!((b.one_step_weight_rank, b.two_step_weight_rank), (29, 29));
        let tol = Tol::new(1e-9, 0.0);
        let pairs = [
            (b.coefficients[0], a.coefficients[0]),
            (1e6 * b.coefficients[1], a.coefficients[1]),
            (b.std_errors[0], a.std_errors[0]),
            (1e6 * b.std_errors[1], a.std_errors[1]),
        ];
        for (got, want) in pairs {
            assert!(tol.allows(got, want), "{got} vs {want}");
        }
    }

    /// X-7, panel/utils.rs:202 (d6a3a3b): guarded since T0-WP14, site removed by T2-WP19;
    /// regression guard for the System path.
    #[test]
    fn test_sysgmm_rejects_max_lag_below_min_lag() {
        let ds = repo_dataset(AB_BALANCED);
        for (min_lag, max_lag) in [(3, Some(2)), (2, Some(1)), (2, Some(0))] {
            let mut cfg = GmmConfig::default();
            cfg.transform = GmmTransform::System;
            cfg.min_lag = min_lag;
            cfg.max_lag = max_lag;
            let e = run_gmm(&ds, "y", &["x"], "id", "year", 1, Some(cfg)).unwrap_err();
            assert!(
                matches!(&e, EconError::InvalidSpecification { message } if message.contains("max_lag")),
                "{min_lag} {max_lag:?}: {e:?}"
            );
        }
    }

    #[test]
    fn test_sysgmm_scaling_many_entities() {
        if cfg!(debug_assertions) {
            return; // release-only timing check
        }
        let ds = synthetic_panel(50_000, 5);
        let mut cfg = GmmConfig::default();
        cfg.transform = GmmTransform::System;
        let t0 = std::time::Instant::now();
        let r = run_gmm(&ds, "y", &["x"], "entity", "time", 1, Some(cfg)).unwrap();
        assert_eq!(r.n_groups, 50_000);
        assert_eq!(r.n_obs, 50_000 * 7);
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(10),
            "took {:?}",
            t0.elapsed()
        );
    }

    /// One-step fits do not need the two-step weight: its rank warning names it as the
    /// Sargan/Hansen weighting matrix without the two-step advice.
    #[test]
    fn test_gmm_onestep_rank_warning_names_the_sargan_weight() {
        let ds = grunfeld();
        let x = ["value", "capital"];
        let run = |transform, step, robust| {
            let mut cfg = GmmConfig::default();
            cfg.transform = transform;
            cfg.step = step;
            cfg.robust = robust;
            run_gmm(&ds, "inv", &x, "firm", "year", 1, Some(cfg)).unwrap()
        };
        for (transform, robust) in [
            (GmmTransform::Difference, false),
            (GmmTransform::Difference, true),
            (GmmTransform::System, true),
        ] {
            let r = run(transform, GmmStep::OneStep, robust);
            assert!(
                !r.warnings.iter().any(|w| w.contains("Two-step GMM needs")),
                "{:?}",
                r.warnings
            );
            let sargan = r
                .warnings
                .iter()
                .find(|w| w.contains("Sargan/Hansen weighting matrix"))
                .unwrap_or_else(|| panic!("{:?}", r.warnings));
            assert!(sargan.contains("a generalized inverse is used"), "{sargan}");
            if transform == GmmTransform::Difference {
                assert!(sargan.contains("rank 9 of 173"), "{sargan}");
            }
        }
        let r = run(GmmTransform::Difference, GmmStep::TwoStep, true);
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("Two-step GMM needs at least as many entities as instruments")),
            "{:?}",
            r.warnings
        );
    }

    #[test]
    fn test_na_gmm_int64_equals_float64() {
        let ds = repo_dataset(AB_BALANCED);
        let y: Vec<f64> = ds
            .df()
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .iter()
            .map(|v| (v.unwrap() * 1e4).round())
            .collect();
        let yi: Vec<i64> = y.iter().map(|&v| v as i64).collect();
        let ds = with_column(&ds, Column::new("yf".into(), y));
        let ds = with_column(&ds, Column::new("yi".into(), yi));
        let a = run_arellano_bond(&ds, "yi", &["x"], "id", "year").unwrap();
        let b = run_arellano_bond(&ds, "yf", &["x"], "id", "year").unwrap();
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.std_errors, b.std_errors);
        assert_eq!(a.sargan_statistic.to_bits(), b.sargan_statistic.to_bits());
    }

    /// Before: every extraction error of y was ColumnNotFound.
    #[test]
    fn test_na_gmm_text_outcome_is_non_numeric() {
        let ds = repo_dataset(AB_COMMON_GAP);
        let e = run_arellano_bond(&ds, "unit", &["x"], "id", "year").unwrap_err();
        assert!(
            matches!(&e, EconError::NonNumericColumn { column } if column == "unit"),
            "{e:?}"
        );
    }

    #[test]
    fn test_na_gmm_absent_column() {
        let ds = repo_dataset(AB_BALANCED);
        for (y, x) in [("nope", "x"), ("y", "nope")] {
            let e = run_arellano_bond(&ds, y, &[x], "id", "year").unwrap_err();
            assert!(
                matches!(&e, EconError::ColumnNotFound { column, .. } if column == "nope"),
                "{e:?}"
            );
        }
    }

    /// Rows with a missing entity id or time are dropped (before: NullValues), and the
    /// fit equals the fit on the frame without them.
    #[test]
    fn test_na_gmm_missing_ids_and_times_are_dropped() {
        let ds = repo_dataset(AB_BALANCED);
        let df = ds.df();
        let ids: Vec<Option<i64>> = df.column("id").unwrap().i64().unwrap().iter().collect();
        let years: Vec<Option<i64>> = df.column("year").unwrap().i64().unwrap().iter().collect();
        // Rows 10 (id 2), 35 (id 5) and 60 (id 8).
        let (mut id_na, mut year_na) = (ids.clone(), years.clone());
        id_na[10] = None;
        id_na[35] = None;
        year_na[60] = None;
        let na = with_column(&ds, Column::new("id".into(), id_na));
        let na = with_column(&na, Column::new("year".into(), year_na));
        let keep: Vec<bool> = (0..df.height())
            .map(|i| ![10, 35, 60].contains(&i))
            .collect();
        let mask = keep
            .iter()
            .copied()
            .collect::<polars::prelude::BooleanChunked>();
        let complete = Dataset::new(df.filter(&mask).unwrap());
        let a = run_arellano_bond(&na, "y", &["x"], "id", "year").unwrap();
        let b = run_arellano_bond(&complete, "y", &["x"], "id", "year").unwrap();
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.std_errors, b.std_errors);
        assert_eq!(a.n_obs, b.n_obs);
        assert_eq!(
            a.missing.by_column,
            [("id".to_string(), 2), ("year".to_string(), 1)]
        );
        assert_eq!((a.missing.n_dropped, a.missing.n_total), (3, 800));
    }
}
