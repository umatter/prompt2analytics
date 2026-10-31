//! Arellano-Bond difference GMM for dynamic panel data, as R's
//! `plm::pgmm(transformation = "d")`.
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
//! - **System GMM** (Blundell-Bond) is not implemented yet (ECON-30): `GmmTransform::System`
//!   runs difference GMM, reports `GmmTransform::Difference` and warns.
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
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{cholesky_inverse, eig_symmetric};
use crate::resource;
use crate::traits::estimator::SignificanceLevel;

use super::utils::{DiffEquations, GmmLagSpec, difference_equations};

/// The inverse of an L x L weighting matrix holds several L x L copies at once (the
/// matrix, its scaled copy, eigenvectors, the factor and the solution). Check the
/// working set against the allocation ceiling before building it.
const DENSE_INVERSE_COPIES: usize = 6;

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
    /// System GMM (Blundell-Bond 1998). Not implemented yet (ECON-30): `run_gmm` runs
    /// difference GMM with a warning.
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

/// Configuration for GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmConfig {
    /// Transformation. `System` is not implemented yet (ECON-30): it runs Difference
    /// with a warning.
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
    /// standard errors. Two-step: Windmeijer (2005)-corrected (true) or uncorrected.
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

/// Result from Arellano-Bond difference GMM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmmResult {
    /// Estimator run (always `Difference`; see `GmmConfig::transform`)
    pub transform: GmmTransform,
    /// Estimation step
    pub step: GmmStep,
    /// Dependent variable name ("D.y")
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
    /// Number of differenced equations used
    pub n_obs: usize,
    /// Number of entities with at least one equation
    pub n_groups: usize,
    /// Number of instrument columns used (columns no equation observes are not created)
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
    /// σ² = Σ (one-step differenced residual)² / (2 (n_obs - k)) (Arellano-Bond 1991).
    #[serde(default)]
    pub sigma2: f64,
    /// Arellano-Bond (1991) Sargan statistic of a one-step fit under homoskedastic errors:
    /// (Z'e)' (Σ_i Z_i' H Z_i)^-1 (Z'e) / σ², df = `sargan_df`. `None` for two-step fits.
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

    // ── Trait-backing storage (LinearEstimator). Skipped from JSON to keep the
    //    public surface stable.
    /// Coefficients as `Array1`, used by `LinearEstimator::coefficients`.
    #[serde(skip, default)]
    pub coef_arr: ndarray::Array1<f64>,
    /// Standard errors as `Array1`, used by `LinearEstimator::std_errors`.
    #[serde(skip, default)]
    pub se_arr: ndarray::Array1<f64>,
    /// Differenced residuals of the reported step.
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
    /// "y lags 2..all, one column per period and lag; D.x": what the instruments are.
    fn instrument_description(&self) -> String {
        let y = self.dep_var.strip_prefix("D.").unwrap_or(&self.dep_var);
        let max = self
            .max_lag
            .map_or_else(|| "all".to_string(), |m| m.to_string());
        let layout = if self.collapse {
            "collapsed: one column per lag depth"
        } else {
            "one column per period and lag"
        };
        let mut text = format!("{y} lags {}..{max}, {layout}", self.min_lag);
        let dx: Vec<String> = self
            .variables
            .iter()
            .skip(self.lags)
            .map(|v| format!("D.{v}"))
            .collect();
        if !dx.is_empty() {
            text.push_str("; ");
            text.push_str(&dx.join(", "));
        }
        text
    }
}

impl fmt::Display for GmmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} ({}) Panel GMM Results", self.transform, self.step)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "Time: {}", self.time_var)?;
        writeln!(
            f,
            "No. Observations: {} (differenced equations)",
            self.n_obs
        )?;
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

        // The missing-value note is the last line, not a warning line.
        let note = self.missing.note();
        let warnings: Vec<&String> = self
            .warnings
            .iter()
            .filter(|w| Some(w.as_str()) != note.as_deref())
            .collect();
        if !warnings.is_empty() {
            writeln!(f)?;
            writeln!(f, "Warnings:")?;
            for w in warnings {
                writeln!(f, "  - {}", w)?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;
        if let Some(note) = note {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Inverse of a symmetric weighting matrix as `plm::pgmm` inverts it.
struct WeightInverse {
    /// The inverse, or the generalized inverse when `rank` < L.
    inv: Array2<f64>,
    /// Numerical rank.
    rank: usize,
    /// The matrix restricted to the kept eigen-directions (`MASS::ginv(MASS::ginv(S))`);
    /// the matrix itself at full rank.
    kept: Array2<f64>,
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
fn weight_inverse(s: &Array2<f64>) -> EconResult<WeightInverse> {
    let s = symmetrized(s);
    let l = s.nrows();
    if let Some(inv) = equilibrated_inverse(&s) {
        return Ok(WeightInverse {
            inv,
            rank: l,
            kept: s,
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
    let mut vk_kept = resource::try_zeros2(l, keep.len(), "GMM generalized inverse")?;
    for (c, &j) in keep.iter().enumerate() {
        vk.column_mut(c).assign(&vectors.column(j));
        vk_inv
            .column_mut(c)
            .assign(&(&vectors.column(j) / values[j]));
        vk_kept
            .column_mut(c)
            .assign(&(&vectors.column(j) * values[j]));
    }
    let inv = symmetrized(&vk_inv.dot(&vk.t()));
    let kept = symmetrized(&vk_kept.dot(&vk.t()));
    Ok(WeightInverse {
        inv,
        rank: keep.len(),
        kept,
    })
}

/// k x k inverse of a bread matrix X'Z A Z'X (equilibrated Cholesky); singular is
/// `EconError::SingularMatrix`.
fn small_inverse(m: &Array2<f64>, what: &str) -> EconResult<Array2<f64>> {
    equilibrated_inverse(&symmetrized(m)).ok_or_else(|| EconError::SingularMatrix {
        context: what.to_string(),
        suggestion: "The regressors are collinear given the instruments (for example a \
                     time-invariant x, whose difference is zero); drop or recode them."
            .to_string(),
    })
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
    ChiSquared::new(df as f64).map_or(f64::NAN, |c| c.sf(stat.max(0.0)))
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

/// Σ_i Z_i' H_i Z_i, H_i[a, b] = 2 if a = b, -1 if the equations are one period apart.
fn one_step_moment_matrix(eq: &DiffEquations, sp: &SparseRows) -> EconResult<Array2<f64>> {
    let l = eq.z.ncols();
    let mut s1 = resource::try_zeros2(l, l, "GMM one-step weighting matrix")?;
    for block in &eq.blocks {
        resource::checkpoint()?;
        for a in block.clone() {
            let (ca, va) = sp.row(a);
            for (&c, &v) in ca.iter().zip(va) {
                for (&d, &w) in ca.iter().zip(va) {
                    s1[[c, d]] += 2.0 * v * w;
                }
            }
            let b = a + 1;
            if b < block.end && eq.time[b] == eq.time[a] + 1 {
                let (cb, vb) = sp.row(b);
                for (&c, &v) in ca.iter().zip(va) {
                    for (&d, &w) in cb.iter().zip(vb) {
                        s1[[c, d]] -= v * w;
                        s1[[d, c]] -= v * w;
                    }
                }
            }
        }
    }
    Ok(s1)
}

/// Σ_i g_i g_i', g_i = Z_i' e_i.
fn two_step_moment_matrix(
    eq: &DiffEquations,
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
    eq: &DiffEquations,
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
fn has_lag_pair(eq: &DiffEquations, order: i64) -> bool {
    eq.blocks.iter().any(|block| {
        let times = &eq.time[block.clone()];
        times
            .iter()
            .enumerate()
            .any(|(p, &t)| times[..p].binary_search(&(t - order)).is_ok())
    })
}

/// The Arellano-Bond m-test of order `order`, as `plm::mtest`: (statistic, p-value), NaN
/// when the variance estimate is not positive (no entity has two equations `order`
/// periods apart, or the estimate is negative). `b` is the bread (X'Z W Z'X)^-1, `w` the
/// weighting matrix and `v_robust` the robust coefficient covariance of the fit whose
/// residuals are `e`.
fn ab_mtest(
    e: &[f64],
    eq: &DiffEquations,
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
        let times = &eq.time[block.clone()];
        let mut s = 0.0;
        for (p, a) in block.clone().enumerate() {
            if let Ok(q) = times[..p].binary_search(&(times[p] - order)) {
                let lagged = e[block.start + q];
                s += e[a] * lagged;
                ex.scaled_add(lagged, &eq.x.row(a));
            }
        }
        num += s;
        eve += s * s;
        if s != 0.0 {
            for a in block.clone() {
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

/// Arellano-Bond difference GMM, as `plm::pgmm(transformation = "d")` (see the module
/// docs for the conventions).
///
/// # Arguments
///
/// * `dataset` - The dataset containing the panel data
/// * `y_col` - Name of the dependent variable column
/// * `x_cols` - Names of the strictly exogenous regressors (not the lagged y: `lags`
///   adds it); they enter in differences and serve as their own instruments
/// * `entity_col` - Column name for entity/individual identifier (numbers or text)
/// * `time_col` - Column name for the time period: whole numbers or dates, lags are
///   steps of 1
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

    if lags == 0 && x_cols.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "GMM has no regressors: set lags to at least 1 or name x columns".to_string(),
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
    };
    let eq = difference_equations(&cols, y_col, x_cols, entity_col, time_col, &spec)?;
    let (n_obs, k) = eq.x.dim();
    let l = eq.z.ncols();

    if config.transform == GmmTransform::System {
        warnings.push(
            "System GMM (Blundell-Bond) is not implemented yet (ECON-30); these are \
             difference-GMM (Arellano-Bond) estimates."
                .to_string(),
        );
    }
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
            lags + 2,
            label_list(&eq.entities_without_equations)
        ));
    }

    ensure_dense_inverse_fits(l, "GMM weighting matrix")?;
    let sp = SparseRows::of(&eq.z)?;
    let mut acc = BlockSum::new(l)?;
    let dy = Array1::from_vec(eq.dy.clone());
    let zx = eq.z.t().dot(&eq.x);
    let zy = eq.z.t().dot(&dy);

    // One step.
    let w1 = weight_inverse(&one_step_moment_matrix(&eq, &sp)?)?;
    let w1zx = w1.inv.dot(&zx);
    let b1 = small_inverse(&zx.t().dot(&w1zx), "X'Z A Z'X in one-step GMM")?;
    let beta1 = b1.dot(&w1zx.t().dot(&zy));
    let e1 = &dy - &eq.x.dot(&beta1);

    // Two-step weights from the one-step residuals: the weights of the Sargan/Hansen test
    // of either step, and of the two-step estimates.
    let w2 = weight_inverse(&two_step_moment_matrix(&eq, &sp, &e1, &mut acc)?)?;
    let two_step = match config.step {
        GmmStep::OneStep => None,
        GmmStep::TwoStep => {
            let w2zx = w2.inv.dot(&zx);
            let b2 = small_inverse(&zx.t().dot(&w2zx), "X'Z A Z'X in two-step GMM")?;
            let beta2 = b2.dot(&w2zx.t().dot(&zy));
            let e2 = &dy - &eq.x.dot(&beta2);
            Some((w2zx, b2, beta2, e2))
        }
    };

    for (name, sum, w) in [
        ("one-step", "Z_i' H Z_i", &w1),
        ("two-step", "Z_i' e_i e_i' Z_i", &w2),
    ] {
        if w.rank < l {
            warnings.push(format!(
                "The {name} weighting matrix (sum of {sum}) has numerical rank {} of {l} \
                 instruments; a generalized inverse is used, as plm::pgmm does.{}",
                w.rank,
                if name == "two-step" {
                    " Two-step GMM needs at least as many entities as instruments: reduce \
                     them with max_lag or collapse."
                } else {
                    " Reduce the instruments with max_lag or collapse."
                }
            ));
        }
    }

    // Covariances.
    let sigma2 = e1.dot(&e1) / (2.0 * (n_obs - k) as f64);
    let v1r = b1.dot(&w1zx.t().dot(&w2.kept).dot(&w1zx)).dot(&b1);
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
        GmmStep::OneStep => {
            let z1 = eq.z.t().dot(&e1);
            let stat = z1.dot(&w1.inv.dot(&z1)) / sigma2;
            (Some(stat), Some(chi_squared_upper(stat, sargan_df)))
        }
        GmmStep::TwoStep => (None, None),
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
        transform: GmmTransform::Difference,
        step: config.step,
        dep_var: format!("D.{}", y_col),
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

    #[test]
    fn test_gmm_system_transform_reports_difference() {
        let ds = repo_dataset(AB_BALANCED);
        let diff = run_arellano_bond(&ds, "y", &["x"], "id", "year").unwrap();
        let cfg = GmmConfig {
            transform: GmmTransform::System,
            ..Default::default()
        };
        let sys = run_gmm(&ds, "y", &["x"], "id", "year", 1, Some(cfg)).unwrap();
        assert_eq!(sys.coefficients, diff.coefficients);
        assert_eq!(sys.std_errors, diff.std_errors);
        assert_eq!(sys.transform, GmmTransform::Difference);
        assert!(
            sys.warnings.iter().any(|w| w.contains("ECON-30")),
            "{:?}",
            sys.warnings
        );
        assert!(
            sys.to_string()
                .starts_with("Difference GMM (Arellano-Bond)")
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
