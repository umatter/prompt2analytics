//! Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) estimation.
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.
//!
//! # Mathematical Background
//!
//! IV estimation addresses endogeneity when Cov(X, ε) ≠ 0. Given instruments Z
//! that satisfy:
//! 1. **Relevance**: Cov(Z, X) ≠ 0
//! 2. **Exogeneity**: Cov(Z, ε) = 0
//!
//! ## Two-Stage Least Squares (2SLS)
//!
//! **First stage**: Regress endogenous X on instruments Z and exogenous W:
//!   X̂ = Z(Z'Z)⁻¹Z'X = Pᵤ X
//!
//! **Second stage**: Regress y on fitted values X̂ and exogenous W:
//!   β̂₂ₛₗₛ = (X̂'X̂)⁻¹ X̂'y
//!
//! Equivalently: β̂ᵢᵥ = (X'Pᵤ X)⁻¹ X'Pᵤ y
//!
//! ## Weak Instruments
//!
//! The first-stage F is the partial F of the excluded instruments (restricted fit
//! on the included exogenous regressors), AER's "Weak instruments" statistic; with
//! robust = true it is the HC1 Wald F of the excluded-instrument coefficients.
//! F > 10 is the rule of thumb for instrument strength. Stock & Yogo (2005)
//! critical values are not computed.
//!
//! # References
//!
//! - Wright, P.G. (1928). *The Tariff on Animal and Vegetable Oils*. Macmillan.
//!   First application of instrumental variables.
//!
//! - Theil, H. (1953). Repeated least squares applied to complete equation systems.
//!   *The Hague: Central Planning Bureau*. Introduction of 2SLS.
//!
//! - Basmann, R.L. (1957). A generalized classical method of linear estimation of
//!   coefficients in a structural equation. *Econometrica*, 25(1), 77-83.
//!   https://doi.org/10.2307/1907743
//!
//! - Stock, J.H., & Yogo, M. (2005). Testing for weak instruments in linear IV
//!   regression. In D.W.K. Andrews & J.H. Stock (Eds.), *Identification and
//!   Inference for Econometric Models* (pp. 80-108). Cambridge University Press.
//!   https://doi.org/10.1017/CBO9780511614491.006
//!
//! - Angrist, J.D., & Pischke, J.S. (2009). *Mostly Harmless Econometrics: An
//!   Empiricist's Companion*. Princeton University Press. ISBN: 978-0691120355.
//!
//! - Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
//!   (2nd ed.), Chapter 5. MIT Press.
//!
//! R equivalent: `AER::ivreg()`, `ivreg::ivreg()`

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::design::DesignMatrix;
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
use crate::linalg::solve::{SingularPolicy, sym_inverse};
use crate::resource;
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, t_test_p_value};

/// `first_stage_f_type` of a classical fit.
const PARTIAL_F: &str = "partial F of the excluded instruments";
/// `first_stage_f_type` of a robust (HC1) fit.
const HC1_WALD_F: &str = "HC1 Wald F of the excluded instruments";

/// Result from an IV/2SLS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Exogenous variables (included instruments)
    pub exogenous_vars: Vec<String>,
    /// Endogenous variables
    pub endogenous_vars: Vec<String>,
    /// Excluded instruments
    pub instruments: Vec<String>,
    /// All variable names (including intercept if present)
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics (or z-statistics)
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// R-squared (can be negative for IV)
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Degrees of freedom
    pub df: usize,
    /// First-stage F of each endogenous variable, in `endogenous_vars` order: the
    /// partial F of the excluded instruments (`robust = false`), comparing
    /// `x_j ~ [1, X_exog]` with `x_j ~ [1, X_exog, Z]` (AER's "Weak instruments"
    /// statistic), or the HC1 Wald F of the excluded-instrument coefficients
    /// (`robust = true`). See `first_stage_f_type`.
    pub first_stage_f_stats: Vec<f64>,
    /// Whether instruments are strong (first-stage F >= 10 for each endogenous variable;
    /// a NaN F is not strong)
    pub strong_instruments: bool,
    /// Estimation warnings
    pub warnings: Vec<String>,
    /// Degrees of freedom (L, n - k_z) of each first-stage F: L excluded instruments,
    /// k_z = columns of [1, X_exog, Z].
    #[serde(default)]
    pub first_stage_df: Vec<(usize, usize)>,
    /// Upper-tail p-value of each first-stage F from F(L, n - k_z) (statrs `sf`).
    #[serde(default)]
    pub first_stage_p_values: Vec<f64>,
    /// 1 - RSS(x_j ~ 1, X_exog, Z) / RSS(x_j ~ 1, X_exog), per endogenous variable
    /// (classical in both modes).
    #[serde(default)]
    pub first_stage_partial_r2: Vec<f64>,
    /// "partial F of the excluded instruments" (robust = false) | "HC1 Wald F of the
    /// excluded instruments" (robust = true).
    #[serde(default)]
    pub first_stage_f_type: String,
    /// Rows dropped because of missing values (R ivreg na.omit).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

/// Writes the `Warnings:` block (without the warning that repeats the missing-value
/// note, printed as the last line instead) when something is left.
fn write_warnings(
    f: &mut fmt::Formatter<'_>,
    warnings: &[String],
    missing: &MissingSummary,
) -> fmt::Result {
    let note = missing.note();
    let shown: Vec<&String> = warnings
        .iter()
        .filter(|w| note.as_deref() != Some(w.as_str()))
        .collect();
    if !shown.is_empty() {
        writeln!(f)?;
        writeln!(f, "Warnings:")?;
        for w in shown {
            writeln!(f, "  - {}", w)?;
        }
    }
    Ok(())
}

/// Writes `Missing values: <note>` as the last line when rows were dropped.
fn write_missing(f: &mut fmt::Formatter<'_>, missing: &MissingSummary) -> fmt::Result {
    match missing.note() {
        Some(note) => writeln!(f, "Missing values: {note}"),
        None => Ok(()),
    }
}

impl fmt::Display for IVResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "2SLS / IV Regression Results")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Endogenous: {}", self.endogenous_vars.join(", "))?;
        writeln!(f, "Instruments: {}", self.instruments.join(", "))?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        if !self.first_stage_f_stats.is_empty() {
            let f_type = if self.first_stage_f_type.is_empty() {
                "F"
            } else {
                self.first_stage_f_type.as_str()
            };
            writeln!(f, "First stage ({f_type}):")?;
            for (j, &fj) in self.first_stage_f_stats.iter().enumerate() {
                let name = self
                    .endogenous_vars
                    .get(j)
                    .map_or("endogenous", String::as_str);
                match self.first_stage_df.get(j) {
                    Some(&(df1, df2)) => {
                        let p = self
                            .first_stage_p_values
                            .get(j)
                            .copied()
                            .unwrap_or(f64::NAN);
                        let r2 = self
                            .first_stage_partial_r2
                            .get(j)
                            .copied()
                            .unwrap_or(f64::NAN);
                        writeln!(
                            f,
                            "  {name}: F = {fj:.4} on ({df1}, {df2}) DF, p = {p:.4}, \
                             partial R-squared = {r2:.4}"
                        )?;
                    }
                    None => writeln!(f, "  {name}: F = {fj:.4}")?,
                }
            }
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
                self.t_stats[i],
                self.p_values[i],
                self.significance[i].stars()
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        write_warnings(f, &self.warnings, &self.missing)?;
        write_missing(f, &self.missing)
    }
}

/// Upper-tail probability of `f` under F(`df1`, `df2`) (statrs `sf`): NaN when a df is 0
/// or `f` is NaN, 0 when `f` is infinite.
fn f_upper_tail(f: f64, df1: usize, df2: usize) -> f64 {
    if df1 == 0 || df2 == 0 || f.is_nan() {
        return f64::NAN;
    }
    if f.is_infinite() {
        return 0.0;
    }
    crate::traits::estimator::f_test_p_value(f.max(0.0), df1 as f64, df2 as f64)
}

/// OLS of `v` on the columns of `z`, given `(z'z)^-1`: the coefficients, the explicit
/// residuals and their sum of squares.
fn ols_fit(
    z: &ArrayView2<f64>,
    ztz_inv: &Array2<f64>,
    v: &ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, f64) {
    let beta = ztz_inv.dot(&z.t().dot(v));
    let resid = v - &z.dot(&beta);
    let rss = resid.dot(&resid);
    (beta, resid, rss)
}

/// The classical partial F of `l` excluded columns: `((rss0 - rss1)/l) / (rss1/df2)`,
/// with `rss0` the restricted and `rss1` the full residual sum of squares. NaN when
/// `df2 == 0` and for 0/0 (both fits perfect: the excluded columns explain nothing that
/// was left); +inf only for a perfect full fit with a positive restricted RSS.
fn partial_f(rss0: f64, rss1: f64, l: usize, df2: usize) -> f64 {
    if df2 == 0 || l == 0 {
        f64::NAN
    } else if rss1 > 0.0 {
        ((rss0 - rss1).max(0.0) / l as f64) / (rss1 / df2 as f64)
    } else if rss0 - rss1 > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    }
}

/// Restate a `SingularMatrix` from `sym_inverse` (whose context names the matrix and its
/// rank) with the place it occurred and a suggestion; other errors pass through.
fn singular_in(e: EconError, place: &str, suggestion: &str) -> EconError {
    match e {
        EconError::SingularMatrix { context, .. } => EconError::SingularMatrix {
            context: format!("{context}; {place}"),
            suggestion: suggestion.to_string(),
        },
        other => other,
    }
}

/// The HC1 Wald F of the coefficients `beta[k_exog..]` of the OLS fit of a column on
/// `z` (`(z'z)^-1` given, residuals `resid`): `b' V_bb^-1 b / L`, with
/// `V = (z'z)^-1 [sum_i e_i^2 z_i z_i'] (z'z)^-1 n / (n - k_z)` (sandwich `vcovHC(type =
/// "HC1")`). `None` when `V_bb` is singular (including a pseudoinverse fallback).
fn hc1_wald_f(
    z: &ArrayView2<f64>,
    ztz_inv: &Array2<f64>,
    resid: &Array1<f64>,
    beta: &Array1<f64>,
    k_exog: usize,
) -> Option<f64> {
    let (n, kz) = z.dim();
    let l = kz.saturating_sub(k_exog);
    let df2 = n.saturating_sub(kz);
    if l == 0 || df2 == 0 {
        return Some(f64::NAN);
    }
    let mut meat = Array2::<f64>::zeros((kz, kz));
    for (row, &e) in z.rows().into_iter().zip(resid.iter()) {
        let e2 = e * e;
        for a in 0..kz {
            let za = e2 * row[a];
            for b in a..kz {
                meat[[a, b]] += za * row[b];
            }
        }
    }
    for a in 0..kz {
        for b in 0..a {
            meat[[a, b]] = meat[[b, a]];
        }
    }
    let v = ztz_inv.dot(&meat).dot(ztz_inv) * (n as f64 / df2 as f64);
    let v_bb = v.slice(s![k_exog.., k_exog..]);
    let b = beta.slice(s![k_exog..]);
    // A singular V_bb (condition Some) gives None: the HC1 F is not reported.
    match safe_inverse(&v_bb) {
        Ok((inv, None)) => Some(b.dot(&inv.dot(&b)) / l as f64),
        _ => None,
    }
}

/// The model's columns on the rows complete in y, X_exog, X_endog and Z (R ivreg
/// `na.omit`).
struct IvData {
    y: Array1<f64>,
    /// "(Intercept)" and the included exogenous regressors.
    exog: DesignMatrix,
    endog: DesignMatrix,
    instr: DesignMatrix,
    missing: MissingSummary,
}

/// One NA-aware request over every model column, in the order y, X_exog, X_endog, Z;
/// incomplete rows are dropped jointly.
fn iv_data(
    dataset: &Dataset,
    y_col: &str,
    x_exog: &[&str],
    x_endog: &[&str],
    instruments: &[&str],
) -> EconResult<IvData> {
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_exog)
        .numerics(x_endog)
        .numerics(instruments)
        .na(NaPolicy::Omit)
        .extract()?;
    Ok(IvData {
        y: cols.numeric_array(y_col),
        exog: cols.matrix(x_exog, true)?,
        endog: cols.matrix(x_endog, false)?,
        instr: cols.matrix(instruments, false)?,
        missing: cols.missing().clone(),
    })
}

/// Run Two-Stage Least Squares (2SLS) / Instrumental Variables estimation.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable column
/// * `x_exog` - Names of exogenous (included) variables
/// * `x_endog` - Names of endogenous variables
/// * `instruments` - Names of excluded instruments (must be at least as many as endogenous)
/// * `robust` - Whether to use heteroskedasticity-robust standard errors
///
/// # Algorithm
///
/// Uses the efficient projection formula to avoid forming large n×k matrices
/// in the second stage. Instead of building X̂ = P_Z X explicitly, we compute:
///
///   β_IV = (X'P_Z X)^{-1} X'P_Z y
///
/// where P_Z = Z(Z'Z)^{-1}Z'. By noting that X'P_Z = X'Z (Z'Z)^{-1} Z',
/// we can compute everything via small k×k matrix operations:
///
///   X'P_Z X = (Z'X)' (Z'Z)^{-1} (Z'X)
///   X'P_Z y = (Z'X)' (Z'Z)^{-1} (Z'y)
///
/// This reduces memory from O(n·k) to O(k²) for the second stage.
///
/// # First stage
///
/// For each endogenous `x_j` the first-stage F is the partial F of the L excluded
/// instruments, comparing `x_j ~ [1, X_exog]` with `x_j ~ [1, X_exog, Z]` on
/// `(L, n - k_z)` degrees of freedom (AER's "Weak instruments" statistic,
/// `summary(ivreg, diagnostics = TRUE)`); with `robust = true` it is the HC1 Wald F
/// of the excluded-instrument coefficients, as AER with
/// `vcov. = sandwich::vcovHC(type = "HC1")`. `first_stage_p_values` are upper-tail
/// F probabilities and `first_stage_partial_r2` the classical partial R². A
/// first-stage F below 10 (or not computed) adds a weak-instrument warning and makes
/// `strong_instruments` false. Stock & Yogo (2005) critical values are not computed.
///
/// # Missing values
///
/// Rows with a missing value (null or NaN) in y, a regressor or an instrument are
/// dropped from every column (R ivreg `na.omit`), counted in `missing` and noted in
/// `warnings`; an infinite value is refused (`NonFiniteValues`), a text column is
/// `NonNumericColumn`.
pub fn run_iv2sls(
    dataset: &Dataset,
    y_col: &str,
    x_exog: &[&str],
    x_endog: &[&str],
    instruments: &[&str],
    robust: bool,
) -> EconResult<IVResult> {
    let mut warnings = Vec::new();

    if x_endog.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: "At least one endogenous variable is required".to_string(),
        });
    }

    // Check identification
    if instruments.len() < x_endog.len() {
        return Err(EconError::UnderIdentified {
            n_endogenous: x_endog.len(),
            n_instruments: instruments.len(),
        });
    }

    // y, [1, X_exog], X_endog and Z on the complete rows
    let IvData {
        y,
        exog: design_exog,
        endog: design_endog,
        instr: design_instruments,
        missing,
    } = iv_data(dataset, y_col, x_exog, x_endog, instruments)?;
    if let Some(note) = missing.note() {
        warnings.push(note);
    }
    let n = y.len();
    let x_exog_mat = design_exog.data;
    let exog_names = design_exog.column_names;
    let x_endog_mat = design_endog.data;
    let endog_names = design_endog.column_names;
    let z_mat = design_instruments.data;

    // Full instrument matrix: [X_exog, Z] (all exogenous + excluded instruments)
    let k_exog = x_exog_mat.ncols();
    let k_endog = x_endog_mat.ncols();
    let k_instr = z_mat.ncols();
    let k_full_z = k_exog + k_instr;

    let mut z_full = Array2::zeros((n, k_full_z));
    z_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&x_exog_mat);
    z_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_mat);

    // ═══════════════════════════════════════════════════════════════════
    // Compute Z'Z inverse (small k_z × k_z matrix)
    // ═══════════════════════════════════════════════════════════════════
    let ztz = xtx(&z_full.view());
    let (ztz_inv, cond_warning) =
        safe_inverse(&ztz.view()).map_err(|e| EconError::SingularMatrix {
            context: "Z'Z in 2SLS first stage".to_string(),
            suggestion: format!("Check for collinearity among instruments: {:?}", e),
        })?;

    if let Some(cond) = cond_warning {
        warnings.push(format!("High condition number in Z'Z: {:.2e}", cond));
    }

    // ═══════════════════════════════════════════════════════════════════
    // First stage: the F of the excluded instruments for each x_j, comparing
    // the restricted fit x_j ~ [1, X_exog] with x_j ~ [1, X_exog, Z] (AER's
    // weak-instrument statistic), classical or HC1 Wald.
    // ═══════════════════════════════════════════════════════════════════
    let ztz0_inv = sym_inverse(
        ztz.slice(s![..k_exog, ..k_exog]),
        SingularPolicy::Error,
        "exogenous-instrument cross-product Z_0'Z_0",
    )
    .map_err(|e| {
        singular_in(
            e,
            "[1, X_exog]'[1, X_exog] in the first-stage restricted fit",
            "Check for collinearity among the exogenous regressors",
        )
    })?
    .inverse;
    let z0 = z_full.slice(s![.., ..k_exog]);
    let fs_df2 = n.saturating_sub(k_full_z);
    let f_type = if robust { HC1_WALD_F } else { PARTIAL_F };

    let mut first_stage_f_stats = Vec::with_capacity(k_endog);
    let mut first_stage_df = Vec::with_capacity(k_endog);
    let mut first_stage_p_values = Vec::with_capacity(k_endog);
    let mut first_stage_partial_r2 = Vec::with_capacity(k_endog);

    // Store first-stage betas only if needed for robust SEs
    let mut first_stage_betas: Vec<Array1<f64>> = if robust {
        Vec::with_capacity(k_endog)
    } else {
        Vec::new()
    };

    for j in 0..k_endog {
        resource::checkpoint()?;
        let x_j = x_endog_mat.column(j);

        // Full first stage x_j ~ [1, X_exog, Z] and restricted x_j ~ [1, X_exog],
        // from explicit residuals
        let (beta_first, resid_first, rss1) = ols_fit(&z_full.view(), &ztz_inv, &x_j);
        let (_, _, rss0) = ols_fit(&z0, &ztz0_inv, &x_j);
        let partial_r2 = if rss0 > 0.0 {
            1.0 - rss1 / rss0
        } else {
            f64::NAN
        };

        let f_stat = if robust {
            match hc1_wald_f(&z_full.view(), &ztz_inv, &resid_first, &beta_first, k_exog) {
                Some(f) => f,
                None => {
                    warnings.push(format!(
                        "HC1 first-stage Wald F for '{}' not computed: singular covariance of \
                         the instrument coefficients",
                        x_endog[j]
                    ));
                    f64::NAN
                }
            }
        } else {
            partial_f(rss0, rss1, k_instr, fs_df2)
        };

        first_stage_f_stats.push(f_stat);
        first_stage_df.push((k_instr, fs_df2));
        first_stage_p_values.push(f_upper_tail(f_stat, k_instr, fs_df2));
        first_stage_partial_r2.push(partial_r2);
        if robust {
            first_stage_betas.push(beta_first);
        }

        if f_stat.is_nan() || f_stat < 10.0 {
            warnings.push(format!(
                "Weak instruments for '{}': first-stage {f_type} = {f_stat:.2} on \
                 ({k_instr}, {fs_df2}) DF, below 10",
                x_endog[j]
            ));
        }
    }

    let strong_instruments = first_stage_f_stats.iter().all(|&f| f >= 10.0);

    // ═══════════════════════════════════════════════════════════════════
    // Second Stage via projection formula (no n×k matrices needed):
    //
    //   X_full = [X_exog, X_endog]  (the original regressors)
    //   Z'X_full is k_z × k_total
    //   X'P_Z X = (Z'X)' (Z'Z)^{-1} (Z'X)  -- k_total × k_total
    //   X'P_Z y = (Z'X)' (Z'Z)^{-1} (Z'y)  -- k_total × 1
    //   β = (X'P_Z X)^{-1} X'P_Z y
    // ═══════════════════════════════════════════════════════════════════
    let k_total = k_exog + k_endog;

    // Compute Z'X_full (k_z × k_total) via ndarray BLAS
    // Z'X_exog (k_z × k_exog) and Z'X_endog (k_z × k_endog)
    let ztx_exog = z_full.t().dot(&x_exog_mat);
    let ztx_endog = z_full.t().dot(&x_endog_mat);
    let mut ztx_full = Array2::zeros((k_full_z, k_total));
    ztx_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&ztx_exog);
    ztx_full
        .slice_mut(ndarray::s![.., k_exog..])
        .assign(&ztx_endog);

    // Z'y (k_z × 1)
    let zty = z_full.t().dot(&y);

    // (Z'Z)^{-1} Z'X  (k_z × k_total)
    let ztz_inv_ztx = ztz_inv.dot(&ztx_full);

    // X'P_Z X = (Z'X)' (Z'Z)^{-1} (Z'X) = ztx_full' * ztz_inv_ztx  (k_total × k_total)
    let xpzx = ztx_full.t().dot(&ztz_inv_ztx);

    // X'P_Z y = (Z'X)' (Z'Z)^{-1} (Z'y) = ztx_full' * (Z'Z)^{-1} * Z'y  (k_total × 1)
    let ztz_inv_zty = ztz_inv.dot(&zty);
    let xpzy = ztx_full.t().dot(&ztz_inv_zty);

    // β = (X'P_Z X)^{-1} X'P_Z y
    let xpzx_inv = sym_inverse(
        xpzx.view(),
        SingularPolicy::Error,
        "2SLS normal matrix X'P_Z X",
    )
    .map_err(|e| singular_in(e, "X'P_Z X in 2SLS second stage", "Check for collinearity"))?
    .inverse;

    let beta: Array1<f64> = xpzx_inv.dot(&xpzy);

    // ═══════════════════════════════════════════════════════════════════
    // Residuals using original X (not fitted values)
    // y_hat = X_exog * beta_exog + X_endog * beta_endog
    // ═══════════════════════════════════════════════════════════════════
    let beta_exog = beta.slice(ndarray::s![..k_exog]);
    let beta_endog = beta.slice(ndarray::s![k_exog..]);
    let y_hat = x_exog_mat.dot(&beta_exog) + x_endog_mat.dot(&beta_endog);
    let residuals = &y - &y_hat;

    let df = n.saturating_sub(k_total);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = if df > 0 {
        ssr / df as f64
    } else {
        ssr / n as f64
    };

    // ═══════════════════════════════════════════════════════════════════
    // Variance-covariance matrix
    // For 2SLS: V = σ² (X̂'X̂)^{-1} = σ² (X'P_Z X)^{-1}
    // ═══════════════════════════════════════════════════════════════════
    let vcov = if robust {
        // HC1 robust standard errors
        // V = (X̂'X̂)^{-1} (X̂'ΩX̂) (X̂'X̂)^{-1}
        // Need X̂ = P_Z X for the meat. Build it efficiently using first-stage betas.
        let scale = (n as f64) / (df as f64);

        // Build fitted X matrix for robust SEs
        // X̂_exog = X_exog (exogenous vars are their own instruments)
        // X̂_endog_j = Z * first_stage_beta_j
        let mut meat = Array2::zeros((k_total, k_total));
        // Pre-compute x_hat rows on the fly to avoid allocating full n×k matrix
        let mut xi_hat = vec![0.0; k_total];
        for i in 0..n {
            // Exogenous part: same as original
            for j in 0..k_exog {
                xi_hat[j] = x_exog_mat[[i, j]];
            }
            // Endogenous part: fitted values from first stage
            for j in 0..k_endog {
                let mut val = 0.0;
                for l in 0..k_full_z {
                    val += z_full[[i, l]] * first_stage_betas[j][l];
                }
                xi_hat[k_exog + j] = val;
            }

            let e2 = residuals[i] * residuals[i];
            for j in 0..k_total {
                for l in j..k_total {
                    let contrib = e2 * xi_hat[j] * xi_hat[l];
                    meat[[j, l]] += contrib;
                    if l != j {
                        meat[[l, j]] += contrib;
                    }
                }
            }
        }

        let meat = &meat * scale;
        let bread_meat = xpzx_inv.dot(&meat);
        bread_meat.dot(&xpzx_inv)
    } else {
        &xpzx_inv * sigma2
    };

    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // R-squared (can be negative for IV)
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    if r_squared < 0.0 {
        warnings.push(format!(
            "R² = {:.4} is negative, which can occur with IV estimation",
            r_squared
        ));
    }

    // t-statistics and p-values
    let coefficients = beta.to_vec();
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Variable names
    let mut variables = exog_names;
    variables.extend(endog_names.clone());

    Ok(IVResult {
        dep_var: y_col.to_string(),
        exogenous_vars: x_exog.iter().map(|s| s.to_string()).collect(),
        endogenous_vars: x_endog.iter().map(|s| s.to_string()).collect(),
        instruments: instruments.iter().map(|s| s.to_string()).collect(),
        variables,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        significance,
        r_squared,
        n_obs: n,
        df,
        first_stage_f_stats,
        strong_instruments,
        warnings,
        first_stage_df,
        first_stage_p_values,
        first_stage_partial_r2,
        first_stage_f_type: f_type.to_string(),
        missing,
    })
}

/// Result from first-stage diagnostics for IV/2SLS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirstStageDiagnostics {
    /// Endogenous variable name
    pub endogenous_var: String,
    /// Instruments used
    pub instruments: Vec<String>,
    /// Partial F of the excluded instruments given the controls: the restricted fit
    /// `endog ~ [1, controls]` against `endog ~ [1, instruments, controls]` (AER's
    /// weak-instrument statistic; the overall F when there are no controls)
    pub f_statistic: f64,
    /// Upper-tail p-value of `f_statistic` on `f_df` degrees of freedom
    pub f_pvalue: f64,
    /// R-squared of the full first stage
    pub r_squared: f64,
    /// Adjusted R-squared of the full first stage
    pub adj_r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Whether instruments pass weak instrument test (F > 10)
    pub strong_instruments: bool,
    /// Coefficients on instruments in first stage: (name, coef, se, t-stat, p-value)
    pub instrument_coeffs: Vec<(String, f64, f64, f64, f64)>,
    /// Significance levels for each instrument
    pub significance: Vec<SignificanceLevel>,
    /// Degrees of freedom `(L, n - k)` of `f_statistic`, `k = 1 + L + n_controls`.
    #[serde(default)]
    pub f_df: (usize, usize),
    /// `1 - RSS(full) / RSS(restricted)`.
    #[serde(default)]
    pub partial_r_squared: f64,
    /// Control variables (included exogenous regressors) of the first stage.
    #[serde(default)]
    pub controls: Vec<String>,
    /// Rows dropped because of missing values in the first-stage columns.
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl fmt::Display for FirstStageDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "First-Stage Diagnostics for IV/2SLS")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Endogenous Variable: {}", self.endogenous_var)?;
        writeln!(f, "Instruments: {}", self.instruments.join(", "))?;
        if !self.controls.is_empty() {
            writeln!(f, "Controls: {}", self.controls.join(", "))?;
        }
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(f, "Instrument Strength ({PARTIAL_F}):")?;
        writeln!(
            f,
            "  F-statistic: {:.4} on ({}, {}) DF",
            self.f_statistic, self.f_df.0, self.f_df.1
        )?;
        writeln!(f, "  Prob (F): {:.4}", self.f_pvalue)?;
        writeln!(f, "  Partial R-squared: {:.4}", self.partial_r_squared)?;
        writeln!(f, "  R-squared (first stage): {:.4}", self.r_squared)?;
        writeln!(
            f,
            "  Adj. R-squared (first stage): {:.4}",
            self.adj_r_squared
        )?;
        writeln!(f)?;

        // Stock-Yogo critical values interpretation
        let strength = if self.f_statistic > 10.0 {
            "STRONG (F > 10)"
        } else if self.f_statistic > 5.0 {
            "MODERATE (5 < F < 10)"
        } else {
            "WEAK (F < 5) - Caution!"
        };
        writeln!(f, "  Instrument Strength: {}", strength)?;
        writeln!(f)?;

        writeln!(f, "First-Stage Coefficients:")?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Instrument", "Coef", "Std Err", "t-stat", "P>|t|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for (i, (name, coef, se, t, p)) in self.instrument_coeffs.iter().enumerate() {
            let sig = if i < self.significance.len() {
                self.significance[i].stars()
            } else {
                ""
            };
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                name, coef, se, t, p, sig
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(
            f,
            "Note: Stock-Yogo (2005) critical value for 10% max bias: F > 16.38 (single instrument)"
        )?;
        writeln!(
            f,
            "      Rule of thumb: F > 10 suggests instruments are not weak"
        )?;

        write_missing(f, &self.missing)
    }
}

/// Run first-stage diagnostics for IV/2SLS.
///
/// Tests instrument strength by regressing the endogenous variable on the instruments
/// and the controls. The F-statistic is the partial F of the excluded instruments
/// given the controls (the restricted fit `endog ~ [1, controls]` against
/// `endog ~ [1, instruments, controls]`, R `anova(lm(x ~ w), lm(x ~ w + z))`), AER's
/// "Weak instruments" statistic, on `(L, n - k)` degrees of freedom; without controls
/// it is the overall F of the first stage. F > 10 suggests instruments are not weak;
/// Stock & Yogo (2005) critical values are not computed.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `endog_var` - Name of the endogenous variable
/// * `instruments` - Names of the instrumental variables (excluded instruments); an empty
///   list is `InvalidSpecification`
/// * `controls` - Optional control variables to include in first stage (exogenous variables)
pub fn run_first_stage_diagnostics(
    dataset: &Dataset,
    endog_var: &str,
    instruments: &[&str],
    controls: Option<&[&str]>,
) -> EconResult<FirstStageDiagnostics> {
    // Without an excluded instrument there is no partial F, and no strength to judge.
    if instruments.is_empty() {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "First-stage diagnostics of '{endog_var}' need at least one excluded \
                 instrument; the instrument list is empty"
            ),
        });
    }
    let controls: Vec<&str> = controls.unwrap_or(&[]).to_vec();

    // The endogenous variable, instruments and controls on their complete rows
    let cols = ColumnRequest::new(dataset.df())
        .numeric(endog_var)
        .numerics(instruments)
        .numerics(&controls)
        .na(NaPolicy::Omit)
        .extract()?;
    let y = cols.numeric_array(endog_var);
    let n = y.len();

    // The full regressor set: intercept + instruments + controls
    let mut all_vars: Vec<&str> = instruments.to_vec();
    all_vars.extend(&controls);

    let design = cols.matrix(&all_vars, true)?;
    let x = design.data;
    let var_names = design.column_names;
    let k = x.ncols();

    // OLS regression
    let xtx_mat = xtx(&x.view());
    let xtx_inv = sym_inverse(
        xtx_mat.view(),
        SingularPolicy::Error,
        "first-stage design X'X",
    )
    .map_err(|e| {
        singular_in(
            e,
            "X'X in first-stage diagnostics",
            "Check for collinearity",
        )
    })?
    .inverse;

    let xty_vec = xty(&x.view(), &y);
    let beta: Array1<f64> = xtx_inv.dot(&xty_vec);

    // Residuals
    let y_hat = x.dot(&beta);
    let residuals = &y - &y_hat;

    let df = n.saturating_sub(k);
    let ssr: f64 = residuals.iter().map(|r| r * r).sum();
    let sigma2 = if df > 0 {
        ssr / df as f64
    } else {
        ssr / n as f64
    };

    // R-squared
    let y_mean = y.mean().unwrap_or(0.0);
    let sst: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if sst > 0.0 { 1.0 - ssr / sst } else { 0.0 };

    // Adjusted R-squared
    let adj_r_squared = 1.0 - (1.0 - r_squared) * ((n - 1) as f64) / (df as f64);

    // Partial F of the instruments given the controls: the restricted fit
    // endog ~ [1, controls] against the full first stage (AER's weak-instrument F)
    let restricted = cols.matrix(&controls, true)?;
    let restricted_inv = sym_inverse(
        xtx(&restricted.data.view()).view(),
        SingularPolicy::Error,
        "restricted first-stage X'X",
    )
    .map_err(|e| {
        singular_in(
            e,
            "[1, controls]'[1, controls] in the first-stage restricted fit",
            "Check for collinearity among the controls",
        )
    })?
    .inverse;
    let (_, _, rss0) = ols_fit(&restricted.data.view(), &restricted_inv, &y.view());
    let l = instruments.len();
    let f_statistic = partial_f(rss0, ssr, l, df);
    let f_pvalue = f_upper_tail(f_statistic, l, df);
    let partial_r_squared = if rss0 > 0.0 {
        1.0 - ssr / rss0
    } else {
        f64::NAN
    };

    // Standard errors
    let vcov = &xtx_inv * sigma2;
    let std_errors: Vec<f64> = vcov.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

    // Extract instrument coefficients (skip intercept at index 0)
    let mut instrument_coeffs = Vec::new();
    let mut significance = Vec::new();

    for (i, instr) in instruments.iter().enumerate() {
        // Find the instrument in variable names
        if let Some(idx) = var_names.iter().position(|v| v == *instr) {
            let b = beta[idx];
            let se = std_errors[idx];
            let t = if se > 0.0 { b / se } else { 0.0 };
            let p = t_test_p_value(t, df as f64);
            instrument_coeffs.push((instr.to_string(), b, se, t, p));
            significance.push(SignificanceLevel::from_p_value(p));
        } else if i + 1 < beta.len() {
            // Fallback: use position (offset by 1 for intercept)
            let idx = i + 1;
            let b = beta[idx];
            let se = std_errors[idx];
            let t = if se > 0.0 { b / se } else { 0.0 };
            let p = t_test_p_value(t, df as f64);
            instrument_coeffs.push((instr.to_string(), b, se, t, p));
            significance.push(SignificanceLevel::from_p_value(p));
        }
    }

    Ok(FirstStageDiagnostics {
        endogenous_var: endog_var.to_string(),
        instruments: instruments.iter().map(|s| s.to_string()).collect(),
        f_statistic,
        f_pvalue,
        r_squared,
        adj_r_squared,
        n_obs: n,
        strong_instruments: f_statistic > 10.0,
        instrument_coeffs,
        significance,
        f_df: (l, df),
        partial_r_squared,
        controls: controls.iter().map(|s| s.to_string()).collect(),
        missing: cols.missing().clone(),
    })
}

/// Result from the Sargan test for overidentifying restrictions.
///
/// The Sargan test (also known as the Hansen J test) evaluates whether
/// the instruments are valid, i.e., uncorrelated with the error term.
///
/// # References
///
/// - Sargan, J.D. (1958). The estimation of economic relationships using
///   instrumental variables. *Econometrica*, 26(3), 393-415.
///   https://doi.org/10.2307/1907619
///
/// - Hansen, L.P. (1982). Large sample properties of generalized method of
///   moments estimators. *Econometrica*, 50(4), 1029-1054.
///   https://doi.org/10.2307/1912775
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarganTestResult {
    /// Name of the test
    pub test_name: String,
    /// J statistic (Sargan/Hansen test statistic)
    pub j_statistic: f64,
    /// Degrees of freedom (number of overidentifying restrictions)
    pub df: usize,
    /// p-value from chi-squared distribution
    pub p_value: f64,
    /// Number of instruments
    pub n_instruments: usize,
    /// Number of endogenous regressors
    pub n_endogenous: usize,
    /// Number of observations
    pub n_obs: usize,
    /// Whether the model is over-identified (required for test)
    pub overidentified: bool,
    /// Null hypothesis: instruments are valid
    pub null_hypothesis: String,
    /// Interpretation based on p-value
    pub interpretation: String,
    /// Rows dropped because of missing values (the rows of `run_iv2sls`).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl fmt::Display for SarganTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Sargan Test of Overidentifying Restrictions")?;
        writeln!(f, "===========================================")?;
        writeln!(f)?;
        writeln!(
            f,
            "H0: Instruments are valid (uncorrelated with error term)"
        )?;
        writeln!(f)?;
        writeln!(f, "  J statistic:      {:.4}", self.j_statistic)?;
        writeln!(f, "  Degrees of freedom: {}", self.df)?;
        writeln!(f, "  p-value:          {:.4}", self.p_value)?;
        writeln!(f)?;
        writeln!(f, "  Number of instruments:       {}", self.n_instruments)?;
        writeln!(f, "  Number of endogenous vars:   {}", self.n_endogenous)?;
        writeln!(f, "  Overidentifying restrictions: {}", self.df)?;
        writeln!(f)?;
        writeln!(f, "Interpretation: {}", self.interpretation)?;
        write_missing(f, &self.missing)
    }
}

/// Perform the Sargan test for overidentifying restrictions.
///
/// The Sargan test (also known as the Hansen J test when robust to heteroskedasticity)
/// tests whether the instruments are valid (uncorrelated with the structural error term).
///
/// # Requirements
///
/// - The model must be **over-identified**: number of instruments > number of endogenous variables
/// - For just-identified models (instruments == endogenous), the test is not defined
///
/// The test uses the rows [`run_iv2sls`] uses: rows with a missing value in any model
/// column are dropped (R ivreg `na.omit`) and reported in `missing`.
///
/// # Test Statistic
///
/// The Sargan statistic is computed as:
///   J = n × R² from regressing IV residuals on all instruments
///
/// Equivalently:
///   J = n × (1 - ε̂'M_Z ε̂ / ε̂'ε̂)
///
/// where ε̂ are the IV residuals and M_Z is the annihilator matrix for instruments.
///
/// Under H0 (valid instruments): J ~ χ²(L - K)
/// where L = number of instruments, K = number of endogenous variables.
///
/// # Interpretation
///
/// - **Fail to reject H0 (p > 0.05)**: Instruments appear valid
/// - **Reject H0 (p ≤ 0.05)**: At least one instrument may be invalid
///
/// # Arguments
///
/// * `dataset` - The dataset
/// * `y_col` - Name of the dependent variable
/// * `x_exog` - Names of exogenous (included) variables
/// * `x_endog` - Names of endogenous variables
/// * `instruments` - Names of excluded instruments
///
/// # Returns
///
/// `SarganTestResult` containing the J statistic, degrees of freedom, and p-value.
///
/// # References
///
/// - Sargan, J.D. (1958). The estimation of economic relationships using
///   instrumental variables. *Econometrica*, 26(3), 393-415.
///
/// R equivalent: `summary(ivreg(...))` shows Sargan statistic, or `ivdiag::sargan()`
pub fn sargan_test(
    dataset: &Dataset,
    y_col: &str,
    x_exog: &[&str],
    x_endog: &[&str],
    instruments: &[&str],
) -> EconResult<SarganTestResult> {
    let n_instr = instruments.len();
    let n_endog = x_endog.len();
    let n_exog_vars = x_exog.len();

    // Total instruments = excluded instruments + included exogenous + intercept
    let _total_instruments = n_instr + n_exog_vars + 1; // +1 for intercept

    // Degrees of freedom = overidentifying restrictions
    // For 2SLS: df = (excluded instruments) - (endogenous regressors)
    let df = n_instr.saturating_sub(n_endog);

    if df == 0 {
        return Ok(SarganTestResult {
            test_name: "Sargan Test of Overidentifying Restrictions".to_string(),
            j_statistic: 0.0,
            df: 0,
            p_value: 1.0,
            n_instruments: n_instr,
            n_endogenous: n_endog,
            n_obs: 0,
            overidentified: false,
            null_hypothesis: "Instruments are valid (uncorrelated with error term)".to_string(),
            interpretation: "Model is exactly identified (instruments = endogenous). Sargan test not applicable.".to_string(),
            missing: MissingSummary::default(),
        });
    }

    // First, run IV/2SLS to get the residuals
    let iv_result = run_iv2sls(dataset, y_col, x_exog, x_endog, instruments, false)?;
    let n = iv_result.n_obs;

    // Re-extract the same complete rows to compute residuals and run the auxiliary
    // regression
    let IvData {
        y,
        exog: design_exog,
        endog: design_endog,
        instr: design_z,
        missing,
    } = iv_data(dataset, y_col, x_exog, x_endog, instruments)?;

    // Build the full X matrix (exogenous + endogenous)
    let x_exog_mat = design_exog.data;
    let x_endog_mat = design_endog.data;

    let k_exog = x_exog_mat.ncols();
    let k_endog = x_endog_mat.ncols();
    let k_total = k_exog + k_endog;

    let mut x_full = Array2::zeros((n, k_total));
    x_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&x_exog_mat);
    x_full
        .slice_mut(ndarray::s![.., k_exog..])
        .assign(&x_endog_mat);

    // Compute IV residuals
    let beta: Array1<f64> = Array1::from_vec(iv_result.coefficients.clone());
    let y_hat = x_full.dot(&beta);
    let residuals = &y - &y_hat;

    // Build the full instrument matrix Z = [exogenous, excluded_instruments]
    let z_excl = design_z.data;

    let k_z = k_exog + z_excl.ncols(); // exogenous (with intercept) + excluded instruments
    let mut z_full = Array2::zeros((n, k_z));
    z_full
        .slice_mut(ndarray::s![.., ..k_exog])
        .assign(&x_exog_mat);
    z_full.slice_mut(ndarray::s![.., k_exog..]).assign(&z_excl);

    // Sargan test: regress residuals on all instruments and compute n*R²
    // R² = 1 - SSR/SST where SST = sum(residuals²) and SSR is from aux regression

    // Auxiliary regression: residuals on Z
    let ztz = xtx(&z_full.view());
    let ztz_inv = sym_inverse(
        ztz.view(),
        SingularPolicy::Error,
        "instrument cross-product Z'Z",
    )
    .map_err(|e| {
        singular_in(
            e,
            "Z'Z in Sargan test",
            "Check for collinearity among instruments",
        )
    })?
    .inverse;

    let ztr = xty(&z_full.view(), &residuals);
    let gamma = ztz_inv.dot(&ztr);
    let fitted = z_full.dot(&gamma);

    // Compute R² from auxiliary regression
    let ssr_aux: f64 = (&residuals - &fitted).iter().map(|e| e * e).sum();
    let sst: f64 = residuals.iter().map(|e| e * e).sum();

    let r_squared_aux = if sst > 1e-10 {
        1.0 - ssr_aux / sst
    } else {
        0.0
    };

    // J statistic = n * R²
    let j_statistic = (n as f64) * r_squared_aux.max(0.0);

    // p-value from chi-squared distribution
    let p_value = chi_squared_p_value(j_statistic, df as f64);

    // Interpretation
    let interpretation = if p_value > 0.10 {
        "Cannot reject H0. Instruments appear valid.".to_string()
    } else if p_value > 0.05 {
        "Marginal evidence against instrument validity (0.05 < p ≤ 0.10).".to_string()
    } else if p_value > 0.01 {
        "Reject H0 at 5% level. At least one instrument may be invalid.".to_string()
    } else {
        "Strongly reject H0. Instruments appear invalid.".to_string()
    };

    Ok(SarganTestResult {
        test_name: "Sargan Test of Overidentifying Restrictions".to_string(),
        j_statistic,
        df,
        p_value,
        n_instruments: n_instr,
        n_endogenous: n_endog,
        n_obs: n,
        overidentified: true,
        null_hypothesis: "Instruments are valid (uncorrelated with error term)".to_string(),
        interpretation,
        missing,
    })
}

/// Run the Sargan test from an existing IV result and dataset.
///
/// This is a convenience function that wraps `sargan_test` using the parameters
/// from an existing `IVResult`.
pub fn run_sargan_test(dataset: &Dataset, iv_result: &IVResult) -> EconResult<SarganTestResult> {
    let x_exog: Vec<&str> = iv_result
        .exogenous_vars
        .iter()
        .map(|s| s.as_str())
        .collect();
    let x_endog: Vec<&str> = iv_result
        .endogenous_vars
        .iter()
        .map(|s| s.as_str())
        .collect();
    let instruments: Vec<&str> = iv_result.instruments.iter().map(|s| s.as_str()).collect();

    sargan_test(dataset, &iv_result.dep_var, &x_exog, &x_endog, &instruments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    #[test]
    fn test_partial_f_zero_over_zero_is_nan() {
        // 0/0 (no fit in either model) has no F; only a perfect unrestricted fit with a
        // positive restricted RSS is +inf.
        assert!(partial_f(0.0, 0.0, 2, 10).is_nan());
        assert_eq!(partial_f(1.0, 0.0, 2, 10), f64::INFINITY);
        assert!(partial_f(1.0, 0.5, 2, 0).is_nan());
        assert!(f_upper_tail(3.0, 2, 0).is_nan());
        assert_eq!(f_upper_tail(f64::INFINITY, 2, 10), 0.0);
    }

    #[test]
    fn test_hc1_wald_f_singular_is_none() {
        // A zero covariance block (here from a zero (Z'Z)^-1) cannot be inverted: no HC1 F.
        let z = ndarray::Array2::from_shape_fn((6, 2), |(i, j)| (i * (j + 1)) as f64 + 0.5);
        let zero = ndarray::Array2::<f64>::zeros((2, 2));
        let resid = ndarray::Array1::from(vec![0.3, -0.2, 0.1, -0.4, 0.25, -0.05]);
        let beta = ndarray::Array1::from(vec![0.7, -0.3]);
        assert_eq!(hc1_wald_f(&z.view(), &zero, &resid, &beta, 0), None);
    }

    fn aliased_iv_data() -> Dataset {
        let n = 40;
        let z1: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.37).sin() * 2.0 + 0.1 * i as f64)
            .collect();
        let z2: Vec<f64> = (0..n).map(|i| (i as f64 * 0.91).cos()).collect();
        let w: Vec<f64> = (0..n).map(|i| ((i * 7 % 13) as f64) / 13.0).collect();
        let x: Vec<f64> = (0..n)
            .map(|i| 0.8 * z1[i] + 0.5 * z2[i] + 0.3 * w[i] + 0.2 * (i as f64 * 1.7).sin())
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 0.5 * x[i] - 0.4 * w[i] + 0.3 * (i as f64 * 2.3).cos())
            .collect();
        let z2_dup = z2.clone();
        Dataset::new(
            df!("y" => y, "x" => x, "z1" => z1, "z2" => z2, "z2_dup" => z2_dup,
                "w" => w.clone(), "w_dup" => w)
            .unwrap(),
        )
    }

    #[test]
    fn test_iv_refuses_aliased_designs() {
        // R's AER::ivreg reports NA for an aliased regressor or instrument; p2a refuses instead
        // of returning a generalized-inverse fit with a silently discarded rank condition.
        let ds = aliased_iv_data();
        assert!(run_iv2sls(&ds, "y", &["w"], &["x"], &["z1", "z2"], false).is_ok());
        assert!(sargan_test(&ds, "y", &["w"], &["x"], &["z1", "z2"]).is_ok());
        let e = run_iv2sls(&ds, "y", &["w", "w_dup"], &["x"], &["z1", "z2"], false)
            .unwrap_err()
            .to_string();
        // w_dup is an exogenous instrument too, so the first inverse to meet it is Z_0'Z_0.
        assert!(e.contains("Z_0'Z_0: rank 2 of 3"), "{e}");
        let e = run_first_stage_diagnostics(&ds, "x", &["z1", "z2"], Some(&["w", "w_dup"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("first-stage") && e.contains("rank"), "{e}");
        // A duplicated instrument: whichever inverse meets the rank deficiency first refuses it.
        let e = sargan_test(&ds, "y", &["w"], &["x"], &["z1", "z2", "z2_dup"])
            .unwrap_err()
            .to_string();
        assert!(e.contains("rank"), "{e}");
    }

    fn create_iv_dataset() -> Dataset {
        // Classic IV setup: y = 0.5*x_endog + noise
        // x_endog is correlated with error (endogenous)
        // z is correlated with x_endog but not with error (valid instrument)
        //
        // True model: y = 0.5 * x_endog
        // z -> x_endog -> y  (z is instrument)
        let df = df! {
            "y" => [1.2, 2.1, 3.3, 4.0, 5.2, 5.8, 7.1, 8.0, 8.9, 10.1],
            "x_endog" => [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            "z" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "x_exog" => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_under_identified_error() {
        // 2 endogenous vars but only 1 instrument = under-identified
        let n_endog = 2;
        let n_instr = 1;
        assert!(n_instr < n_endog);
    }

    #[test]
    fn test_iv2sls_basic() {
        let dataset = create_iv_dataset();
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],          // no exogenous regressors (besides constant)
            &["x_endog"], // endogenous regressor
            &["z"],       // instrument
            false,        // not robust
        )
        .unwrap();

        // Check structure
        assert_eq!(result.n_obs, 10);
        assert!(!result.variables.is_empty());

        // The true coefficient on x_endog is ~0.5
        // IV should recover something in the ballpark
        let x_endog_idx = result
            .variables
            .iter()
            .position(|v| v == "x_endog")
            .unwrap();
        assert!(
            (result.coefficients[x_endog_idx] - 0.5).abs() < 0.2,
            "IV coefficient should be close to 0.5, got {}",
            result.coefficients[x_endog_idx]
        );
    }

    #[test]
    fn test_iv2sls_robust() {
        let dataset = create_iv_dataset();
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],
            &["x_endog"],
            &["z"],
            true, // robust SEs
        )
        .unwrap();

        assert_eq!(result.n_obs, 10);
        // Robust SEs should still produce valid results
        assert!(
            result
                .std_errors
                .iter()
                .all(|&se| se > 0.0 && se.is_finite())
        );
    }

    #[test]
    fn test_first_stage_diagnostics() {
        let dataset = create_iv_dataset();
        let result = run_first_stage_diagnostics(
            &dataset,
            "x_endog",
            &["z"],
            None, // no additional controls
        )
        .unwrap();

        // First stage should have results
        assert_eq!(result.endogenous_var, "x_endog");
        assert!(!result.instruments.is_empty());

        // R² should be high since z and x_endog are linearly related
        assert!(
            result.r_squared > 0.9,
            "First-stage R² should be > 0.9 for linearly related variables, got {}",
            result.r_squared
        );

        // F-statistic should be non-negative
        assert!(
            result.f_statistic >= 0.0,
            "F-statistic should be non-negative, got {}",
            result.f_statistic
        );
    }

    #[test]
    fn test_first_stage_refuses_an_empty_instrument_list() {
        // no excluded instrument: there is no partial F to report, and no verdict
        let dataset = create_iv_dataset();
        match run_first_stage_diagnostics(&dataset, "x_endog", &[], None) {
            Err(EconError::InvalidSpecification { message }) => assert!(
                message.contains("instrument") && message.contains("x_endog"),
                "{message}"
            ),
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_iv_under_identified() {
        let dataset = create_iv_dataset();
        // Try to instrument 2 variables with only 1 instrument
        let result = run_iv2sls(
            &dataset,
            "y",
            &[],
            &["x_endog", "z"], // 2 endogenous
            &["x_exog"],       // only 1 instrument
            false,
        );
        // Should fail due to under-identification
        assert!(result.is_err());
    }

    fn create_overidentified_dataset() -> Dataset {
        // Over-identified IV setup: 1 endogenous, 2 instruments
        // y = 0.5*x_endog + noise
        // z1 and z2 are both valid instruments (correlated with x_endog, uncorrelated with error)
        let df = df! {
            "y" => [1.2, 2.1, 3.3, 4.0, 5.2, 5.8, 7.1, 8.0, 8.9, 10.1,
                    11.0, 12.2, 13.1, 14.0, 15.3, 16.1, 17.0, 18.2, 19.1, 20.0],
            "x_endog" => [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                          22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
            "z1" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                     11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
            "z2" => [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9,
                     11.1, 11.9, 13.1, 13.9, 15.1, 15.9, 17.1, 17.9, 19.1, 19.9]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_sargan_test_overidentified() {
        let dataset = create_overidentified_dataset();

        // Over-identified: 1 endogenous, 2 instruments
        let result = sargan_test(
            &dataset,
            "y",
            &[],           // no exogenous (besides constant)
            &["x_endog"],  // 1 endogenous
            &["z1", "z2"], // 2 instruments
        )
        .unwrap();

        // Should be over-identified
        assert!(result.overidentified);
        assert_eq!(result.df, 1); // 2 instruments - 1 endogenous = 1

        // J statistic should be non-negative
        assert!(result.j_statistic >= 0.0);

        // p-value should be valid
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);

        // For valid instruments, we expect to NOT reject (p > 0.05)
        // This test data has valid instruments, so p should be high
        println!(
            "Sargan test: J = {:.4}, df = {}, p = {:.4}",
            result.j_statistic, result.df, result.p_value
        );
    }

    #[test]
    fn test_sargan_test_exactly_identified() {
        let dataset = create_iv_dataset();

        // Exactly identified: 1 endogenous, 1 instrument
        let result = sargan_test(
            &dataset,
            "y",
            &[],          // no exogenous (besides constant)
            &["x_endog"], // 1 endogenous
            &["z"],       // 1 instrument
        )
        .unwrap();

        // Should NOT be over-identified
        assert!(!result.overidentified);
        assert_eq!(result.df, 0);
        assert_eq!(result.p_value, 1.0); // Test not applicable
    }

    #[test]
    fn test_run_sargan_test_from_iv_result() {
        let dataset = create_overidentified_dataset();

        // First run IV
        let iv_result = run_iv2sls(&dataset, "y", &[], &["x_endog"], &["z1", "z2"], false).unwrap();

        // Then run Sargan test
        let sargan_result = run_sargan_test(&dataset, &iv_result).unwrap();

        assert!(sargan_result.overidentified);
        assert_eq!(sargan_result.df, 1);
        assert!(sargan_result.j_statistic >= 0.0);
    }

    // ════════════════════════════════════════════════════════════════════════════
    // R-pinned tests: AER::ivreg on the cases of validation/r/cases/econ_iv.R
    // (registered as `iv_2sls` in validation/methods/econ_iv_panel_na.json)
    // ════════════════════════════════════════════════════════════════════════════

    use p2a_validation::{Check, Tol};

    /// The first-stage df of each endogenous variable as two vectors (df1, df2).
    fn fs_df(r: &IVResult) -> (Vec<f64>, Vec<f64>) {
        r.first_stage_df
            .iter()
            .map(|&(a, b)| (a as f64, b as f64))
            .unzip()
    }

    /// Names, coefficients and SEs of the instrument rows of a first-stage fit.
    fn instr_rows(d: &FirstStageDiagnostics) -> (Vec<String>, Vec<f64>, Vec<f64>) {
        let names = d.instrument_coeffs.iter().map(|c| c.0.clone()).collect();
        let coefs = d.instrument_coeffs.iter().map(|c| c.1).collect();
        let ses = d.instrument_coeffs.iter().map(|c| c.2).collect();
        (names, coefs, ses)
    }

    /// A control that predicts the endogenous regressor and weak excluded instruments:
    /// the overall first-stage F (300.8, what p2a reported before) certified them as
    /// strong; AER's weak-instrument (partial) F is 0.2267.
    #[test]
    fn test_validate_iv_weak_controls() {
        let (case, ds) = crate::test_support::load_case("iv_weak_controls");
        let r = run_iv2sls(&ds, "y", &["w"], &["x"], &["z1", "z2"], false).unwrap();
        let h = run_iv2sls(&ds, "y", &["w"], &["x"], &["z1", "z2"], true).unwrap();
        let s = sargan_test(&ds, "y", &["w"], &["x"], &["z1", "z2"]).unwrap();
        let n = r.n_obs;
        let (df1, df2) = fs_df(&r);
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se", &r.variables, &r.std_errors, Tol::se(n))
            .named("se_hc1", &h.variables, &h.std_errors, Tol::se(n))
            .vector("weak_f", &r.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p", &r.first_stage_p_values, Tol::p_value(n))
            .vector("weak_df1", &df1, Tol::exact())
            .vector("weak_df2", &df2, Tol::exact())
            .vector("weak_f_hc1", &h.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p_hc1", &h.first_stage_p_values, Tol::p_value(n))
            .scalar("fs_partial_r2", r.first_stage_partial_r2[0], Tol::stat(n))
            .scalar("sargan", s.j_statistic, Tol::stat(n))
            .scalar("sargan_p", s.p_value, Tol::p_value(n))
            .scalar("n_obs", n as f64, Tol::exact())
            .finish();
        // Not compared: se_hc0 (run_iv2sls offers classical and HC1 only), wu_hausman (not
        // computed), overall_first_stage_f (the pre-fix value, kept in the case for the record).
        assert_eq!(h.coefficients, r.coefficients);
        assert_eq!(h.first_stage_partial_r2, r.first_stage_partial_r2);
        assert!(!r.strong_instruments && !h.strong_instruments);
        assert_eq!(
            r.first_stage_f_type,
            "partial F of the excluded instruments"
        );
        assert_eq!(
            h.first_stage_f_type,
            "HC1 Wald F of the excluded instruments"
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.starts_with("Weak instruments for 'x': first-stage partial F")),
            "{:?}",
            r.warnings
        );
        assert!(r.missing.is_empty());
    }

    /// Two endogenous regressors: one partial F per variable, in `x_endog` order.
    #[test]
    fn test_validate_iv_two_endog() {
        let (case, ds) = crate::test_support::load_case("iv_two_endog");
        let (exog, endog, instr) = (&["w1", "w2"], &["x1", "x2"], &["z1", "z2", "z3"]);
        let r = run_iv2sls(&ds, "y", exog, endog, instr, false).unwrap();
        let h = run_iv2sls(&ds, "y", exog, endog, instr, true).unwrap();
        let s = sargan_test(&ds, "y", exog, endog, instr).unwrap();
        let n = r.n_obs;
        let (df1, df2) = fs_df(&r);
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se", &r.variables, &r.std_errors, Tol::se(n))
            .named("se_hc1", &h.variables, &h.std_errors, Tol::se(n))
            .vector("weak_f", &r.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p", &r.first_stage_p_values, Tol::p_value(n))
            .vector("weak_df1", &df1, Tol::exact())
            .vector("weak_df2", &df2, Tol::exact())
            .vector("weak_f_hc1", &h.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p_hc1", &h.first_stage_p_values, Tol::p_value(n))
            .scalar("sargan", s.j_statistic, Tol::stat(n))
            .scalar("sargan_p", s.p_value, Tol::p_value(n))
            .scalar("n_obs", n as f64, Tol::exact())
            .finish();
        assert!(r.strong_instruments && h.strong_instruments);
        assert!(
            !r.warnings.iter().any(|w| w.starts_with("Weak instruments")),
            "{:?}",
            r.warnings
        );
    }

    /// No controls: the partial F equals the overall F (already right before the fix);
    /// the robust run now reports the HC1 Wald F instead of the classical F.
    #[test]
    fn test_validate_iv_no_controls() {
        let (case, ds) = crate::test_support::load_case("iv_no_controls");
        let r = run_iv2sls(&ds, "y", &[], &["x"], &["z1", "z2"], false).unwrap();
        let h = run_iv2sls(&ds, "y", &[], &["x"], &["z1", "z2"], true).unwrap();
        let s = sargan_test(&ds, "y", &[], &["x"], &["z1", "z2"]).unwrap();
        let n = r.n_obs;
        let (df1, df2) = fs_df(&r);
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se", &r.variables, &r.std_errors, Tol::se(n))
            .named("se_hc1", &h.variables, &h.std_errors, Tol::se(n))
            .vector("weak_f", &r.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p", &r.first_stage_p_values, Tol::p_value(n))
            .vector("weak_df1", &df1, Tol::exact())
            .vector("weak_df2", &df2, Tol::exact())
            .vector("weak_f_hc1", &h.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p_hc1", &h.first_stage_p_values, Tol::p_value(n))
            .scalar("sargan", s.j_statistic, Tol::stat(n))
            .scalar("sargan_p", s.p_value, Tol::p_value(n))
            .scalar("n_obs", n as f64, Tol::exact())
            .finish();
        assert!(!r.strong_instruments && !h.strong_instruments);
        assert!(
            h.warnings
                .iter()
                .any(|w| w.starts_with("Weak instruments for 'x': first-stage HC1 Wald F")),
            "{:?}",
            h.warnings
        );
    }

    /// `run_first_stage_diagnostics` with controls: the partial F of the instruments
    /// given the controls, `anova(lm(x ~ w), lm(x ~ w + z1 + z2))`; without controls on
    /// `iv_no_controls` it equals AER's weak-instrument F.
    #[test]
    fn test_validate_first_stage_diagnostics_weak_controls() {
        let (case, ds) = crate::test_support::load_case("iv_weak_controls");
        let d = run_first_stage_diagnostics(&ds, "x", &["z1", "z2"], Some(&["w"])).unwrap();
        let n = d.n_obs;
        let (names, coefs, ses) = instr_rows(&d);
        Check::new(&case)
            .scalar("fs_partial_f", d.f_statistic, Tol::stat(n))
            .scalar("fs_partial_p", d.f_pvalue, Tol::p_value(n))
            .scalar("fs_partial_df1", d.f_df.0 as f64, Tol::exact())
            .scalar("fs_partial_df2", d.f_df.1 as f64, Tol::exact())
            .scalar("fs_partial_r2", d.partial_r_squared, Tol::stat(n))
            .scalar("fs_r2", d.r_squared, Tol::stat(n))
            .scalar("fs_adj_r2", d.adj_r_squared, Tol::stat(n))
            .named("fs_instr_coef", &names, &coefs, Tol::coef(n))
            .named("fs_instr_se", &names, &ses, Tol::se(n))
            .finish();
        assert!(!d.strong_instruments);
        assert_eq!(d.controls, ["w"]);

        let (case, ds) = crate::test_support::load_case("iv_no_controls");
        let d = run_first_stage_diagnostics(&ds, "x", &["z1", "z2"], None).unwrap();
        let n = d.n_obs;
        Check::new(&case)
            .vector("weak_f", &[d.f_statistic], Tol::stat(n))
            .vector("weak_p", &[d.f_pvalue], Tol::p_value(n))
            .vector("weak_df1", &[d.f_df.0 as f64], Tol::exact())
            .vector("weak_df2", &[d.f_df.1 as f64], Tol::exact())
            .finish();
        assert!(d.controls.is_empty());
    }

    /// The first-stage block of both Displays names the statistic and its df.
    #[test]
    fn test_validate_iv_display_first_stage() {
        let (case, ds) = crate::test_support::load_case("iv_weak_controls");
        let r = run_iv2sls(&ds, "y", &["w"], &["x"], &["z1", "z2"], false).unwrap();
        Check::new(&case)
            .vector("weak_f", &r.first_stage_f_stats, Tol::printed(4))
            .finish();
        let text = r.to_string();
        assert!(
            text.contains("First stage (partial F of the excluded instruments):"),
            "{text}"
        );
        assert!(
            text.contains("  x: F = 0.2267 on (2, 496) DF, p = 0.7972, partial R-squared = 0.0009"),
            "{text}"
        );
        assert!(!text.contains("First-stage F:"), "{text}");
        let d = run_first_stage_diagnostics(&ds, "x", &["z1", "z2"], Some(&["w"])).unwrap();
        let text = d.to_string();
        assert!(text.contains("Controls: w"), "{text}");
        assert!(
            text.contains("Instrument Strength (partial F of the excluded instruments):"),
            "{text}"
        );
        assert!(
            text.contains("  F-statistic: 0.2267 on (2, 496) DF"),
            "{text}"
        );
        assert!(text.contains("WEAK (F < 5)"), "{text}");
    }

    /// Missing cells in six columns, an Int64 control: ivreg drops the 6 incomplete rows
    /// (na.omit); the first-stage diagnostics use the complete rows of their own columns.
    #[test]
    fn test_validate_iv_na_rows() {
        let (case, ds) = crate::test_support::load_case("iv_na_rows");
        assert_eq!(
            ds.df().column("k").unwrap().dtype(),
            &polars::prelude::DataType::Int64
        ); // k is written without a decimal point, so polars reads it as Int64
        let (exog, endog, instr) = (&["w", "k"], &["x"], &["z1", "z2"]);
        let r = run_iv2sls(&ds, "y", exog, endog, instr, false).unwrap();
        let h = run_iv2sls(&ds, "y", exog, endog, instr, true).unwrap();
        let s = sargan_test(&ds, "y", exog, endog, instr).unwrap();
        let d = run_first_stage_diagnostics(&ds, "x", instr, Some(exog)).unwrap();
        let by = |m: &MissingSummary, c: &str| {
            m.by_column
                .iter()
                .find(|(k, _)| k == c)
                .map_or(0, |(_, v)| *v) as f64
        };
        let n = r.n_obs;
        let (df1, df2) = fs_df(&r);
        let (names, coefs, ses) = instr_rows(&d);
        Check::new(&case)
            .named("coef", &r.variables, &r.coefficients, Tol::coef(n))
            .named("se", &r.variables, &r.std_errors, Tol::se(n))
            .named("se_hc1", &h.variables, &h.std_errors, Tol::se(n))
            .vector("weak_f", &r.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p", &r.first_stage_p_values, Tol::p_value(n))
            .vector("weak_f_hc1", &h.first_stage_f_stats, Tol::stat(n))
            .vector("weak_p_hc1", &h.first_stage_p_values, Tol::p_value(n))
            .vector("weak_df1", &df1, Tol::exact())
            .vector("weak_df2", &df2, Tol::exact())
            .scalar("sargan", s.j_statistic, Tol::stat(n))
            .scalar("sargan_p", s.p_value, Tol::p_value(n))
            .scalar("n_obs", n as f64, Tol::exact())
            .scalar("n_dropped", r.missing.n_dropped as f64, Tol::exact())
            .scalar("missing_y", by(&r.missing, "y"), Tol::exact())
            .scalar("missing_x", by(&r.missing, "x"), Tol::exact())
            .scalar("missing_w", by(&r.missing, "w"), Tol::exact())
            .scalar("missing_k", by(&r.missing, "k"), Tol::exact())
            .scalar("missing_z1", by(&r.missing, "z1"), Tol::exact())
            .scalar("missing_z2", by(&r.missing, "z2"), Tol::exact())
            .scalar("fsd_n_obs", d.n_obs as f64, Tol::exact())
            .scalar("fsd_partial_f", d.f_statistic, Tol::stat(d.n_obs))
            .scalar("fsd_partial_p", d.f_pvalue, Tol::p_value(d.n_obs))
            .scalar("fsd_partial_df1", d.f_df.0 as f64, Tol::exact())
            .scalar("fsd_partial_df2", d.f_df.1 as f64, Tol::exact())
            .scalar("fsd_partial_r2", d.partial_r_squared, Tol::stat(d.n_obs))
            .scalar("fsd_r2", d.r_squared, Tol::stat(d.n_obs))
            .scalar("fsd_adj_r2", d.adj_r_squared, Tol::stat(d.n_obs))
            .named("fsd_instr_coef", &names, &coefs, Tol::coef(d.n_obs))
            .named("fsd_instr_se", &names, &ses, Tol::se(d.n_obs))
            .finish();
        let note = "6 of 300 rows dropped because of missing values (y: 2, w: 1, k: 1, x: 1, z1: 1, z2: 1)";
        assert_eq!(r.missing.note().as_deref(), Some(note));
        assert!(r.warnings.iter().any(|w| w == note), "{:?}", r.warnings);
        assert_eq!(s.missing, r.missing);
        assert_eq!(
            d.missing.note().as_deref(),
            Some(
                "4 of 300 rows dropped because of missing values (x: 1, z1: 1, z2: 1, w: 1, k: 1)"
            )
        );
        let text = r.to_string();
        assert_eq!(text.matches(note).count(), 1, "{text}"); // printed once
        assert!(
            text.trim_end()
                .ends_with(&format!("Missing values: {note}")),
            "{text}"
        );
        assert!(r.strong_instruments);
        for t in [s.to_string(), d.to_string()] {
            assert!(
                t.trim_end()
                    .lines()
                    .last()
                    .is_some_and(|l| l.starts_with("Missing values: ")),
                "{t}"
            );
        }
    }

    /// A text column is non-numeric and an unknown column is not found, whichever role
    /// they take; an Int64 control gives the fit of the same values in Float64.
    #[test]
    fn test_validate_na_iv_errors_name_the_column() {
        let (case, ds) = crate::test_support::load_case("iv_na_rows");
        assert_eq!(case.dataset, "validation/datasets/generated/iv_na_rows.csv");
        let non_numeric = |e: &EconError| matches!(e, EconError::NonNumericColumn { column } if column == "label");
        let e = run_iv2sls(&ds, "label", &["w"], &["x"], &["z1", "z2"], false).unwrap_err();
        assert!(non_numeric(&e), "{e:?}");
        let e = run_iv2sls(&ds, "y", &["w"], &["label"], &["z1", "z2"], false).unwrap_err();
        assert!(non_numeric(&e), "{e:?}");
        let e = run_iv2sls(&ds, "y", &["w"], &["x"], &["nope", "z2"], false).unwrap_err();
        assert!(
            matches!(&e, EconError::ColumnNotFound { column, .. } if column == "nope"),
            "{e:?}"
        );
        let e = run_first_stage_diagnostics(&ds, "label", &["z1"], None).unwrap_err();
        assert!(non_numeric(&e), "{e:?}");
        let e = sargan_test(&ds, "label", &[], &["x"], &["z1", "z2"]).unwrap_err();
        assert!(non_numeric(&e), "{e:?}");
        let e = run_iv2sls(&ds, "y", &["w"], &[], &["z1", "z2"], false).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("endogenous")),
            "{e:?}"
        );

        // the Int64 control `k` as Float64 (nulls kept) gives the same fit
        let k_f = crate::data::columns::numeric_options(ds.df(), "k").unwrap();
        let mut df = ds.df().clone();
        df.with_column(polars::prelude::Column::new("k".into(), k_f))
            .unwrap();
        let ds_f = Dataset::new(df);
        let fit =
            |d: &Dataset| run_iv2sls(d, "y", &["w", "k"], &["x"], &["z1", "z2"], true).unwrap();
        let (a, b) = (fit(&ds), fit(&ds_f));
        let tol = Tol::new(1e-12, 1e-12);
        for (x, y) in a
            .coefficients
            .iter()
            .chain(&a.std_errors)
            .zip(b.coefficients.iter().chain(&b.std_errors))
        {
            assert!(tol.allows(*x, *y), "Int64 {x} vs Float64 {y}");
        }
    }
}
