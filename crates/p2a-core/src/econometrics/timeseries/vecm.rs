//! Vector error correction models by the Johansen procedure, as R's `urca::ca.jo` and
//! `urca::cajorls` ([`run_vecm_with`], [`run_vecm()`], [`johansen_vecm`]).
//!
//! # Model
//!
//! A VAR(K) of k variables in levels, `y_t = ν + A_1 y_{t-1} + ... + A_K y_{t-K} + e_t`, in the
//! transitory error-correction form (`urca::ca.jo(spec = "transitory")`, the form of `vars` and
//! Lütkepohl): `Δy_t = Π y_{t-1} + Γ_1 Δy_{t-1} + ... + Γ_{K-1} Δy_{t-K+1} + μ + e_t`. Cointegration of
//! rank r means `Π = α β'` with k × r loadings α and cointegrating vectors β.
//!
//! # Johansen procedure (`urca::ca.jo`)
//!
//! 1. `Z0 = Δy_t`; `Z1` = the K - 1 lagged differences, after an unrestricted constant when `ecdet`
//!    is `none` or `trend`; `ZK = y_{t-1}`, plus the restricted constant (`const`) or trend (`trend`).
//!    T = N - K rows.
//! 2. `R0`, `RK`: residuals of `Z0` and `ZK` on `Z1` (pivoted QR); `S00 = R0'R0/T`,
//!    `S01 = R0'RK/T`, `S11 = RK'RK/T`.
//! 3. The eigenvalues `λ_1 >= λ_2 >= ...` of `S11^-1 S10 S00^-1 S01`, symmetrised through the Cholesky
//!    factors of `S11` and `S00`; the eigenvectors `V`, normalised on the first variable;
//!    `W = S01 V (V'S11V)^-1` and `Π = S01 S11^-1`.
//! 4. Trace statistic for H0 rank <= r: `-T Σ_{i > r} ln(1 - λ_i)`; maximal-eigenvalue statistic
//!    `-T ln(1 - λ_{r+1})`; critical values from the Osterwald-Lenum (1992) tables urca embeds
//!    (10%, 5%, 1%; at most [`VECM_MAX_VARIABLES`] variables).
//!
//! # Restricted VECM (`urca::cajorls`)
//!
//! For rank r (configured, else the first r the trace test does not reject at 5%):
//! `β = V_r (V_r[1:r, 1:r])^-1`; α, Γ and the unrestricted constant by OLS of `Z0` on `[ZK β, Z1]`.
//!
//! Missing values are refused (urca drops incomplete rows with `na.omit`, joining non-adjacent periods).
//! Not supported: seasonal dummies (`season`), exogenous dummies (`dumvar`), urca's long-run form
//! (`spec = "longrun"`, which changes only Γ), urca's `@GAMMA` slot and the implied levels VAR
//! (`vars::vec2var`).
//!
//! # References
//!
//! - Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics
//!   and Control*, 12(2-3), 231-254. <https://doi.org/10.1016/0165-1889(88)90041-3>
//! - Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector
//!   autoregressive models. *Econometrica*, 59(6), 1551-1580. <https://doi.org/10.2307/2938278>
//! - Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic distribution of the maximum
//!   likelihood cointegration rank test statistics. *Oxford Bulletin of Economics and Statistics*,
//!   54(3), 461-472. <https://doi.org/10.1111/j.1468-0084.1992.tb00013.x>
//! - Pfaff, B. (2008). *Analysis of Integrated and Cointegrated Time Series with R* (2nd ed.). Springer.
//!
//! R equivalent: `urca::ca.jo()`, `urca::cajorls()`.

use std::fmt;

use ndarray::{Array1, Array2, ArrayView2, s};
use serde::{Deserialize, Serialize};

use super::common::{array2_to_vec, df_to_array2};
use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::{
    CONDITION_THRESHOLD, cholesky, condition_number, eig_symmetric, matrix_inverse,
};
use crate::linalg::solve::{LM_QR_TOL, Qr, SingularPolicy, sym_inverse};
use crate::options::Choices;
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};

/// Most variables the Osterwald-Lenum tables in `urca::ca.jo` cover (rows k - r = 1..=11).
pub const VECM_MAX_VARIABLES: usize = 11;
/// Largest accepted squared canonical correlation: at 1, ln(1 - λ) is infinite.
const MAX_EIGENVALUE: f64 = 1.0 - 1e-10;

/// Deterministic terms of the Johansen procedure (`urca::ca.jo(ecdet = ...)`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum VecmDeterministic {
    /// `"none"`: an unrestricted constant in the VECM (a linear trend in the levels) and nothing
    /// in the cointegration relations (urca's default).
    #[default]
    None,
    /// `"const"`: a constant restricted to the cointegration relations (the last row of beta), no
    /// unrestricted constant.
    Const,
    /// `"trend"`: a linear trend restricted to the cointegration relations, plus an unrestricted constant.
    Trend,
}

crate::option_enum!(
    VecmDeterministic,
    Choices {
        option: "ecdet",
        values: &[
            (&["none"], VecmDeterministic::None),
            (&["const", "constant"], VecmDeterministic::Const),
            (&["trend"], VecmDeterministic::Trend),
        ],
    }
);

impl VecmDeterministic {
    /// urca's `@model` text for the case.
    pub fn description(self) -> &'static str {
        match self {
            Self::None => "with linear trend",
            Self::Const => "without linear trend and constant in cointegration",
            Self::Trend => "with linear trend in cointegration",
        }
    }
}

/// Configuration of [`run_vecm_with`] and [`johansen_vecm`].
///
/// Only the transitory error-correction form is estimated (level term `y_{t-1}`, as `vars` and
/// Lütkepohl write it; `urca::ca.jo(spec = "transitory")`).
///
/// ```
/// use p2a_core::econometrics::timeseries::vecm::{VecmConfig, VecmDeterministic};
/// let mut config = VecmConfig::default(); // K = 2, ecdet = none, rank by the trace test
/// config.ecdet = "const".parse::<VecmDeterministic>().unwrap();
/// config.rank = Some(1);
/// assert_eq!((config.lag_order, config.ecdet), (2, VecmDeterministic::Const));
/// assert!("drift".parse::<VecmDeterministic>().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VecmConfig {
    /// Lag order K of the VAR in levels (urca's `K`); the VECM has K - 1 lagged differences. At least 2.
    pub lag_order: usize,
    /// Deterministic terms (urca's `ecdet`).
    pub ecdet: VecmDeterministic,
    /// Cointegration rank r of the restricted estimates, 1..=k-1 (`cajorls`'s `r`);
    /// `None`: the first r that the trace test does not reject at the 5% level.
    pub rank: Option<usize>,
}

impl Default for VecmConfig {
    fn default() -> Self {
        Self {
            lag_order: 2,
            ecdet: VecmDeterministic::None,
            rank: None,
        }
    }
}

// Osterwald-Lenum (1992) critical values exactly as urca::ca.jo embeds them (cv.none, cv.const,
// cv.trend; [, , 1] = maximal eigenvalue, [, , 2] = trace). Row index m - 1 for m = k - r = 1..=11
// non-cointegrating directions; columns 10%, 5%, 1%.
#[rustfmt::skip]
const NONE_EIGEN: [[f64; 3]; VECM_MAX_VARIABLES] = [
    [6.50, 8.18, 11.65], [12.91, 14.90, 19.19], [18.90, 21.07, 25.75], [24.78, 27.14, 32.14],
    [30.84, 33.32, 38.78], [36.25, 39.43, 44.59], [42.06, 44.91, 51.30], [48.43, 51.07, 57.07],
    [54.01, 57.00, 63.37], [59.00, 62.42, 68.61], [65.07, 68.27, 74.36],
];
#[rustfmt::skip]
const NONE_TRACE: [[f64; 3]; VECM_MAX_VARIABLES] = [
    [6.50, 8.18, 11.65], [15.66, 17.95, 23.52], [28.71, 31.52, 37.22], [45.23, 48.28, 55.43],
    [66.49, 70.60, 78.87], [85.18, 90.39, 104.20], [118.99, 124.25, 136.06], [151.38, 157.11, 168.92],
    [186.54, 192.84, 204.79], [226.34, 232.49, 246.27], [269.53, 277.39, 292.65],
];
#[rustfmt::skip]
const CONST_EIGEN: [[f64; 3]; VECM_MAX_VARIABLES] = [
    [7.52, 9.24, 12.97], [13.75, 15.67, 20.20], [19.77, 22.00, 26.81], [25.56, 28.14, 33.24],
    [31.66, 34.40, 39.79], [37.45, 40.30, 46.82], [43.25, 46.45, 51.91], [48.91, 52.00, 57.95],
    [54.35, 57.42, 63.71], [60.25, 63.57, 69.94], [66.02, 69.74, 76.63],
];
#[rustfmt::skip]
const CONST_TRACE: [[f64; 3]; VECM_MAX_VARIABLES] = [
    [7.52, 9.24, 12.97], [17.85, 19.96, 24.60], [32.00, 34.91, 41.07], [49.65, 53.12, 60.16],
    [71.86, 76.07, 84.45], [97.18, 102.14, 111.01], [126.58, 131.70, 143.09], [159.48, 165.58, 177.20],
    [196.37, 202.92, 215.74], [236.54, 244.15, 257.68], [282.45, 291.40, 307.64],
];
#[rustfmt::skip]
const TREND_EIGEN: [[f64; 3]; VECM_MAX_VARIABLES] = [
    [10.49, 12.25, 16.26], [16.85, 18.96, 23.65], [23.11, 25.54, 30.34], [29.12, 31.46, 36.65],
    [34.75, 37.52, 42.36], [40.91, 43.97, 49.51], [46.32, 49.42, 54.71], [52.16, 55.50, 62.46],
    [57.87, 61.29, 67.88], [63.18, 66.23, 73.73], [69.26, 72.72, 79.23],
];
#[rustfmt::skip]
const TREND_TRACE: [[f64; 3]; VECM_MAX_VARIABLES] = [
    [10.49, 12.25, 16.26], [22.76, 25.32, 30.45], [39.06, 42.44, 48.45], [59.14, 62.99, 70.05],
    [83.20, 87.31, 96.58], [110.42, 114.90, 124.75], [141.01, 146.76, 158.49], [176.67, 182.82, 196.08],
    [215.17, 222.21, 234.41], [256.72, 263.42, 279.07], [303.13, 310.81, 327.45],
];

/// A table of critical values, one row per number of non-cointegrating directions.
type CvTable = [[f64; 3]; VECM_MAX_VARIABLES];

/// (trace, maximal eigenvalue) tables for the case.
fn critical_value_tables(ecdet: VecmDeterministic) -> (&'static CvTable, &'static CvTable) {
    match ecdet {
        VecmDeterministic::None => (&NONE_TRACE, &NONE_EIGEN),
        VecmDeterministic::Const => (&CONST_TRACE, &CONST_EIGEN),
        VecmDeterministic::Trend => (&TREND_TRACE, &TREND_EIGEN),
    }
}

/// Result of the Johansen procedure and the restricted VECM (urca::ca.jo, urca::cajorls).
///
/// Vectors indexed by r (`trace_stat`, `trace_cv`, `eigen_stat`, `eigen_cv`) run from H0: rank = 0
/// (index 0) to H0: rank <= k - 1; urca stores them in the reverse order. Matrices are `Vec`s of rows.
/// When `rank` is 0 or k (possible only when the trace test chose it), the restricted estimates
/// (`beta`, `alpha`, `gamma`, `constant`) are empty and `warnings` says why.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecmResult {
    /// Rank of the restricted estimates: the configured rank, else `selected_rank`.
    pub rank: usize,
    /// Lag order K of the levels VAR (urca's `K`).
    pub lags: usize,
    /// Number of variables k.
    pub n_vars: usize,
    /// Observations used, T = N - K.
    pub n_obs: usize,
    /// Eigenvalues (squared canonical correlations) in decreasing order: k of them for `ecdet = none`,
    /// k + 1 otherwise (the last is structurally zero).
    pub eigenvalues: Vec<f64>,
    /// Cointegrating vectors (`cajorls`): rows `beta_rows`, columns ect1..ect_r; the top r x r block is I.
    pub beta: Vec<Vec<f64>>,
    /// Adjustment coefficients: k x r.
    pub alpha: Vec<Vec<f64>>,
    /// Short-run coefficients Γ_1..Γ_{K-1}: k x k(K-1), columns `<var>.dl1` for every variable, then `.dl2`, ...
    pub gamma: Vec<Vec<f64>>,
    /// Trace statistics for H0 rank <= r, r = 0..k-1 (equal to `trace_stat`; kept for existing callers).
    pub trace_stats: Vec<f64>,
    /// 5% critical values of `trace_stats` (the middle column of `trace_cv`).
    pub trace_crit_values: Vec<f64>,
    /// Variable names, in the order of the columns.
    #[serde(default)]
    pub variables: Vec<String>,
    /// Deterministic terms estimated (urca's `ecdet`).
    #[serde(default)]
    pub deterministic: VecmDeterministic,
    /// Row names of `beta` and `v_normalised`: `<var>.l1`, then `constant` (const) or `trend.l1` (trend).
    #[serde(default)]
    pub beta_rows: Vec<String>,
    /// Trace statistics `-T Σ_{i > r} ln(1 - λ_i)`, index r = H0 rank <= r.
    #[serde(default)]
    pub trace_stat: Vec<f64>,
    /// Critical values of `trace_stat`: [10%, 5%, 1%] (Osterwald-Lenum, as urca).
    #[serde(default)]
    pub trace_cv: Vec<[f64; 3]>,
    /// Maximal-eigenvalue statistics `-T ln(1 - λ_{r+1})`, index r = H0 rank = r against r + 1.
    #[serde(default)]
    pub eigen_stat: Vec<f64>,
    /// Critical values of `eigen_stat`: [10%, 5%, 1%].
    #[serde(default)]
    pub eigen_cv: Vec<[f64; 3]>,
    /// Rank chosen by the trace test at 5%: the first r with `trace_stat[r] <= trace_cv[r][1]`, else k.
    #[serde(default)]
    pub selected_rank: usize,
    /// Eigenvectors normalised on the first variable (urca `@V`): rows `beta_rows`, one column per eigenvalue.
    #[serde(default)]
    pub v_normalised: Vec<Vec<f64>>,
    /// Weights `S01 V (V'S11V)^-1` (urca `@W`): k x p1.
    #[serde(default)]
    pub w_loadings: Vec<Vec<f64>>,
    /// Unrestricted `S01 S11^-1` (urca `@PI`): k x p1.
    #[serde(default)]
    pub pi: Vec<Vec<f64>>,
    /// Names of the short-run regressors: `constant` (ecdet none or trend), then `<var>.dl<i>`.
    #[serde(default)]
    pub short_run_names: Vec<String>,
    /// Unrestricted VECM constant (cajorls row `constant`); empty for `ecdet = const`.
    #[serde(default)]
    pub constant: Vec<f64>,
    /// Notes on what was not estimated or how eigenvectors were normalised.
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ResultNotes for VecmResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for VecmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Johansen procedure and VECM (urca::ca.jo, urca::cajorls)"
        )?;
        writeln!(f, "{}", "=".repeat(72))?;
        writeln!(f, "Variables: {}", self.variables.join(", "))?;
        writeln!(
            f,
            "Deterministic terms: ecdet = {} ({})",
            self.deterministic,
            self.deterministic.description()
        )?;
        writeln!(
            f,
            "Form: transitory (level term y_{{t-1}}); lag order of the levels VAR K = {}",
            self.lags
        )?;
        writeln!(
            f,
            "Observations used: {} (of {})",
            self.n_obs,
            self.n_obs + self.lags
        )?;
        write!(f, "\nEigenvalues (lambda):")?;
        for (i, ev) in self.eigenvalues.iter().enumerate() {
            // The structural zero of a restricted constant or trend prints as 0.
            let shown = if ev.abs() < 5e-13 { 0.0 } else { *ev };
            write!(f, "{}{shown:.6}", if i == 0 { " " } else { ", " })?;
        }
        writeln!(f, "\n")?;
        for (title, stats, cvs) in [
            ("Trace test", &self.trace_stat, &self.trace_cv),
            ("Maximal eigenvalue test", &self.eigen_stat, &self.eigen_cv),
        ] {
            writeln!(
                f,
                "{title:<24} {:>10} {:>8} {:>8} {:>8}",
                "statistic", "10%", "5%", "1%"
            )?;
            for r in (0..self.n_vars).rev() {
                // urca's row labels, in urca's order
                let label = if r == 0 {
                    "r = 0  |".to_string()
                } else {
                    format!("r <= {r} |")
                };
                let cv = cvs.get(r).copied().unwrap_or([f64::NAN; 3]);
                let stat = stats.get(r).copied().unwrap_or(f64::NAN);
                writeln!(
                    f,
                    "  {label:<22} {stat:>10.4} {:>8.2} {:>8.2} {:>8.2}",
                    cv[0], cv[1], cv[2]
                )?;
            }
        }
        writeln!(
            f,
            "Rank chosen by the trace test at 5%: {}",
            self.selected_rank
        )?;
        let ncol = self.v_normalised.first().map_or(0, Vec::len);
        let vec_names: Vec<String> = (1..=ncol).map(|i| format!("[{i}]")).collect();
        let eq_names: Vec<String> = self.variables.iter().map(|v| format!("{v}.d")).collect();
        writeln!(f, "\nEigenvectors, normalised on the first variable (V):")?;
        write_matrix(f, &self.v_normalised, &self.beta_rows, &vec_names, 6)?;
        writeln!(f, "Weights (W):")?;
        write_matrix(f, &self.w_loadings, &eq_names, &vec_names, 6)?;
        if self.beta.is_empty() {
            writeln!(f, "\nRestricted VECM: not estimated (rank {})", self.rank)?;
        } else {
            let ect: Vec<String> = (1..=self.rank).map(|i| format!("ect{i}")).collect();
            writeln!(
                f,
                "\nRestricted VECM with rank r = {} (urca::cajorls)",
                self.rank
            )?;
            writeln!(f, "Cointegrating vectors (beta):")?;
            write_matrix(f, &self.beta, &self.beta_rows, &ect, 6)?;
            writeln!(f, "Adjustment coefficients (alpha):")?;
            write_matrix(f, &self.alpha, &eq_names, &ect, 6)?;
            if !self.constant.is_empty() {
                let rows: Vec<Vec<f64>> = self.constant.iter().map(|c| vec![*c]).collect();
                writeln!(f, "Unrestricted constant:")?;
                write_matrix(f, &rows, &eq_names, &["constant".to_string()], 6)?;
            }
            let dl: Vec<String> = self
                .short_run_names
                .iter()
                .filter(|n| *n != "constant")
                .cloned()
                .collect();
            writeln!(f, "Short-run coefficients (Gamma):")?;
            write_matrix(f, &self.gamma, &eq_names, &dl, 6)?;
        }
        write!(f, "{}", notes_text(self))
    }
}

/// A labelled matrix; names are padded, never cut.
fn write_matrix(
    f: &mut fmt::Formatter<'_>,
    rows: &[Vec<f64>],
    row_names: &[String],
    col_names: &[String],
    decimals: usize,
) -> fmt::Result {
    let width = 12usize.max(decimals + 6);
    write!(f, "  {:<14}", "")?;
    for c in col_names {
        write!(f, " {c:>width$}")?;
    }
    writeln!(f)?;
    for (name, row) in row_names.iter().zip(rows) {
        write!(f, "  {name:<14}")?;
        for x in row {
            write!(f, " {x:>width$.decimals$}")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

/// Run the Johansen procedure and VECM on `columns` of `dataset` (see the module docs).
pub fn run_vecm_with(
    dataset: &Dataset,
    columns: &[&str],
    config: &VecmConfig,
) -> EconResult<VecmResult> {
    if let Some(dup) = columns
        .iter()
        .enumerate()
        .find(|(i, c)| columns[..*i].contains(c))
        .map(|(_, c)| *c)
    {
        return Err(EconError::InvalidSpecification {
            message: format!("VECM columns must be distinct; '{dup}' is given twice"),
        });
    }
    let (data, names) = df_to_array2(dataset.df(), columns)?;
    johansen_vecm(data.view(), &names, config)
}

/// Run a Vector Error Correction Model by the Johansen procedure.
///
/// The same as [`run_vecm_with`] with `lag_order = lags`, `rank = Some(rank)` and `ecdet = none`:
/// `urca::ca.jo(x, ecdet = "none", K = lags, spec = "transitory")`, then `cajorls(r = rank)`.
/// `lags` is the lag order of the levels VAR (at least 2) and `rank` the cointegration rank (1..=k-1).
pub fn run_vecm(
    dataset: &Dataset,
    columns: &[&str],
    lags: usize,
    rank: usize,
) -> EconResult<VecmResult> {
    let config = VecmConfig {
        lag_order: lags,
        rank: Some(rank),
        ..VecmConfig::default()
    };
    run_vecm_with(dataset, columns, &config)
}

/// The Johansen procedure on a data matrix (rows = time, columns = variables named `names`).
pub fn johansen_vecm(
    y: ArrayView2<'_, f64>,
    names: &[String],
    config: &VecmConfig,
) -> EconResult<VecmResult> {
    let (n, k) = y.dim();
    let invalid = |message: String| EconError::InvalidSpecification { message };
    if names.len() != k {
        return Err(invalid(format!(
            "VECM: {} variable names for {k} columns",
            names.len()
        )));
    }
    if k < 2 {
        return Err(invalid(format!(
            "a VECM needs at least 2 variables; got {k}"
        )));
    }
    if k > VECM_MAX_VARIABLES {
        return Err(invalid(format!(
            "the Johansen critical values (Osterwald-Lenum 1992, as embedded in urca::ca.jo) cover at most \
             {VECM_MAX_VARIABLES} variables; got {k}"
        )));
    }
    if let Some(dup) = names
        .iter()
        .enumerate()
        .find(|(i, a)| names[..*i].contains(a))
        .map(|(_, a)| a)
    {
        return Err(invalid(format!(
            "VECM columns must be distinct; '{dup}' is given twice"
        )));
    }
    let big_k = config.lag_order;
    if big_k < 2 {
        return Err(invalid(format!(
            "lag_order (K, the lag order of the VAR in levels) must be at least 2, as urca::ca.jo requires; got {big_k}"
        )));
    }
    if let Some(r) = config.rank.filter(|r| !(1..k).contains(r)) {
        return Err(invalid(format!(
            "rank must be between 1 and k - 1 = {} (urca::cajorls); got {r}",
            k - 1
        )));
    }
    if let Some(((i, j), v)) = y.indexed_iter().find(|(_, v)| !v.is_finite()) {
        return Err(invalid(format!(
            "VECM data must be finite; column '{}' row {} is {v}",
            names[j],
            i + 1
        )));
    }

    let ecdet = config.ecdet;
    let has_const = ecdet != VecmDeterministic::Const; // unrestricted constant in Z1
    let p1 = k + usize::from(ecdet != VecmDeterministic::None); // columns of ZK
    // N >= K + m + k + p1 keeps T - m >= k + p1, so the squared canonical correlations are not 1 by
    // construction (urca's weaker degrees-of-freedom rule admits such samples and returns NaN statistics).
    let n_min = k
        .checked_mul(big_k - 1)
        .and_then(|lagged| lagged.checked_add(usize::from(has_const)))
        .and_then(|m| big_k.checked_add(m)?.checked_add(k)?.checked_add(p1));
    if !n_min.is_some_and(|required| n >= required) {
        return Err(EconError::InsufficientData {
            required: n_min.unwrap_or(usize::MAX),
            provided: n,
            context: format!(
                "Johansen VECM with k = {k} variables, K = {big_k} and ecdet = {ecdet}: N must be at least K + \
                 (short-run regressors) + k + (level terms), or the squared canonical correlations are 1 by construction"
            ),
        });
    }
    // n_min fits in usize and is at most n, so none of these sizes overflows.
    let m = k * (big_k - 1) + usize::from(has_const);
    let t = n - big_k;

    let mut beta_rows: Vec<String> = names.iter().map(|v| format!("{v}.l1")).collect();
    match ecdet {
        VecmDeterministic::None => {}
        VecmDeterministic::Const => beta_rows.push("constant".to_string()),
        VecmDeterministic::Trend => beta_rows.push("trend.l1".to_string()),
    }
    let mut short_run_names: Vec<String> = resource::try_with_capacity(m, "VECM regressor names")?;
    if has_const {
        short_run_names.push("constant".to_string());
    }
    for i in 1..big_k {
        for v in names {
            short_run_names.push(format!("{v}.dl{i}"));
        }
    }

    // Z0 = Δy_t, Z1 = [1,] Δy_{t-1..t-K+1}, ZK = y_{t-1} [, 1 | trend], t = K..N-1 (0-based).
    let mut z0 = resource::try_zeros2(t, k, "VECM differences")?;
    let mut z1 = resource::try_zeros2(t, m, "VECM short-run regressors")?;
    let mut zk = resource::try_zeros2(t, p1, "VECM level terms")?;
    let off = usize::from(has_const);
    for j in 0..t {
        let s = j + big_k;
        for v in 0..k {
            z0[[j, v]] = y[[s, v]] - y[[s - 1, v]];
            zk[[j, v]] = y[[s - 1, v]];
        }
        if has_const {
            z1[[j, 0]] = 1.0;
        }
        for i in 1..big_k {
            for v in 0..k {
                z1[[j, off + (i - 1) * k + v]] = y[[s - i, v]] - y[[s - i - 1, v]];
            }
        }
        match ecdet {
            VecmDeterministic::None => {}
            VecmDeterministic::Const => zk[[j, k]] = 1.0,
            // urca's trend is 1..N; the level term is at 0-based time s - 1, i.e. trend value s.
            VecmDeterministic::Trend => zk[[j, k]] = s as f64,
        }
    }
    resource::checkpoint()?;

    let qr1 = Qr::decompose(z1.view(), LM_QR_TOL)?;
    if qr1.rank() < m {
        let aliased: Vec<&str> = qr1
            .aliased()
            .iter()
            .map(|&c| short_run_names[c].as_str())
            .collect();
        return Err(EconError::SingularMatrix {
            context: format!(
                "Johansen VECM: the short-run regressors are collinear (rank {} of {m}; aliased: {})",
                qr1.rank(),
                aliased.join(", ")
            ),
            suggestion: "drop constant or duplicated series, or lower lag_order".to_string(),
        });
    }
    let r0 = residuals(&qr1, &z1, &z0)?;
    let rk = residuals(&qr1, &z1, &zk)?;
    let tf = t as f64;
    let s00 = symmetrize(&(r0.t().dot(&r0) / tf));
    let s0k = r0.t().dot(&rk) / tf;
    let skk = symmetrize(&(rk.t().dot(&rk) / tf));
    sym_inverse(
        s00.view(),
        SingularPolicy::Error,
        "S00 (moments of the concentrated differences)",
    )?;
    sym_inverse(
        skk.view(),
        SingularPolicy::Error,
        "S11 (moments of the concentrated level terms)",
    )?;
    let l = cholesky(&skk.view())?;
    let g = cholesky(&s00.view())?;

    // A = G^-1 S01 L^-T (k x p1); A'A = L^-1 S10 S00^-1 S01 L^-T has the eigenvalues of S11^-1 S10 S00^-1 S01.
    let a = solve_lower(&g, &s0k);
    let a = solve_lower(&l, &a.t().to_owned()).t().to_owned();
    let (vals, vecs) = eig_symmetric(&symmetrize(&a.t().dot(&a)).view())?;
    let mut order: Vec<usize> = (0..p1).collect();
    order.sort_by(|&i, &j| vals[j].total_cmp(&vals[i]));
    let eigenvalues: Vec<f64> = order.iter().map(|&i| vals[i]).collect();
    if eigenvalues[0].is_nan() || eigenvalues[0] > MAX_EIGENVALUE {
        return Err(EconError::SingularMatrix {
            context: format!(
                "Johansen VECM: the largest squared canonical correlation is {:.12} (1 up to rounding): a \
                 combination of the differences is an exact linear function of the level terms and short-run \
                 regressors, so the test statistics are infinite",
                eigenvalues[0]
            ),
            suggestion: "check for identities or deterministic series among the variables"
                .to_string(),
        });
    }
    let e = Array2::from_shape_fn((p1, p1), |(i, c)| vecs[[i, order[c]]]);
    let mut v = solve_lower_t(&l, &e); // V = L^-T E
    let mut unnormalised = Vec::new();
    for c in 0..p1 {
        let mut col = v.column_mut(c);
        let first = col[0];
        let scale = col.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));
        if first.abs() > 1e-12 * scale {
            col.mapv_inplace(|x| x / first);
        } else {
            // urca divides by the zero first element (Inf/NaN); scale by the largest entry instead.
            let pivot = col
                .iter()
                .copied()
                .fold(0.0_f64, |acc, x| if x.abs() > acc.abs() { x } else { acc });
            col.mapv_inplace(|x| x / pivot);
            unnormalised.push(c);
        }
    }
    let vtsv = symmetrize(&v.t().dot(&skk).dot(&v));
    let vtsv_inv = sym_inverse(vtsv.view(), SingularPolicy::Error, "V'S11V")?.inverse;
    let w = s0k.dot(&v).dot(&vtsv_inv);
    let pi = solve_lower_t(&l, &solve_lower(&l, &s0k.t().to_owned()))
        .t()
        .to_owned(); // S01 S11^-1

    let log1m: Vec<f64> = eigenvalues.iter().map(|&lam| (-lam).ln_1p()).collect();
    let trace_stat: Vec<f64> = (0..k)
        .map(|r| -tf * log1m[r..].iter().sum::<f64>())
        .collect();
    let eigen_stat: Vec<f64> = (0..k).map(|r| -tf * log1m[r]).collect();
    let (trace_table, eigen_table) = critical_value_tables(ecdet);
    let trace_cv: Vec<[f64; 3]> = (0..k).map(|r| trace_table[k - r - 1]).collect();
    let eigen_cv: Vec<[f64; 3]> = (0..k).map(|r| eigen_table[k - r - 1]).collect();
    let selected_rank = (0..k)
        .take_while(|&r| trace_stat[r] > trace_cv[r][1])
        .count();
    let rank = config.rank.unwrap_or(selected_rank);

    let mut warnings = Vec::new();
    let mut restricted = Restricted::default();
    if (1..k).contains(&rank) {
        if let Some(&c) = unnormalised.iter().find(|&&c| c < rank) {
            return Err(invalid(format!(
                "cointegrating vector {} has no weight on the first variable '{}', so beta cannot be normalised \
                 on it as urca::cajorls does; put a variable that enters the cointegrating relations first",
                c + 1,
                names[0]
            )));
        }
        let design = Design {
            z0: &z0,
            z1: &z1,
            zk: &zk,
            t,
            m,
            has_const,
        };
        restricted = restricted_vecm(&design, &v, rank)?;
    } else if rank == 0 {
        warnings.push(format!(
            "the trace test at the 5% level does not reject rank 0 (no cointegration), so beta, alpha and Gamma \
             are not estimated; set rank (1 to {}) to estimate a restricted VECM",
            k - 1
        ));
    } else {
        warnings.push(format!(
            "the trace test at the 5% level rejects every rank below k = {k} (the levels VAR is stationary), so \
             beta, alpha and Gamma are not estimated; set rank (1 to {}) to estimate a restricted VECM",
            k - 1
        ));
    }
    if !unnormalised.is_empty() {
        let list: Vec<String> = unnormalised.iter().map(|c| (c + 1).to_string()).collect();
        warnings.push(format!(
            "eigenvectors {} have no weight on the first variable '{}' and are scaled by their largest element \
             instead (urca would divide by zero)",
            list.join(", "),
            names[0]
        ));
    }

    let trace_crit_values = trace_cv.iter().map(|c| c[1]).collect();
    Ok(VecmResult {
        rank,
        lags: big_k,
        n_vars: k,
        n_obs: t,
        eigenvalues,
        beta: restricted.beta,
        alpha: restricted.alpha,
        gamma: restricted.gamma,
        trace_stats: trace_stat.clone(),
        trace_crit_values,
        variables: names.to_vec(),
        deterministic: ecdet,
        beta_rows,
        trace_stat,
        trace_cv,
        eigen_stat,
        eigen_cv,
        selected_rank,
        v_normalised: array2_to_vec(&v),
        w_loadings: array2_to_vec(&w),
        pi: array2_to_vec(&pi),
        short_run_names,
        constant: restricted.constant,
        warnings,
    })
}

/// The design matrices of the restricted stage.
struct Design<'a> {
    z0: &'a Array2<f64>,
    z1: &'a Array2<f64>,
    zk: &'a Array2<f64>,
    t: usize,
    m: usize,
    has_const: bool,
}

#[derive(Default)]
struct Restricted {
    beta: Vec<Vec<f64>>,
    alpha: Vec<Vec<f64>>,
    gamma: Vec<Vec<f64>>,
    constant: Vec<f64>,
}

/// urca::cajorls (beta, alpha, Gamma, constant) for 1 <= r <= k - 1.
fn restricted_vecm(d: &Design<'_>, v: &Array2<f64>, r: usize) -> EconResult<Restricted> {
    let beta_raw = v.slice(s![.., ..r]).to_owned();
    let top = beta_raw.slice(s![..r, ..r]).to_owned();
    // matrix_inverse refuses a condition number above CONDITION_THRESHOLD, so the check uses that bound.
    if condition_number(&top.view())? > CONDITION_THRESHOLD {
        return Err(EconError::SingularMatrix {
            context: format!(
                "Johansen VECM: the top {r} x {r} block of the first {r} cointegrating vectors is singular, so beta \
                 cannot be normalised as urca::cajorls does"
            ),
            suggestion:
                "reorder the variables so that the first r enter the cointegrating relations independently"
                    .to_string(),
        });
    }
    let beta = beta_raw.dot(&matrix_inverse(&top.view())?);
    let mut x = resource::try_zeros2(d.t, r + d.m, "VECM restricted regressors")?;
    x.slice_mut(s![.., ..r]).assign(&d.zk.dot(&beta));
    x.slice_mut(s![.., r..]).assign(d.z1);
    let qr2 = Qr::decompose(x.view(), LM_QR_TOL)?;
    if qr2.rank() < r + d.m {
        return Err(EconError::SingularMatrix {
            context: "Johansen VECM: the error-correction terms are collinear with the short-run regressors"
                .to_string(),
            suggestion: "lower the rank or lag_order".to_string(),
        });
    }
    let coef = coefficient_rows(&qr2, d.z0)?; // k x (r + m): ect1..ectr, [constant], lagged differences
    let alpha: Vec<Vec<f64>> = coef.iter().map(|row| row[..r].to_vec()).collect();
    let skip = r + usize::from(d.has_const);
    let constant: Vec<f64> = if d.has_const {
        coef.iter().map(|row| row[r]).collect()
    } else {
        Vec::new()
    };
    let gamma: Vec<Vec<f64>> = coef.iter().map(|row| row[skip..].to_vec()).collect();
    Ok(Restricted {
        beta: array2_to_vec(&beta),
        alpha,
        gamma,
        constant,
    })
}

/// Residuals of every column of `ys` on the full-rank design `x` (whose QR is `qr`).
fn residuals(qr: &Qr, x: &Array2<f64>, ys: &Array2<f64>) -> EconResult<Array2<f64>> {
    let mut out = ys.clone();
    for c in 0..ys.ncols() {
        let fitted: Array1<f64> = x.dot(&qr.coefficients(ys.column(c))?);
        out.column_mut(c).zip_mut_with(&fitted, |o, f| *o -= f);
    }
    Ok(out)
}

/// OLS coefficients of every column of `ys` on the design of `qr`, one row per column of `ys`.
fn coefficient_rows(qr: &Qr, ys: &Array2<f64>) -> EconResult<Vec<Vec<f64>>> {
    (0..ys.ncols())
        .map(|c| Ok(qr.coefficients(ys.column(c))?.to_vec()))
        .collect()
}

fn symmetrize(m: &Array2<f64>) -> Array2<f64> {
    (m + &m.t()) * 0.5
}

/// Solves L X = B for lower-triangular L with a positive diagonal (forward substitution).
fn solve_lower(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let mut x = b.clone();
    for col in 0..x.ncols() {
        for i in 0..n {
            let mut acc = x[[i, col]];
            for p in 0..i {
                acc -= l[[i, p]] * x[[p, col]];
            }
            x[[i, col]] = acc / l[[i, i]];
        }
    }
    x
}

/// Solves L' X = B for lower-triangular L with a positive diagonal (back substitution).
fn solve_lower_t(l: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n = l.nrows();
    let mut x = b.clone();
    for col in 0..x.ncols() {
        for i in (0..n).rev() {
            let mut acc = x[[i, col]];
            for p in (i + 1)..n {
                acc -= l[[p, i]] * x[[p, col]];
            }
            x[[i, col]] = acc / l[[i, i]];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::econometrics::timeseries::common::test_fixtures::box_muller_ts;
    use crate::test_support::load_case;
    use p2a_validation::{Check, Tol};
    use polars::prelude::*;

    // Aliases for brevity; never glob-import the variants (VecmDeterministic::None shadows Option::None).
    use VecmDeterministic as D;

    const DEN: [&str; 4] = ["LRM", "LRY", "IBO", "IDE"];
    const SIM: [&str; 3] = ["x1", "x2", "x3"];

    /// R's vec(): column-major order of a row-major matrix.
    fn col_major(m: &[Vec<f64>]) -> Vec<f64> {
        let nc = m.first().map_or(0, Vec::len);
        (0..nc)
            .flat_map(|j| m.iter().map(move |row| row[j]))
            .collect()
    }
    /// urca stores statistics and critical values from H0 r <= k-1 down to r = 0.
    fn urca_order(v: &[f64]) -> Vec<f64> {
        v.iter().rev().copied().collect()
    }
    fn cfg(ecdet: VecmDeterministic, lag_order: usize, rank: Option<usize>) -> VecmConfig {
        VecmConfig {
            lag_order,
            ecdet,
            rank,
        }
    }
    fn cv_col(cv: &[[f64; 3]], col: usize) -> Vec<f64> {
        cv.iter().rev().map(|row| row[col]).collect()
    }
    /// cajorls' coef(rlm) in R's vec() order: per equation, ect1..ectr, [constant], lagged differences.
    fn rlm_coef(r: &VecmResult) -> Vec<f64> {
        (0..r.n_vars)
            .flat_map(|j| {
                let mut v = r.alpha[j].clone();
                if !r.constant.is_empty() {
                    v.push(r.constant[j]);
                }
                v.extend_from_slice(&r.gamma[j]);
                v
            })
            .collect()
    }
    fn rlm_rows(r: &VecmResult) -> usize {
        r.rank + usize::from(!r.constant.is_empty()) + r.gamma.first().map_or(0, Vec::len)
    }

    /// R, CLI and MCP spellings of urca's ecdet; unknown values list the accepted ones.
    #[test]
    fn test_vecm_deterministic_parses_aliases_and_rejects_unknown() {
        VecmDeterministic::choices().validate().unwrap();
        for (s, v) in [
            ("none", VecmDeterministic::None),
            ("NONE", VecmDeterministic::None),
            ("const", VecmDeterministic::Const),
            ("constant", VecmDeterministic::Const),
            (" Const ", VecmDeterministic::Const),
            ("trend", VecmDeterministic::Trend),
        ] {
            assert_eq!(s.parse::<VecmDeterministic>().unwrap(), v, "{s}");
        }
        let message = "drift"
            .parse::<VecmDeterministic>()
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("unknown ecdet 'drift'") && message.contains("none, const, trend"),
            "{message}"
        );
        assert_eq!(VecmDeterministic::Const.to_string(), "const");
        assert_eq!(
            serde_json::to_value(VecmDeterministic::Trend).unwrap(),
            "trend"
        );
    }

    /// urca's defaults K = 2 and ecdet = "none"; rank by the trace test.
    #[test]
    fn test_vecm_config_default() {
        let c = VecmConfig::default();
        assert_eq!(
            (c.lag_order, c.ecdet, c.rank),
            (2, VecmDeterministic::None, None)
        );
    }

    /// Before T3-WP14, run_vecm returned beta = (1, 0, 0, 0)', alpha = -0.1 and Gamma = 0 for any data, and
    /// its "eigenvalues" 1 / (1 + diag((Y'Y)^-1)) moved with the scale of the data (ECON-52). R:
    /// ca.jo(denmark, ecdet = "none", K = 2, spec = "transitory"), the legacy wrapper's configuration;
    /// LRM x 100 leaves the eigenvalues unchanged (R: 1.8e-11) and multiplies beta[2..4] by 100 (R: 3.2e-11).
    #[test]
    fn test_validate_vecm_beta_depends_on_data() {
        let (case, ds) = load_case("vecm_denmark");
        let r = run_vecm(&ds, &DEN, 2, 1).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .vector("none_eigenvalues", &r.eigenvalues, Tol::coef(n))
            .vector("none_trace_stat", &urca_order(&r.trace_stats), Tol::stat(n))
            .vector("none_beta", &col_major(&r.beta), Tol::coef(n))
            .finish();
        let (y, _) = df_to_array2(ds.df(), &DEN).unwrap();
        let col =
            |j: usize, scale: f64| y.column(j).iter().map(|v| scale * v).collect::<Vec<f64>>();
        let scaled = Dataset::new(
            df!("LRM" => col(0, 100.0), "LRY" => col(1, 1.0), "IBO" => col(2, 1.0), "IDE" => col(3, 1.0))
                .unwrap(),
        );
        let s = run_vecm(&scaled, &DEN, 2, 1).unwrap();
        let tol = Tol::new(1e-8, 1e-12);
        for (a, b) in s.eigenvalues.iter().zip(&r.eigenvalues) {
            assert!(tol.allows(*a, *b), "eigenvalue {a} vs {b}");
        }
        for i in 1..4 {
            assert!(
                tol.allows(s.beta[i][0], 100.0 * r.beta[i][0]),
                "beta[{i}] {} vs {}",
                s.beta[i][0],
                100.0 * r.beta[i][0]
            );
        }
    }

    /// R: urca::ca.jo(type = "trace" | "eigen") on vecm_denmark (N = 55, T = 53): ecdet = "const", K = 2,
    /// transitory (eigenvalues, statistics, critical values, V, W, PI), plus the ecdet = "none" statistics.
    #[test]
    fn test_validate_vecm_denmark_trace_eigen() {
        let (case, ds) = load_case("vecm_denmark");
        let r = run_vecm_with(&ds, &DEN, &cfg(D::Const, 2, Some(1))).unwrap();
        let rn = run_vecm_with(&ds, &DEN, &cfg(D::None, 2, Some(1))).unwrap();
        let n = r.n_obs;
        // W's fifth column belongs to the structural zero eigenvalue of the restricted constant: zero in
        // theory, rounding noise in R (|w| <= 8.8e-13), hence abs 1e-11.
        let w_tol = Tol::new(Tol::coef(n).rel, 1e-11);
        Check::new(&case)
            .scalar("n_obs", (r.n_obs + r.lags) as f64, Tol::exact())
            .vector("trace_stat", &urca_order(&r.trace_stat), Tol::stat(n))
            .vector("trace_cv10", &cv_col(&r.trace_cv, 0), Tol::exact())
            .vector("trace_cv5", &cv_col(&r.trace_cv, 1), Tol::exact())
            .vector("trace_cv1", &cv_col(&r.trace_cv, 2), Tol::exact())
            .vector("eigen_stat", &urca_order(&r.eigen_stat), Tol::stat(n))
            .vector("eigen_cv10", &cv_col(&r.eigen_cv, 0), Tol::exact())
            .vector("eigen_cv5", &cv_col(&r.eigen_cv, 1), Tol::exact())
            .vector("eigen_cv1", &cv_col(&r.eigen_cv, 2), Tol::exact())
            .vector("eigenvalues", &r.eigenvalues, Tol::coef(n))
            .vector("v_normalised", &col_major(&r.v_normalised), Tol::coef(n))
            .vector("w_loadings", &col_major(&r.w_loadings), w_tol)
            .vector("pi", &col_major(&r.pi), Tol::coef(n))
            .vector("none_eigenvalues", &rn.eigenvalues, Tol::coef(n))
            .vector("none_trace_stat", &urca_order(&rn.trace_stat), Tol::stat(n))
            .vector("none_trace_cv5", &cv_col(&rn.trace_cv, 1), Tol::exact())
            .finish();
        assert!(
            r.w_loadings.iter().all(|row| row[4].abs() < 1e-10),
            "W[, 5] must vanish: {:?}",
            r.w_loadings
        );
        assert_eq!(r.trace_stats, r.trace_stat);
        assert_eq!(r.trace_crit_values, vec![53.12, 34.91, 19.96, 9.24]);
        assert_eq!(
            (r.deterministic, r.eigenvalues.len(), rn.eigenvalues.len()),
            (D::Const, 5, 4)
        );
        assert_eq!(
            r.beta_rows,
            ["LRM.l1", "LRY.l1", "IBO.l1", "IDE.l1", "constant"]
        );
        assert_eq!(
            r.short_run_names,
            ["LRM.dl1", "LRY.dl1", "IBO.dl1", "IDE.dl1"]
        );
        assert_eq!(rn.short_run_names[0], "constant");
    }

    /// R: urca::cajorls(r = 1 | 2) on the same fits: beta normalised by its top r x r block, and the OLS
    /// coefficients (alpha, [constant], Gamma) of Z0 on [ZK beta, Z1].
    #[test]
    fn test_validate_vecm_denmark_beta_alpha_gamma() {
        let (case, ds) = load_case("vecm_denmark");
        let r = run_vecm_with(&ds, &DEN, &cfg(D::Const, 2, Some(1))).unwrap();
        let rn = run_vecm_with(&ds, &DEN, &cfg(D::None, 2, Some(1))).unwrap();
        let r2 = run_vecm_with(&ds, &DEN, &cfg(D::Const, 2, Some(2))).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .vector("beta", &col_major(&r.beta), Tol::coef(n))
            .vector("rlm_coef", &rlm_coef(&r), Tol::coef(n))
            .scalar("rlm_rownames_n", rlm_rows(&r) as f64, Tol::exact())
            .vector("none_beta", &col_major(&rn.beta), Tol::coef(n))
            .vector("none_rlm_coef", &rlm_coef(&rn), Tol::coef(n))
            .vector("r2_beta", &col_major(&r2.beta), Tol::coef(n))
            .vector("r2_rlm_coef", &rlm_coef(&r2), Tol::coef(n))
            .scalar("r2_rlm_rownames_n", rlm_rows(&r2) as f64, Tol::exact())
            .finish();
        assert!(r.constant.is_empty());
        assert_eq!(
            (rn.constant.len(), r2.alpha[0].len(), r2.beta.len()),
            (4, 2, 5)
        );
    }

    /// R: urca::ca.jo / cajorls on vecm_sim3 (N = 250, T = 247, K = 3), ecdet = "none" and "trend"
    /// (restricted trend: beta row trend.l1).
    #[test]
    fn test_validate_vecm_sim3_none_trend() {
        let (case, ds) = load_case("vecm_sim3");
        let r = run_vecm_with(&ds, &SIM, &cfg(D::None, 3, Some(1))).unwrap();
        let rt = run_vecm_with(&ds, &SIM, &cfg(D::Trend, 3, Some(1))).unwrap();
        let n = r.n_obs;
        Check::new(&case)
            .scalar("n_obs", (r.n_obs + r.lags) as f64, Tol::exact())
            .vector("trace_stat", &urca_order(&r.trace_stat), Tol::stat(n))
            .vector("trace_cv10", &cv_col(&r.trace_cv, 0), Tol::exact())
            .vector("trace_cv5", &cv_col(&r.trace_cv, 1), Tol::exact())
            .vector("trace_cv1", &cv_col(&r.trace_cv, 2), Tol::exact())
            .vector("eigen_stat", &urca_order(&r.eigen_stat), Tol::stat(n))
            .vector("eigen_cv10", &cv_col(&r.eigen_cv, 0), Tol::exact())
            .vector("eigen_cv5", &cv_col(&r.eigen_cv, 1), Tol::exact())
            .vector("eigen_cv1", &cv_col(&r.eigen_cv, 2), Tol::exact())
            .vector("eigenvalues", &r.eigenvalues, Tol::coef(n))
            .vector("v_normalised", &col_major(&r.v_normalised), Tol::coef(n))
            .vector("w_loadings", &col_major(&r.w_loadings), Tol::coef(n))
            .vector("pi", &col_major(&r.pi), Tol::coef(n))
            .vector("beta", &col_major(&r.beta), Tol::coef(n))
            .vector("rlm_coef", &rlm_coef(&r), Tol::coef(n))
            .scalar("rlm_rownames_n", rlm_rows(&r) as f64, Tol::exact())
            .vector("trend_eigenvalues", &rt.eigenvalues, Tol::coef(n))
            .vector(
                "trend_trace_stat",
                &urca_order(&rt.trace_stat),
                Tol::stat(n),
            )
            .vector("trend_trace_cv5", &cv_col(&rt.trace_cv, 1), Tol::exact())
            .vector(
                "trend_eigen_stat",
                &urca_order(&rt.eigen_stat),
                Tol::stat(n),
            )
            .vector("trend_beta", &col_major(&rt.beta), Tol::coef(n))
            .vector("trend_rlm_coef", &rlm_coef(&rt), Tol::coef(n))
            .finish();
        assert_eq!((r.selected_rank, rt.selected_rank), (1, 1));
        assert_eq!(rt.beta_rows, ["x1.l1", "x2.l1", "x3.l1", "trend.l1"]);
        assert_eq!(r.short_run_names.len(), 7);
    }

    /// Rank from the trace test at 5% (derived from the pinned statistics and critical values):
    /// denmark const 52.71 < 53.12 -> 0; denmark none 48.80 > 48.28, 17.29 < 31.52 -> 1; sim3 none -> 1;
    /// denmark[1:16, ] none rejects every rank (112.69, 45.61, 26.78, 10.45 against 48.28, 31.52, 17.95, 8.18) -> 4.
    #[test]
    fn test_validate_vecm_rank_selected_by_trace_test() {
        let (_, den) = load_case("vecm_denmark");
        let (_, sim) = load_case("vecm_sim3");
        let c0 = run_vecm_with(&den, &DEN, &cfg(D::Const, 2, None)).unwrap();
        assert_eq!((c0.selected_rank, c0.rank), (0, 0));
        assert!(
            c0.beta.is_empty()
                && c0.alpha.is_empty()
                && c0.gamma.is_empty()
                && c0.constant.is_empty()
        );
        assert_eq!(c0.trace_stat.len(), 4, "the test itself is still reported");
        assert_eq!(c0.warnings.len(), 1);
        assert!(
            c0.warnings[0].contains("does not reject rank 0"),
            "{:?}",
            c0.warnings
        );
        let c1 = run_vecm_with(&den, &DEN, &cfg(D::Const, 2, Some(1))).unwrap();
        assert_eq!((c1.selected_rank, c1.rank), (0, 1));
        assert!(c1.warnings.is_empty());
        let n1 = run_vecm_with(&den, &DEN, &cfg(D::None, 2, None)).unwrap();
        let n1_fixed = run_vecm_with(&den, &DEN, &cfg(D::None, 2, Some(1))).unwrap();
        assert_eq!((n1.selected_rank, n1.rank), (1, 1));
        assert_eq!(n1.beta, n1_fixed.beta);
        assert_eq!(
            run_vecm_with(&sim, &SIM, &cfg(D::None, 3, None))
                .unwrap()
                .selected_rank,
            1
        );
        let (y, names) = df_to_array2(den.df(), &DEN).unwrap();
        let full = johansen_vecm(y.slice(s![..16, ..]), &names, &cfg(D::None, 2, None)).unwrap();
        assert_eq!((full.selected_rank, full.rank), (4, 4));
        assert!(full.beta.is_empty());
        assert!(
            full.warnings[0].contains("rejects every rank below k = 4"),
            "{:?}",
            full.warnings
        );
    }

    /// Degenerate inputs: every case names the parameter or the condition; none panics.
    #[test]
    fn test_validate_vecm_rejects_degenerate_inputs() {
        let (_, ds) = load_case("vecm_denmark");
        fn invalid(res: EconResult<VecmResult>, needle: &str) {
            match res {
                Err(EconError::InvalidSpecification { message }) => {
                    assert!(message.contains(needle), "{message}")
                }
                other => panic!("expected InvalidSpecification naming '{needle}', got {other:?}"),
            }
        }
        let base = VecmConfig::default();
        invalid(
            run_vecm_with(&ds, &DEN, &cfg(D::None, 1, None)),
            "lag_order",
        ); // urca: K >= 2
        invalid(
            run_vecm_with(&ds, &DEN, &cfg(D::None, 0, None)),
            "lag_order",
        );
        invalid(run_vecm_with(&ds, &DEN, &cfg(D::None, 2, Some(0))), "rank"); // cajorls: 1..k-1
        invalid(run_vecm_with(&ds, &DEN, &cfg(D::None, 2, Some(4))), "rank");
        invalid(run_vecm(&ds, &DEN, 2, 0), "rank");
        invalid(run_vecm_with(&ds, &["LRM"], &base), "at least 2 variables");
        invalid(run_vecm_with(&ds, &["LRM", "LRM"], &base), "distinct");
        let mut seed = 7_u64;
        let y12 = Array2::from_shape_fn((200, 12), |_| box_muller_ts(&mut seed));
        let names12: Vec<String> = (1..=12).map(|i| format!("v{i}")).collect();
        invalid(
            johansen_vecm(y12.view(), &names12, &base),
            "at most 11 variables",
        ); // urca: no critical values
        let (y, names) = df_to_array2(ds.df(), &DEN).unwrap();
        let mut y_nan = y.clone();
        y_nan[[10, 2]] = f64::NAN;
        invalid(johansen_vecm(y_nan.view(), &names, &base), "finite");
        // N = 14 < K + m + k + p1 = 15 (urca::ca.jo runs from N = 12 and returns lambda = 1, NaN statistics).
        for ecdet in [D::None, D::Const] {
            let c = cfg(ecdet, 2, None);
            match johansen_vecm(y.slice(s![..14, ..]), &names, &c) {
                Err(EconError::InsufficientData {
                    required, provided, ..
                }) => assert_eq!((required, provided), (15, 14)),
                other => panic!("{ecdet}: expected InsufficientData, got {other:?}"),
            }
            assert!(
                johansen_vecm(y.slice(s![..15, ..]), &names, &c).is_ok(),
                "{ecdet}: N = 15 is enough"
            );
        }
        assert!(matches!(
            run_vecm_with(&ds, &DEN, &cfg(D::None, usize::MAX, Some(1))),
            Err(EconError::InsufficientData { .. })
        ));
        // A constant series: its lagged difference is an all-zero short-run regressor (dqrdc2 marks IDE.dl1).
        let mut y_const = y.clone();
        y_const.column_mut(3).fill(0.1);
        match johansen_vecm(y_const.view(), &names, &base) {
            Err(EconError::SingularMatrix { context, .. }) => {
                assert!(context.contains("IDE.dl1"), "{context}")
            }
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
        // An exact error-correction equation, Δy2_t = y1_{t-2}: S00 and S11 are well conditioned but the largest
        // squared canonical correlation is 1 (R emulation: 1 + 4.4e-16; urca returns the statistic 4335.18).
        let mut seed = 11_u64;
        let n = 120;
        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];
        for t in 1..n {
            y1[t] = y1[t - 1] + box_muller_ts(&mut seed);
            if t >= 2 {
                y2[t] = y2[t - 1] + y1[t - 2];
            }
        }
        let y_exact = Array2::from_shape_fn((n, 2), |(t, j)| if j == 0 { y1[t] } else { y2[t] });
        match johansen_vecm(y_exact.view(), &["y1".to_string(), "y2".to_string()], &base) {
            Err(EconError::SingularMatrix { context, .. }) => assert!(
                context.contains("squared canonical correlation"),
                "{context}"
            ),
            other => panic!("expected SingularMatrix, got {other:?}"),
        }
    }

    /// Display shows the urca-style tables, the rank choice and the notes (MCP ts_vecm prints Display).
    #[test]
    fn test_validate_vecm_display_shows_tests_and_notes() {
        let (_, ds) = load_case("vecm_denmark");
        let text = run_vecm_with(&ds, &DEN, &cfg(D::Const, 2, None))
            .unwrap()
            .to_string();
        for needle in [
            "Johansen procedure",
            "ecdet = const",
            "Form: transitory",
            "Trace test",
            "Maximal eigenvalue test",
            "r <= 3 |",
            "r = 0  |",
            "52.7109",
            "53.12",
            "Rank chosen by the trace test at 5%: 0",
            "Restricted VECM: not estimated (rank 0)",
            "Warning: the trace test",
        ] {
            assert!(text.contains(needle), "'{needle}' missing:\n{text}");
        }
        let text = run_vecm_with(&ds, &DEN, &cfg(D::Const, 2, Some(1)))
            .unwrap()
            .to_string();
        for needle in [
            "Restricted VECM with rank r = 1",
            "Cointegrating vectors (beta):",
            "-6.478051",
            "Adjustment coefficients (alpha):",
            "-0.299784",
            "Short-run coefficients (Gamma):",
        ] {
            assert!(text.contains(needle), "'{needle}' missing:\n{text}");
        }
        assert!(!text.contains("Warning:"));
    }

    /// Phantom-option check: the ecdet values R distinguishes give different fits (R: largest eigenvalue
    /// 0.4482 under none, 0.4697 under const; beta has 4 rows under none, 5 under const).
    #[test]
    fn test_validate_vecm_options_change_results() {
        let (_, ds) = load_case("vecm_denmark");
        let none = run_vecm_with(&ds, &DEN, &cfg(D::None, 2, Some(1))).unwrap();
        let cnst = run_vecm_with(&ds, &DEN, &cfg(D::Const, 2, Some(1))).unwrap();
        assert!((none.eigenvalues[0] - cnst.eigenvalues[0]).abs() > 0.01);
        assert_eq!((none.beta.len(), cnst.beta.len()), (4, 5));
        assert_eq!((none.constant.len(), cnst.constant.len()), (4, 0));
        assert_eq!(
            (none.deterministic, cnst.deterministic),
            (D::None, D::Const)
        );
    }

    /// The result round-trips through JSON; ecdet serialises as urca spells it.
    #[test]
    fn test_validate_vecm_result_serde_round_trip() {
        let (_, ds) = load_case("vecm_sim3");
        let r = run_vecm_with(&ds, &SIM, &cfg(D::Trend, 3, Some(1))).unwrap();
        let json = serde_json::to_value(&r).unwrap();
        assert_eq!(json["deterministic"].as_str(), Some("trend"));
        let back: VecmResult = serde_json::from_value(json).unwrap();
        assert_eq!(
            (back.trace_cv, back.beta_rows, back.beta),
            (r.trace_cv.clone(), r.beta_rows.clone(), r.beta.clone())
        );
        assert_eq!(back.deterministic, D::Trend);
    }
}
