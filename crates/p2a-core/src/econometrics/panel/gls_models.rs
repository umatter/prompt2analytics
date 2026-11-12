//! Panel GLS (FGLS) estimators.
//!
//! Feasible Generalized Least Squares for panel data with cross-sectional
//! correlation and heteroskedasticity.
//!
//! # Reference
//!
//! Parks, R.W. (1967). "Efficient Estimation of a System of Regression Equations
//! When Disturbances Are Both Serially and Contemporaneously Correlated."
//! *Journal of the American Statistical Association*, 62, 500-509.
//!
//! R equivalent: `plm::pggls()`
//!
//! # Algorithm (as `pggls`)
//!
//! 1. The model is transformed: pooling keeps the levels with an intercept, within
//!    demeans y and X by entity, first differences take consecutive observed rows of an
//!    entity in time order (a gap is differenced across) with an intercept column of ones.
//! 2. A first-step OLS on the transformed model gives residuals; each cell `(s, t)` of the
//!    unrestricted error covariance Omega averages `e_is e_it` over the entities observed
//!    in both periods.
//! 3. Each entity is weighted by the generalized inverse of its own observed-period block
//!    `Omega_ii`, computed by `bdsmatrix`'s generalized Cholesky: LDL' in period order with
//!    no pivoting, and a pivot below `1e-10` times the largest diagonal of Omega set to
//!    zero with its column. On a balanced within model Omega has rank T - 1 (the residuals
//!    sum to zero in every entity); on an unbalanced panel the cell-averaged Omega can be
//!    indefinite, and the non-positive pivots are dropped with a warning.
//! 4. `beta = (sum X_i' G_i X_i)^-1 sum X_i' G_i y_i`, with variance
//!    `(sum X_i' G_i X_i)^-1` (no residual-variance factor), z statistics and normal
//!    p-values; R-squared is `1 - RSS / TSS` with TSS of the raw outcome.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::resource;
use crate::traits::convergence::ResultNotes;
use crate::traits::estimator::{SignificanceLevel, normal_p_value_two_sided};

use super::utils::{demean_by_entity, entity_codes, time_codes};

/// `sym_inverse` holds several m x m copies at once (copy, equilibration, eigenvectors,
/// solution). Check the working set against the allocation ceiling before building it.
const DENSE_INVERSE_COPIES: usize = 6;

/// Relative pivot tolerance of the generalized Cholesky (`bdsmatrix`'s default): a pivot
/// below this times the largest diagonal of Omega is dropped.
const GCHOL_TOLERANCE: f64 = 1e-10;

fn ensure_dense_inverse_fits(m: usize, what: &str) -> EconResult<()> {
    let cells = resource::checked_len(&[m, m, DENSE_INVERSE_COPIES], what)?;
    resource::check_alloc::<f64>(cells, what)
}

/// Panel GLS model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PanelGlsModel {
    /// Fixed effects GLS (within transformation before FGLS)
    #[default]
    FixedEffects,
    /// Pooled GLS (no effects, just FGLS)
    Pooling,
    /// First-difference GLS
    FirstDifference,
}

impl fmt::Display for PanelGlsModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelGlsModel::FixedEffects => write!(f, "Fixed Effects GLS (FEGLS)"),
            PanelGlsModel::Pooling => write!(f, "Pooled GLS"),
            PanelGlsModel::FirstDifference => write!(f, "First-Difference GLS (FDGLS)"),
        }
    }
}

/// Result from Panel GLS estimation.
///
/// Panel GLS (FGLS) allows for cross-sectional correlation and heteroskedasticity
/// in the error structure while being robust to any form of intragroup correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelGlsResult {
    /// Model type used
    pub model: PanelGlsModel,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z statistics (`coef / se`, as `summary.pggls`; NaN when the SE is not positive)
    pub t_stats: Vec<f64>,
    /// Two-sided normal p-values of the z statistics
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// R-squared: `1 - RSS / TSS`, the RSS of the transformed model against the total sum
    /// of squares of the raw outcome (as `pggls`)
    pub r_squared: f64,
    /// Number of observations of the transformed model (first differences: the
    /// differenced rows)
    pub n_obs: usize,
    /// Number of entities (cross-sectional units)
    pub n_groups: usize,
    /// Number of time periods
    pub n_periods: usize,
    /// Residual degrees of freedom: `n_obs` less the number of coefficients
    pub df_residual: usize,
    /// Residual sum of squares
    pub rss: f64,
    /// Residual standard error
    pub sigma: f64,
    /// Dimension of the estimated error covariance: the number of distinct periods among
    /// the rows of the transformed model (first differences: T - 1 when every entity has
    /// the first period)
    #[serde(skip)]
    pub error_cov_dim: usize,
    /// Any warnings generated during estimation
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,

    // ── Trait-backing storage (LinearEstimator). Skipped from JSON.
    #[serde(skip, default)]
    pub coef_arr: ndarray::Array1<f64>,
    #[serde(skip, default)]
    pub se_arr: ndarray::Array1<f64>,
    #[serde(skip, default)]
    pub residuals: ndarray::Array1<f64>,
    #[serde(skip, default)]
    pub vcov: ndarray::Array2<f64>,

    /// Rows dropped because of a missing value in y, a regressor, the entity or the time
    /// (plm `na.omit`).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl ResultNotes for PanelGlsResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// Residuals of a first-step OLS fit; aliased columns get R's aliasing warning.
fn first_step_residuals(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    names: &[String],
    warnings: &mut Vec<String>,
) -> EconResult<Vec<f64>> {
    let ls = least_squares(x, y)?;
    if let Some(w) = ls.aliasing_warning(names) {
        warnings.push(format!("first-step OLS: {w}"));
    }
    Ok(ls.residuals.to_vec())
}

impl crate::traits::estimator::LinearEstimator for PanelGlsResult {
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
        self.df_residual
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

impl fmt::Display for PanelGlsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n{}", "=".repeat(70))?;
        writeln!(f, "Panel GLS (FGLS) Estimation Results")?;
        writeln!(f, "{}", "=".repeat(70))?;
        writeln!(f, "Model: {}", self.model)?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(
            f,
            "Observations: {}  Entities: {}  Time periods: {}",
            self.n_obs, self.n_groups, self.n_periods
        )?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>10} {:>10}",
            "Variable", "Coefficient", "Std.Error", "z", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            let sig = match &self.significance[i] {
                SignificanceLevel::NotSignificant => "",
                SignificanceLevel::TenPercent => ".",
                SignificanceLevel::FivePercent => "*",
                SignificanceLevel::OnePercent => "**",
                SignificanceLevel::TenthPercent => "***",
            };
            writeln!(
                f,
                "{:<20} {:>12.6} {:>12.6} {:>10.4} {:>9.4}{}",
                self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.t_stats[i],
                self.p_values[i],
                sig
            )?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;
        writeln!(f, "Sigma: {:.6}  RSS: {:.4}", self.sigma, self.rss)?;

        // The missing-value note is printed once, as the last line
        super::types::write_warnings_then_missing(f, "\nWarnings:", &self.warnings, &self.missing)?;

        Ok(())
    }
}

/// Run Panel GLS (Feasible Generalized Least Squares) estimation.
///
/// Panel GLS estimates regression coefficients while allowing an unrestricted
/// within-entity (serial) error covariance. The method (see the module docs):
/// 1. Transforms the model (pooled levels, within, or first differences) and fits OLS
/// 2. Uses its residuals to estimate the T x T error covariance
/// 3. Re-estimates the transformed model by GLS, each entity weighted by the
///    generalized inverse of its own observed-period block
///
/// # Arguments
///
/// * `dataset` - The dataset containing panel data
/// * `y_col` - Name of dependent variable column
/// * `x_cols` - Names of independent variable columns
/// * `entity_col` - Name of entity/group identifier column
/// * `time_col` - Name of time period identifier column
/// * `model` - Model type: FixedEffects, Pooling, or FirstDifference
///
/// # Returns
///
/// `PanelGlsResult` containing estimates, standard errors, and diagnostics.
///
/// # Errors
///
/// A panel with more time periods than entities (T > N) is refused with
/// `InvalidSpecification`: the unrestricted T x T error covariance is estimated
/// from N residual vectors and has no inverse. Two rows of one entity in one period
/// are `InvalidSpecification` too.
///
/// # Missing values
///
/// Rows with a missing value (null or NaN) in `y`, a regressor, the entity or the time
/// are dropped and reported in `missing` (plm `na.omit`). Integer outcomes and times of
/// any integer width are accepted; a fractional time is `InvalidSpecification`.
///
/// # Reference
///
/// Implementation adapted from R's plm package pggls() function.
/// Croissant, Y., & Millo, G. (2008). "Panel Data Econometrics in R: The plm Package."
/// *Journal of Statistical Software*, 27(2).
pub fn run_panel_gls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
    model: Option<PanelGlsModel>,
) -> EconResult<PanelGlsResult> {
    let model = model.unwrap_or_default();
    let mut warnings = Vec::new();

    // y, X, the entity and the time on the rows complete in all of them (plm
    // na.omit); integer outcomes and times of any integer width are numbers, a
    // fractional time is an error
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .factor(entity_col)
        .time_index(time_col)
        .na(NaPolicy::Omit)
        .extract()?;
    let missing = cols.missing().clone();
    if let Some(note) = missing.note() {
        warnings.push(note);
    }
    let y: &[f64] = cols.numeric(y_col);
    let entity = cols.factor(entity_col);
    let (entity_ids, n_groups) = entity_codes(entity);
    let time_values: &[i64] = cols.time_index(time_col);
    let (row_time_idx, unique_times) = time_codes(time_values);
    let n_periods = unique_times.len();

    let n = y.len();

    // The unrestricted T x T error covariance is estimated from N residual
    // vectors, so its rank is at most N: with T > N it has no inverse.
    if n_periods > n_groups {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Panel GLS estimates an unrestricted T x T error covariance from the \
                 residuals of N entities and needs N >= T; found N = {n_groups} entities and \
                 T = {n_periods} distinct values of '{time_col}'. Use a coarser time variable, \
                 or a fixed- or random-effects model."
            ),
        });
    }

    // Rows of each entity in time order, as pggls's order(id, time): the generalized
    // Cholesky below has no pivoting, so the period order matters. Two rows of one
    // entity in one period are refused.
    let mut rows_by_entity: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &entity_id) in entity_ids.iter().enumerate() {
        rows_by_entity[entity_id].push(i);
    }
    for rows in &mut rows_by_entity {
        rows.sort_by_key(|&i| row_time_idx[i]);
        if let Some(w) = rows
            .windows(2)
            .find(|w| row_time_idx[w[0]] == row_time_idx[w[1]])
        {
            return Err(EconError::InvalidSpecification {
                message: format!(
                    "Panel GLS needs at most one row per entity and period; entity '{}' of \
                     '{entity_col}' has two rows in period {} of '{time_col}'",
                    entity.label(w[0]),
                    time_values[w[0]]
                ),
            });
        }
    }

    // Design matrix ("(Intercept)" first for pooling and first differences)
    let has_intercept = model != PanelGlsModel::FixedEffects;
    let dm = cols.matrix(x_cols, has_intercept)?;
    let x = &dm.data;
    let k = x.ncols();

    // Step 1: the transformed model on which both GLS steps run. `groups` holds each
    // entity's transformed rows in time order and `row_period` the period of each
    // transformed row.
    let transformed: Option<(Array1<f64>, Array2<f64>)>;
    let (groups, row_period, t_dim) = match model {
        PanelGlsModel::Pooling => {
            transformed = None;
            (rows_by_entity, row_time_idx, n_periods)
        }
        PanelGlsModel::FixedEffects => {
            // Within transformation (demean by entity)
            let y_arr = Array1::from(y.to_vec());
            let y_demeaned = demean_by_entity(&y_arr, &entity_ids, n_groups);
            let mut x_demeaned = resource::try_zeros2(n, k, "panel GLS within design")?;
            for j in 0..k {
                let col = x.column(j).to_owned();
                x_demeaned
                    .column_mut(j)
                    .assign(&demean_by_entity(&col, &entity_ids, n_groups));
            }
            transformed = Some((y_demeaned, x_demeaned));
            (rows_by_entity, row_time_idx, n_periods)
        }
        PanelGlsModel::FirstDifference => {
            warnings.push("First-difference GLS drops first observation per entity".to_string());
            // Consecutive observed rows of an entity in time order are differenced (a gap
            // is differenced across, as plm); each differenced row takes the period of its
            // later row, and the intercept column stays a column of ones
            let n_diff: usize = rows_by_entity
                .iter()
                .map(|r| r.len().saturating_sub(1))
                .sum();
            if n_diff == 0 {
                return Err(EconError::InsufficientData {
                    required: 2,
                    provided: 1,
                    context: "First-difference requires at least 2 time periods per entity"
                        .to_string(),
                });
            }
            let what = "panel GLS first-difference design";
            let mut y_diff = resource::try_zeros1(n_diff, what)?;
            let mut x_diff = resource::try_zeros2(n_diff, k, what)?;
            let mut period = resource::try_with_capacity(n_diff, what)?;
            let mut groups = Vec::with_capacity(n_groups);
            let mut r = 0;
            for rows in &rows_by_entity {
                let mut g = Vec::with_capacity(rows.len().saturating_sub(1));
                for w in rows.windows(2) {
                    let (a, b) = (w[0], w[1]);
                    y_diff[r] = y[b] - y[a];
                    for j in 0..k {
                        x_diff[[r, j]] = x[[b, j]] - x[[a, j]];
                    }
                    if has_intercept {
                        x_diff[[r, 0]] = 1.0;
                    }
                    period.push(row_time_idx[b]);
                    g.push(r);
                    r += 1;
                }
                groups.push(g);
            }
            // Number the periods that occur among the differenced rows
            let mut code = vec![usize::MAX; n_periods];
            for &p in &period {
                code[p] = 0;
            }
            let mut t_dim = 0;
            for c in code.iter_mut().filter(|c| **c == 0) {
                *c = t_dim;
                t_dim += 1;
            }
            for p in period.iter_mut() {
                *p = code[*p];
            }
            transformed = Some((y_diff, x_diff));
            (groups, period, t_dim)
        }
    };
    let (yt, xt) = match &transformed {
        Some((yt, xt)) => (yt.view(), xt.view()),
        None => (ArrayView1::from(y), x.view()),
    };
    let n_t = yt.len();

    let initial_resid = first_step_residuals(xt, yt, &dm.column_names, &mut warnings)?;

    // Step 2: the T x T error covariance, each cell averaged over the entities
    // observed in both periods
    let omega_what = "panel GLS T x T error covariance";
    ensure_dense_inverse_fits(t_dim, omega_what)?;
    let mut omega = resource::try_zeros2(t_dim, t_dim, omega_what)?;
    let mut omega_counts: Vec<usize> = resource::try_vec(
        resource::checked_len(&[t_dim, t_dim], omega_what)?,
        0,
        omega_what,
    )?;

    for rows in &groups {
        resource::checkpoint()?;
        for (a, &i) in rows.iter().enumerate() {
            if a % 256 == 255 {
                resource::checkpoint()?;
            }
            let s_idx = row_period[i];
            for &j in rows {
                let t_idx = row_period[j];
                omega[[s_idx, t_idx]] += initial_resid[i] * initial_resid[j];
                omega_counts[s_idx * t_dim + t_idx] += 1;
            }
        }
    }

    for s in 0..t_dim {
        for t in 0..t_dim {
            let count = omega_counts[s * t_dim + t];
            if count > 0 {
                omega[[s, t]] /= count as f64;
            }
        }
    }

    // Pivots below this are dropped (bdsmatrix's tolerance, relative to the largest
    // diagonal of the whole Omega)
    let eps = GCHOL_TOLERANCE * omega.diag().iter().copied().fold(0.0_f64, f64::max);

    // The full Omega once, for its rank: the within residuals sum to zero in every
    // entity of a balanced panel, so its rank is T - 1
    let full = gchol_inverse(omega.view(), eps)?;
    if !full.negative && full.rank < t_dim {
        warnings.push(format!(
            "panel GLS error covariance (T x T) is singular (rank {} of {t_dim}); a \
             generalized inverse was used",
            full.rank
        ));
    }
    drop(full);

    // Step 3: GLS. Each entity is weighted by the generalized-Cholesky inverse of its
    // own observed-period block of Omega; the entities are grouped by period pattern so
    // that each distinct block is factorised once.
    let mut pattern_index: HashMap<Box<[usize]>, usize> = HashMap::new();
    let mut pattern_members: Vec<Vec<usize>> = Vec::new();
    let mut scratch: Vec<usize> = Vec::with_capacity(t_dim);
    for (e, rows) in groups.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        scratch.clear();
        scratch.extend(rows.iter().map(|&r| row_period[r]));
        match pattern_index.get(scratch.as_slice()) {
            Some(&p) => pattern_members[p].push(e),
            None => {
                pattern_index.insert(scratch.clone().into_boxed_slice(), pattern_members.len());
                pattern_members.push(vec![e]);
            }
        }
    }
    drop(pattern_index);

    let mut xtox = Array2::<f64>::zeros((k, k));
    let mut xtoy = Array1::<f64>::zeros(k);
    let block_what = "panel GLS per-entity covariance";
    let mut gx: Vec<f64> = resource::try_vec(
        resource::checked_len(&[t_dim, k], block_what)?,
        0.0,
        block_what,
    )?;
    let mut gy: Vec<f64> = resource::try_vec(t_dim, 0.0, block_what)?;
    let mut indefinite_entities = 0usize;
    for members in &pattern_members {
        resource::checkpoint()?;
        let periods: Vec<usize> = groups[members[0]].iter().map(|&r| row_period[r]).collect();
        let m = periods.len();
        let mut block = resource::try_zeros2(m, m, block_what)?;
        for (a, &pa) in periods.iter().enumerate() {
            for (b, &pb) in periods.iter().enumerate() {
                block[[a, b]] = omega[[pa, pb]];
            }
        }
        let g = gchol_inverse(block.view(), eps)?;
        if g.negative {
            indefinite_entities += members.len();
        }
        let g = g.inverse;
        for &e in members {
            resource::checkpoint()?;
            let rows = &groups[e];
            // G X_i and G y_i
            for a in 0..m {
                let mut sy = 0.0;
                for l in 0..k {
                    gx[a * k + l] = 0.0;
                }
                for (b, &rb) in rows.iter().enumerate() {
                    let gab = g[[a, b]];
                    sy += gab * yt[rb];
                    for l in 0..k {
                        gx[a * k + l] += gab * xt[[rb, l]];
                    }
                }
                gy[a] = sy;
            }
            // X_i' G X_i and X_i' G y_i
            for (a, &ra) in rows.iter().enumerate() {
                for j in 0..k {
                    let xaj = xt[[ra, j]];
                    xtoy[j] += xaj * gy[a];
                    for l in 0..k {
                        xtox[[j, l]] += xaj * gx[a * k + l];
                    }
                }
            }
        }
    }
    if indefinite_entities > 0 {
        warnings.push(format!(
            "panel GLS: the error covariance averaged over the entities observed in each pair \
             of periods is not positive semidefinite (an unbalanced panel); the non-positive \
             pivots of {indefinite_entities} entities' blocks were dropped (generalized \
             Cholesky, as plm::pggls)"
        ));
    }

    // Solve for GLS coefficients
    let xtox_s = sym_inverse(
        xtox.view(),
        SingularPolicy::Error,
        "X'Omega^-1 X in panel GLS",
    )?;
    let xtox_inv = xtox_s.inverse;

    if xtox_s.reciprocal_condition < 1e-10 {
        warnings.push(
            "High condition number detected, results may be numerically unstable".to_string(),
        );
    }

    let beta: Array1<f64> = xtox_inv.dot(&xtoy);

    // Residuals of the transformed model (pggls's residuals)
    let residuals = &yt - &xt.dot(&beta);
    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    // R-squared against the total sum of squares of the raw outcome, as pggls
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    let df_residual =
        n_t.checked_sub(k)
            .filter(|&d| d > 0)
            .ok_or_else(|| EconError::InsufficientData {
                required: k + 1,
                provided: n_t,
                context: format!("panel GLS with {k} coefficients"),
            })?;
    let sigma = (rss / df_residual as f64).sqrt();

    // Standard errors from (X'Omega^-1 X)^-1, with no residual-variance factor (Omega
    // already carries the scale)
    let std_errors: Vec<f64> = (0..k).map(|j| xtox_inv[[j, j]].sqrt()).collect();

    // z statistics and normal p-values (summary.pggls)
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| {
            if se.is_finite() && se > 0.0 {
                b / se
            } else {
                f64::NAN
            }
        })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&z| normal_p_value_two_sided(z))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    let variables: Vec<String> = dm.column_names.clone();
    let se_arr: Array1<f64> = Array1::from_vec(std_errors.clone());

    Ok(PanelGlsResult {
        model,
        dep_var: y_col.to_string(),
        variables,
        coefficients: beta.to_vec(),
        std_errors,
        t_stats,
        p_values,
        significance,
        r_squared,
        n_obs: n_t,
        n_groups,
        n_periods,
        df_residual,
        rss,
        sigma,
        error_cov_dim: t_dim,
        warnings,
        coef_arr: beta,
        se_arr,
        residuals,
        vcov: xtox_inv,
        missing,
    })
}

/// A generalized-Cholesky inverse with the rank it kept and whether a pivot was
/// clearly negative.
struct GcholInverse {
    inverse: Array2<f64>,
    rank: usize,
    negative: bool,
}

/// Generalized inverse of a symmetric matrix by the generalized Cholesky of
/// `bdsmatrix` (used by `plm::pggls`): LDL' in the given order with no pivoting, a
/// pivot below `eps` set to zero together with its column of L, and
/// `G = L^-T D^+ L^-1`. `negative` is set when a pivot is below `-8 eps` (survival's
/// `cholesky2` flag). On a positive-definite matrix this is the inverse.
fn gchol_inverse(a: ArrayView2<f64>, eps: f64) -> EconResult<GcholInverse> {
    let m = a.nrows();
    let what = "panel GLS generalized Cholesky";
    let mut l = resource::try_zeros2(m, m, what)?;
    for i in 0..m {
        for j in 0..=i {
            l[[i, j]] = a[[i, j]];
        }
    }
    let mut d = vec![0.0; m];
    let mut rank = 0;
    let mut negative = false;
    for i in 0..m {
        let piv = l[[i, i]];
        if piv.is_nan() || piv < eps {
            if piv.is_nan() || piv < -8.0 * eps {
                negative = true;
            }
            for j in i + 1..m {
                l[[j, i]] = 0.0;
            }
        } else {
            rank += 1;
            d[i] = piv;
            for j in i + 1..m {
                let tmp = l[[j, i]] / piv;
                l[[j, j]] -= tmp * tmp * piv;
                for r in j + 1..m {
                    l[[r, j]] -= tmp * l[[r, i]];
                }
                l[[j, i]] = tmp;
            }
        }
        l[[i, i]] = 1.0;
    }
    // L^-1 (unit lower triangular), by forward substitution
    let mut linv = resource::try_zeros2(m, m, what)?;
    for i in 0..m {
        linv[[i, i]] = 1.0;
        for j in 0..i {
            let mut s = 0.0;
            for r in j..i {
                s -= l[[i, r]] * linv[[r, j]];
            }
            linv[[i, j]] = s;
        }
    }
    // G = L^-T D^+ L^-1
    let mut g = resource::try_zeros2(m, m, what)?;
    for r in 0..m {
        if d[r] == 0.0 {
            continue;
        }
        let w = 1.0 / d[r];
        for p in 0..=r {
            let lp = linv[[r, p]] * w;
            if lp == 0.0 {
                continue;
            }
            for q in 0..=r {
                g[[p, q]] += lp * linv[[r, q]];
            }
        }
    }
    Ok(GcholInverse {
        inverse: g,
        rank,
        negative,
    })
}

/// Convenience function for Panel GLS with fixed effects (FEGLS).
///
/// # Errors
///
/// Refuses T > N, as [`run_panel_gls`] does.
pub fn run_fegls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<PanelGlsResult> {
    run_panel_gls(
        dataset,
        y_col,
        x_cols,
        entity_col,
        time_col,
        Some(PanelGlsModel::FixedEffects),
    )
}

/// Convenience function for Pooled GLS.
///
/// # Errors
///
/// Refuses T > N, as [`run_panel_gls`] does.
pub fn run_pooled_gls(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    time_col: &str,
) -> EconResult<PanelGlsResult> {
    run_panel_gls(
        dataset,
        y_col,
        x_cols,
        entity_col,
        time_col,
        Some(PanelGlsModel::Pooling),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource::{self, Budget, CancelToken};

    const GRUNFELD_CSV: &str = include_str!("../../../../../validation/datasets/grunfeld.csv");

    /// Grunfeld with years up to `last_year` (firm, year, inv, value, capital).
    fn grunfeld_until(last_year: i64) -> Dataset {
        let csv: String = GRUNFELD_CSV
            .lines()
            .enumerate()
            .filter(|(i, line)| {
                *i == 0
                    || line
                        .split(',')
                        .nth(1)
                        .and_then(|y| y.trim().parse::<i64>().ok())
                        .is_some_and(|y| y <= last_year)
            })
            .map(|(_, line)| format!("{line}\n"))
            .collect();
        Dataset::new(crate::data::DataLoader::from_csv_string(&csv).unwrap())
    }

    /// Pinned-value check. A wrong or empty `expected` prints the actual values to paste.
    fn assert_pinned(what: &str, actual: &[f64], expected: &[f64]) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{what}: pin these values: {actual:?}"
        );
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1e-3),
                "{what}[{i}]: got {a}, pinned {e}; all: {actual:?}"
            );
        }
    }

    fn cancelled() -> Budget {
        let t = CancelToken::new();
        t.cancel();
        Budget::new(None, t)
    }

    /// A balanced synthetic panel of `n_ent` entities and `n_t` periods.
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
    fn test_panel_gls_refuses_more_periods_than_entities() {
        let e = run_panel_gls(
            &synthetic_panel(3, 6),
            "y",
            &["x"],
            "entity",
            "time",
            Some(PanelGlsModel::Pooling),
        )
        .unwrap_err();
        match e {
            EconError::InvalidSpecification { message } => {
                assert!(
                    message.contains("N = 3") && message.contains("T = 6"),
                    "{message}"
                )
            }
            e => panic!("{e:?}"),
        }
        // full Grunfeld has N = 10 firms and T = 20 years
        let e = run_fegls(
            &grunfeld_until(1954),
            "inv",
            &["value", "capital"],
            "firm",
            "year",
        )
        .unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message } if message.contains("T = 20")),
            "{e:?}"
        );
        assert!(
            run_panel_gls(
                &synthetic_panel(5, 5),
                "y",
                &["x"],
                "entity",
                "time",
                Some(PanelGlsModel::Pooling)
            )
            .is_ok()
        );
    }

    #[test]
    fn test_panel_gls_covariance_respects_alloc_ceiling() {
        // T = 60: 60*60*6*8 = 172,800 bytes > 64 KiB
        let ds = synthetic_panel(60, 60);
        let e = resource::with_alloc_ceiling(Some(64 << 10), || {
            run_panel_gls(
                &ds,
                "y",
                &["x"],
                "entity",
                "time",
                Some(PanelGlsModel::Pooling),
            )
        })
        .unwrap_err();
        assert!(matches!(e, EconError::ResourceLimit(_)), "{e:?}");
    }

    #[test]
    fn test_panel_gls_stops_when_cancelled() {
        let ds = synthetic_panel(10, 4);
        let e = resource::with_budget(cancelled(), || {
            run_panel_gls(&ds, "y", &["x"], "entity", "time", None)
        })
        .unwrap_err();
        assert!(matches!(e, EconError::Cancelled(_)), "{e:?}");
    }

    #[test]
    fn test_panel_gls_scaling_many_entities() {
        if cfg!(debug_assertions) {
            return; // timing is checked only in release builds
        }
        let ds = synthetic_panel(100_000, 5);
        let t0 = std::time::Instant::now();
        let r = run_panel_gls(
            &ds,
            "y",
            &["x"],
            "entity",
            "time",
            Some(PanelGlsModel::Pooling),
        )
        .unwrap();
        assert_eq!(r.n_groups, 100_000);
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(10),
            "took {:?}: per-entity work must be linear",
            t0.elapsed()
        );
    }

    /// Pooled GLS on Grunfeld 1935-1944 (N = T = 10) vs `plm::pggls` (plm 2.6.7),
    /// quoted:
    ///
    /// ```r
    /// g <- read.csv("validation/datasets/grunfeld.csv")
    /// d <- pdata.frame(subset(g, year <= 1944), index = c("firm", "year"))
    /// m <- pggls(inv ~ value + capital, data = d, model = "pooling")
    /// coef(m)
    /// # -13.7465321982 0.0959140825 0.1596389388
    /// sqrt(diag(vcov(m)))
    /// # 1.53909079042087326705 0.00082839098966173616 0.00455253946084683073
    /// ```
    #[test]
    fn test_validate_pooled_gls_square_grunfeld_vs_plm() {
        let r = run_pooled_gls(
            &grunfeld_until(1944),
            "inv",
            &["value", "capital"],
            "firm",
            "year",
        )
        .unwrap();
        assert_eq!((r.n_groups, r.n_periods), (10, 10));
        let plm = [-13.7465321982, 0.0959140825, 0.1596389388];
        for (got, want) in r.coefficients.iter().zip(plm) {
            assert!(
                (got - want).abs() < 1e-8,
                "{:?} vs plm {plm:?}",
                r.coefficients
            );
        }
        let plm_se = [
            1.539_090_790_420_873_3,
            0.000_828_390_989_661_736_2,
            0.004_552_539_460_846_830_7,
        ];
        for (got, want) in r.std_errors.iter().zip(plm_se) {
            assert!(
                (got - want).abs() <= 1e-8 * want,
                "{:?} vs plm {plm_se:?}",
                r.std_errors
            );
        }
    }

    /// The rows of `df` whose `year` is missing or at most 1944 (N = T = 10 after
    /// deletion, as T0's pinned test).
    fn early(df: &polars::prelude::DataFrame) -> polars::prelude::DataFrame {
        use polars::prelude::{BooleanChunked, NewChunkedArray};
        let years = crate::data::columns::numeric_options(df, "year").unwrap();
        let mask: Vec<bool> = years
            .iter()
            .map(|y| y.is_none_or(|y| y <= 1944.0))
            .collect();
        df.filter(&BooleanChunked::from_slice("keep".into(), &mask))
            .unwrap()
    }

    /// Two fits agree to 1e-12 (coefficients, SEs, RSS and R-squared).
    fn assert_same_gls(a: &PanelGlsResult, b: &PanelGlsResult) {
        let tol = p2a_validation::Tol::new(1e-12, 1e-12);
        assert_eq!(a.variables, b.variables);
        let va = a
            .coefficients
            .iter()
            .chain(&a.std_errors)
            .chain([&a.rss, &a.r_squared]);
        let vb = b
            .coefficients
            .iter()
            .chain(&b.std_errors)
            .chain([&b.rss, &b.r_squared]);
        for (x, y) in va.zip(vb) {
            assert!(tol.allows(*x, *y), "{x} vs {y}");
        }
    }

    const MODELS: [PanelGlsModel; 3] = [
        PanelGlsModel::FixedEffects,
        PanelGlsModel::Pooling,
        PanelGlsModel::FirstDifference,
    ];

    /// Panel GLS drops rows with a missing outcome or time (plm na.omit) and equals
    /// the fit on the complete rows. Before, the null outcomes were read through
    /// `into_no_null_iter` and the missing years became period 0.
    #[test]
    fn test_validate_na_panel_gls_omit_equals_complete_rows() {
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        assert_eq!(
            case.dataset,
            "validation/datasets/generated/grunfeld_na.csv"
        );
        let df = early(ds.df());
        assert_eq!(df.height(), 101);
        let complete = Dataset::new(df.drop_nulls::<String>(None).unwrap());
        let ds = Dataset::new(df);
        let note = "4 of 101 rows dropped because of missing values (inv: 2, year: 2)";
        for m in MODELS {
            let fit = |d: &Dataset| {
                run_panel_gls(
                    d,
                    "inv",
                    &["value", "capital"],
                    "firm_name",
                    "year",
                    Some(m),
                )
                .unwrap_or_else(|e| panic!("{m:?}: {e}"))
            };
            let r = fit(&ds);
            assert_same_gls(&r, &fit(&complete));
            // first differences: 97 rows less one per entity
            let n_obs = if m == PanelGlsModel::FirstDifference {
                87
            } else {
                97
            };
            assert_eq!((r.n_obs, r.n_groups, r.n_periods), (n_obs, 10, 10), "{m:?}");
            assert_eq!(r.missing.note().as_deref(), Some(note), "{m:?}");
            assert!(r.warnings.iter().any(|w| w == note), "{m:?}");
            let text = r.to_string();
            assert!(
                text.trim_end()
                    .ends_with(&format!("Missing values: {note}")),
                "{text}"
            );
        }
    }

    /// The rows whose text firm id is missing (grunfeld_na.csv rows 77 and 133) are
    /// dropped, not fitted as an 11th entity "".
    #[test]
    fn test_validate_na_panel_gls_missing_firm_id() {
        let (_case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        let null_id = ds.df().column("firm_name").unwrap().is_null();
        let mut df = early(ds.df());
        df.vstack_mut(&ds.df().filter(&null_id).unwrap()).unwrap();
        assert_eq!(df.height(), 103);
        let complete = Dataset::new(df.drop_nulls::<String>(None).unwrap());
        let ds = Dataset::new(df);
        let note = "6 of 103 rows dropped because of missing values (inv: 2, firm_name: 2, \
                    year: 2)";
        for m in MODELS {
            let fit = |d: &Dataset| {
                run_panel_gls(
                    d,
                    "inv",
                    &["value", "capital"],
                    "firm_name",
                    "year",
                    Some(m),
                )
                .unwrap_or_else(|e| panic!("{m:?}: {e}"))
            };
            let r = fit(&ds);
            assert_same_gls(&r, &fit(&complete));
            assert_eq!(r.n_groups, 10, "{m:?}");
            assert_eq!(r.missing.note().as_deref(), Some(note), "{m:?}");
        }
    }

    /// Integer outcomes and times of any integer width are read as numbers; a
    /// fractional time is an error; the errors name the column.
    #[test]
    fn test_validate_na_panel_gls_integer_outcome_and_time() {
        use crate::data::columns::{NaPolicy, numeric_column};
        use polars::prelude::{Column, DataType};
        let (case, _) = crate::test_support::load_case("panel_grunfeld_na");
        let path = p2a_validation::repo_path(&case.datasets[0]);
        let g = crate::data::DataLoader::load_csv(&path).unwrap();
        let mut df = early(&g);
        assert_eq!(df.height(), 100);
        let inv = numeric_column(&df, "inv", NaPolicy::Fail).unwrap().0;
        let inv_int: Vec<i64> = inv.iter().map(|v| v.round() as i64).collect();
        let inv_f: Vec<f64> = inv_int.iter().map(|&v| v as f64).collect();
        df.with_column(Column::new("inv_int".into(), inv_int))
            .unwrap();
        df.with_column(Column::new("inv_f".into(), inv_f)).unwrap();
        let ds = Dataset::new(df.clone());
        let fit = |d: &Dataset, y: &str, m| {
            run_panel_gls(d, y, &["value", "capital"], "firm", "year", Some(m))
        };
        for m in MODELS {
            let a = fit(&ds, "inv_int", m).unwrap_or_else(|e| panic!("{m:?}: {e}"));
            assert_same_gls(&a, &fit(&ds, "inv_f", m).unwrap());
        }

        let year32 = df.column("year").unwrap().cast(&DataType::Int32).unwrap();
        let mut df32 = df.clone();
        df32.with_column(year32).unwrap();
        for m in MODELS {
            let a =
                fit(&Dataset::new(df32.clone()), "inv", m).unwrap_or_else(|e| panic!("{m:?}: {e}"));
            assert_same_gls(&a, &fit(&ds, "inv", m).unwrap());
        }

        let mut years = numeric_column(&df, "year", NaPolicy::Fail).unwrap().0;
        years[0] = 1935.25;
        let mut frac = df.clone();
        frac.with_column(Column::new("year".into(), years)).unwrap();
        match fit(&Dataset::new(frac), "inv", PanelGlsModel::Pooling) {
            Err(EconError::InvalidSpecification { message }) => {
                assert!(message.contains("whole numbers"), "{message}")
            }
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }

        match fit(&ds, "nope", PanelGlsModel::Pooling) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        let (_, na) = crate::test_support::load_case("panel_grunfeld_na");
        match fit(&na, "firm_name", PanelGlsModel::Pooling) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "firm_name"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
    }

    /// The within residuals sum to zero in every entity, so the T x T error covariance has
    /// rank T - 1: its generalized inverse is reported.
    #[test]
    fn test_validate_panel_gls_within_reports_singular_error_covariance() {
        let (_case, ds) = crate::test_support::load_case("pggls_sim");
        let r = run_fegls(&ds, "y", &["x1", "x2"], "id", "t").unwrap();
        assert!(
            r.warnings.iter().any(|w| w
                == "panel GLS error covariance (T x T) is singular (rank 4 of 5); a \
                    generalized inverse was used"),
            "{:?}",
            r.warnings
        );
        assert!(r.to_string().contains("rank 4 of 5"));
        let json = serde_json::to_value(&r).unwrap();
        assert!(json["warnings"].as_array().is_some_and(|w| !w.is_empty()));
    }

    /// The p-values are normal: `2 * pnorm(-|coef / se|)` of the fitted values.
    fn assert_normal_p_values(r: &PanelGlsResult) {
        use crate::traits::estimator::normal_p_value_two_sided;
        for j in 0..r.variables.len() {
            let z = r.coefficients[j] / r.std_errors[j];
            assert!(
                (r.t_stats[j] - z).abs() <= 1e-12 * z.abs(),
                "{:?}",
                r.t_stats
            );
            let p = normal_p_value_two_sided(z);
            assert!(
                (r.p_values[j] - p).abs() <= 1e-12 * p.max(1e-300),
                "{:?}",
                r.p_values
            );
        }
    }

    /// plm::pggls(model = "pooling") on pggls_sim (N = 40, T = 5).
    #[test]
    fn test_validate_pggls_pooling_sim() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pggls_sim");
        let r = run_pooled_gls(&ds, "y", &["x1", "x2"], "id", "t").unwrap();
        Check::new(&case)
            .named(
                "pooling_coef",
                &r.variables,
                &r.coefficients,
                Tol::coef(200),
            )
            .named("pooling_se", &r.variables, &r.std_errors, Tol::se(200))
            .scalar("pooling_rss", r.rss, Tol::stat(200))
            .scalar("pooling_r2", r.r_squared, Tol::stat(200))
            .scalar("pooling_df_resid", r.df_residual as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!((r.n_obs, r.n_groups, r.n_periods), (200, 40, 5));
        assert_normal_p_values(&r);
    }

    /// plm::pggls(model = "within") on pggls_sim: GLS on the entity-demeaned model.
    #[test]
    fn test_validate_pggls_within_sim() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pggls_sim");
        let r = run_fegls(&ds, "y", &["x1", "x2"], "id", "t").unwrap();
        Check::new(&case)
            .named("within_coef", &r.variables, &r.coefficients, Tol::coef(200))
            .named("within_se", &r.variables, &r.std_errors, Tol::se(200))
            .scalar("within_rss", r.rss, Tol::stat(200))
            .scalar("within_r2", r.r_squared, Tol::stat(200))
            .scalar("within_df_resid", r.df_residual as f64, Tol::exact())
            .finish();
        assert_eq!(r.n_obs, 200);
        assert_normal_p_values(&r);
    }

    /// plm::pggls(model = "fd") on pggls_sim: GLS on the differenced model with an
    /// intercept column of ones (160 differenced rows).
    #[test]
    fn test_validate_pggls_fd_sim() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pggls_sim");
        let r = run_panel_gls(
            &ds,
            "y",
            &["x1", "x2"],
            "id",
            "t",
            Some(PanelGlsModel::FirstDifference),
        )
        .unwrap();
        Check::new(&case)
            .named("fd_coef", &r.variables, &r.coefficients, Tol::coef(200))
            .named("fd_se", &r.variables, &r.std_errors, Tol::se(200))
            .scalar("fd_rss", r.rss, Tol::stat(200))
            .scalar("fd_r2", r.r_squared, Tol::stat(200))
            .scalar("fd_df_resid", r.df_residual as f64, Tol::exact())
            .finish();
        assert_eq!(r.df_residual, 157);
        assert_eq!(r.n_obs, 160);
        assert_eq!(r.residuals.len(), 160);
        assert!(
            !r.warnings.iter().any(|w| w.starts_with("first-step OLS:")),
            "{:?}",
            r.warnings
        );
        assert_normal_p_values(&r);
    }

    /// One model of pggls_unbalanced against its R pins (coefficients, SEs, z, p, df,
    /// RSS, R-squared, observations).
    fn check_unbalanced(model: PanelGlsModel, prefix: &str) -> PanelGlsResult {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pggls_unbalanced");
        let r = run_panel_gls(&ds, "y", &["x1", "x2"], "id", "t", Some(model)).unwrap();
        let key = |s: &str| format!("{prefix}_{s}");
        Check::new(&case)
            .named(&key("coef"), &r.variables, &r.coefficients, Tol::coef(212))
            .named(&key("se"), &r.variables, &r.std_errors, Tol::se(212))
            .named(&key("z"), &r.variables, &r.t_stats, Tol::stat(212))
            .named(&key("p"), &r.variables, &r.p_values, Tol::p_value(212))
            .scalar(&key("df_resid"), r.df_residual as f64, Tol::exact())
            .scalar(&key("rss"), r.rss, Tol::stat(212))
            .scalar(&key("r2"), r.r_squared, Tol::stat(212))
            .scalar(&key("nobs"), r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!((r.n_groups, r.n_periods), (40, 6));
        r
    }

    /// plm::pggls(model = "pooling") on an unbalanced panel: each entity is weighted by
    /// the inverse of its own observed-period block of Omega.
    #[test]
    fn test_validate_pggls_unbalanced_pooling() {
        let r = check_unbalanced(PanelGlsModel::Pooling, "pooling");
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains("not positive semidefinite")),
            "{:?}",
            r.warnings
        );
    }

    /// plm::pggls(model = "fd") on an unbalanced panel: gaps are differenced across, each
    /// differenced row takes the period of its later row.
    #[test]
    fn test_validate_pggls_unbalanced_fd() {
        let r = check_unbalanced(PanelGlsModel::FirstDifference, "fd");
        assert_eq!(r.error_cov_dim, 5);
    }

    /// plm::pggls(model = "within") on an unbalanced panel: the cell-averaged Omega is
    /// indefinite, and the generalized Cholesky drops the non-positive pivots.
    #[test]
    fn test_validate_pggls_unbalanced_within() {
        let r = check_unbalanced(PanelGlsModel::FixedEffects, "within");
        assert!(
            r.warnings.iter().any(|w| w.starts_with(
                "panel GLS: the error covariance averaged over the entities observed in each \
                 pair of periods is not positive semidefinite (an unbalanced panel); the \
                 non-positive pivots of "
            ) && w
                .ends_with(" entities' blocks were dropped (generalized Cholesky, as plm::pggls)")),
            "{:?}",
            r.warnings
        );
    }

    /// Two rows of one entity in one period are refused, naming both.
    #[test]
    fn test_panel_gls_refuses_duplicate_entity_period() {
        let base = synthetic_panel(6, 4);
        let dup = base.df().slice(9, 1); // entity 2, period 2001
        let mut df = base.df().clone();
        df.vstack_mut(&dup).unwrap();
        for m in MODELS {
            match run_panel_gls(
                &Dataset::new(df.clone()),
                "y",
                &["x"],
                "entity",
                "time",
                Some(m),
            ) {
                Err(EconError::InvalidSpecification { message }) => assert!(
                    message.contains("'2'") && message.contains("2001"),
                    "{m:?}: {message}"
                ),
                other => panic!("{m:?}: expected InvalidSpecification, got {other:?}"),
            }
        }
    }

    /// The generalized-Cholesky inverse: the plain inverse on a positive-definite
    /// matrix, a generalized inverse (A G A = A) with one pivot dropped on a rank-3
    /// PSD matrix, and a negative pivot reported on an indefinite matrix.
    #[test]
    fn test_panel_gls_block_inverse_matches_dense_on_pd() {
        let b = ndarray::array![
            [1.3, -0.2, 0.7],
            [0.4, 1.1, -0.5],
            [-0.8, 0.3, 0.9],
            [0.2, -0.6, 0.35]
        ];
        let close = |a: &Array2<f64>, b: &Array2<f64>| {
            a.iter()
                .zip(b)
                .all(|(x, y)| (x - y).abs() <= 1e-12 * (1.0 + y.abs()))
        };
        // positive definite: B B' + diag
        let mut pd = b.dot(&b.t());
        for i in 0..4 {
            pd[[i, i]] += 0.5 + 0.1 * i as f64;
        }
        let eps = |a: &Array2<f64>| 1e-10 * a.diag().iter().cloned().fold(0.0, f64::max);
        let g = gchol_inverse(pd.view(), eps(&pd)).unwrap();
        let dense = sym_inverse(pd.view(), SingularPolicy::Error, "pd")
            .unwrap()
            .inverse;
        assert_eq!((g.rank, g.negative), (4, false));
        assert!(close(&g.inverse, &dense), "{:?} vs {dense:?}", g.inverse);

        // rank 3 PSD
        let psd = b.dot(&b.t());
        let g = gchol_inverse(psd.view(), eps(&psd)).unwrap();
        assert_eq!((g.rank, g.negative), (3, false));
        let aga = psd.dot(&g.inverse).dot(&psd);
        assert!(close(&aga, &psd), "{aga:?} vs {psd:?}");

        // indefinite
        let mut ind = pd.clone();
        ind[[3, 3]] -= 10.0;
        let g = gchol_inverse(ind.view(), eps(&ind)).unwrap();
        assert!(g.negative);
        assert_eq!(g.rank, 3);
    }
}
