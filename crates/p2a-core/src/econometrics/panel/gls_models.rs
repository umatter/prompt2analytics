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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::resource;
use crate::traits::convergence::ResultNotes;
use crate::traits::estimator::{SignificanceLevel, t_test_p_value};

use super::utils::{demean_by_entity, entity_codes, time_codes};

/// `sym_inverse` holds several m x m copies at once (copy, equilibration, eigenvectors,
/// solution). Check the working set against the allocation ceiling before building it.
const DENSE_INVERSE_COPIES: usize = 6;

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
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values (two-sided)
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// R-squared (based on transformed model)
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities (cross-sectional units)
    pub n_groups: usize,
    /// Number of time periods
    pub n_periods: usize,
    /// Degrees of freedom
    pub df_residual: usize,
    /// Residual sum of squares
    pub rss: f64,
    /// Residual standard error
    pub sigma: f64,
    /// Estimated error covariance matrix dimension (T x T)
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
            "Variable", "Coefficient", "Std.Error", "t-stat", "P>|t|"
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
/// Panel GLS estimates regression coefficients while accounting for cross-sectional
/// correlation and heteroskedasticity in the error terms. The method:
/// 1. Estimates initial model (OLS, FE, or FD depending on model type)
/// 2. Uses residuals to estimate the error covariance matrix
/// 3. Re-estimates using GLS with the estimated covariance
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
/// from N residual vectors and has no inverse.
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

    // Rows of each entity (in row order), built once; each row's time index is
    // `row_time_idx`
    let mut rows_by_entity: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (i, &entity_id) in entity_ids.iter().enumerate() {
        rows_by_entity[entity_id].push(i);
    }

    // Build design matrix (includes intercept for non-FE models)
    let has_intercept = model != PanelGlsModel::FixedEffects;
    let dm = cols.matrix(x_cols, has_intercept)?;
    let x = &dm.data;
    let k = x.ncols();

    // Step 1: Initial estimation based on model type
    let initial_resid = match model {
        PanelGlsModel::FixedEffects => {
            // Within transformation (demean by entity)
            let y_arr = Array1::from(y.to_vec());
            let y_demeaned = demean_by_entity(&y_arr, &entity_ids, n_groups);

            let mut x_demeaned = Array2::<f64>::zeros((n, k));
            for j in 0..k {
                let col = x.column(j).to_owned();
                let demeaned = demean_by_entity(&col, &entity_ids, n_groups);
                for i in 0..n {
                    x_demeaned[[i, j]] = demeaned[i];
                }
            }

            // OLS on demeaned data
            first_step_residuals(
                x_demeaned.view(),
                y_demeaned.view(),
                &dm.column_names,
                &mut warnings,
            )?
        }
        PanelGlsModel::Pooling => {
            // Simple OLS
            first_step_residuals(
                x.view(),
                ArrayView1::from(y),
                &dm.column_names,
                &mut warnings,
            )?
        }
        PanelGlsModel::FirstDifference => {
            // First-difference transformation
            warnings.push("First-difference GLS drops first observation per entity".to_string());

            // Sort data by entity and time for differencing. The order is by the
            // entity label as a string, as before the shared extractor, so the
            // first-difference residual alignment (ECON-33, Tier 4) and T0's
            // pinned FD outputs are unchanged.
            let entity_strings: Vec<&str> = (0..n).map(|i| entity.label(i)).collect();
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by_key(|&i| (entity_strings[i], time_values[i]));

            // Compute first differences
            let mut y_diff = Vec::new();
            let mut x_diff = Vec::new();

            let mut prev_entity: Option<&str> = None;
            let mut prev_y = 0.0;
            let mut prev_x = vec![0.0; k];

            for &idx in &sorted_indices {
                let curr_entity = entity_strings[idx];

                if prev_entity == Some(curr_entity) {
                    // Same entity, compute difference
                    y_diff.push(y[idx] - prev_y);
                    let mut row = Vec::with_capacity(k);
                    for j in 0..k {
                        row.push(x[[idx, j]] - prev_x[j]);
                    }
                    x_diff.push(row);
                }

                prev_entity = Some(curr_entity);
                prev_y = y[idx];
                prev_x = (0..k).map(|j| x[[idx, j]]).collect();
            }

            if y_diff.is_empty() {
                return Err(EconError::InsufficientData {
                    required: 2,
                    provided: 1,
                    context: "First-difference requires at least 2 time periods per entity"
                        .to_string(),
                });
            }

            let n_diff = y_diff.len();
            let y_diff_arr = Array1::from(y_diff);
            let mut x_diff_arr = Array2::<f64>::zeros((n_diff, k));
            for (i, row) in x_diff.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    x_diff_arr[[i, j]] = val;
                }
            }

            // OLS on differenced data (the differenced intercept is zero: aliased)
            first_step_residuals(
                x_diff_arr.view(),
                y_diff_arr.view(),
                &dm.column_names,
                &mut warnings,
            )?
        }
    };

    // Step 2: Estimate error covariance matrix
    // We estimate a T x T covariance matrix from cross-sectional residuals
    // Omega[s,t] = (1/N) * sum_i(e_is * e_it)

    // Build T x T omega matrix
    let t_dim = n_periods;
    let omega_what = "panel GLS T x T error covariance";
    ensure_dense_inverse_fits(t_dim, omega_what)?;
    let mut omega = resource::try_zeros2(t_dim, t_dim, omega_what)?;
    let mut omega_counts: Vec<usize> = resource::try_vec(
        resource::checked_len(&[t_dim, t_dim], omega_what)?,
        0,
        omega_what,
    )?;

    for rows in &rows_by_entity {
        resource::checkpoint()?;
        // Rows past the end of the residuals are skipped (the first-difference
        // residuals are shorter than the data)
        let rows = &rows[..rows.partition_point(|&i| i < initial_resid.len())];

        for (a, &i) in rows.iter().enumerate() {
            if a % 256 == 255 {
                resource::checkpoint()?;
            }
            let s_idx = row_time_idx[i];
            for &j in rows {
                let t_idx = row_time_idx[j];
                omega[[s_idx, t_idx]] += initial_resid[i] * initial_resid[j];
                omega_counts[s_idx * t_dim + t_idx] += 1;
            }
        }
    }

    // Average the covariance estimates
    for s in 0..t_dim {
        for t in 0..t_dim {
            let count = omega_counts[s * t_dim + t];
            if count > 0 {
                omega[[s, t]] /= count as f64;
            }
        }
    }

    // A singular Omega (the within residuals sum to zero in every entity, so its
    // rank is T - 1) gets a generalized inverse, reported with its rank.
    let omega_what = "panel GLS error covariance (T x T)";
    let omega_s = sym_inverse(omega.view(), SingularPolicy::Generalized, omega_what)?;
    if let Some(w) = omega_s.warning(omega_what) {
        warnings.push(w);
    }
    let omega_inv = omega_s.inverse;

    // Step 3: GLS estimation
    let y_arr = Array1::from(y.to_vec());

    // Compute GLS estimates using the formula:
    // beta_GLS = (X' * Omega_block^{-1} * X)^{-1} * (X' * Omega_block^{-1} * y)

    let mut xtox = Array2::<f64>::zeros((k, k));
    let mut xtoy = Array1::<f64>::zeros(k);

    for entity_indices in &rows_by_entity {
        resource::checkpoint()?;
        let ti = entity_indices.len();
        if ti == 0 {
            continue;
        }

        // Extract submatrices for this entity
        let mut x_i = Array2::<f64>::zeros((ti, k));
        let mut y_i = Array1::<f64>::zeros(ti);

        for (local_idx, &global_idx) in entity_indices.iter().enumerate() {
            y_i[local_idx] = y_arr[global_idx];
            for j in 0..k {
                x_i[[local_idx, j]] = x[[global_idx, j]];
            }
        }

        // Extract relevant part of Omega^{-1} for this entity's time periods
        let mut omega_inv_i = resource::try_zeros2(ti, ti, "panel GLS per-entity covariance")?;
        for (s, &row_s) in entity_indices.iter().enumerate() {
            let s_idx = row_time_idx[row_s];
            for (t, &row_t) in entity_indices.iter().enumerate() {
                omega_inv_i[[s, t]] = omega_inv[[s_idx, row_time_idx[row_t]]];
            }
        }

        // Accumulate X'Omega^{-1}X and X'Omega^{-1}y
        let xto = x_i.t().dot(&omega_inv_i);
        xtox = &xtox + &xto.dot(&x_i);
        xtoy = &xtoy + &xto.dot(&y_i);
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

    // Compute residuals and diagnostics
    let fitted = x.dot(&beta);
    let residuals = &y_arr - &fitted;

    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    // The mean once, not once per observation (that made this loop O(n^2))
    let y_mean = y_arr.mean().unwrap_or(0.0);
    let tss: f64 = y_arr.iter().map(|yi| (yi - y_mean).powi(2)).sum();

    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

    let df_residual =
        n.checked_sub(k)
            .filter(|&d| d > 0)
            .ok_or_else(|| EconError::InsufficientData {
                required: k + 1,
                provided: n,
                context: format!("panel GLS with {k} coefficients"),
            })?;
    let sigma = (rss / df_residual as f64).sqrt();

    // Standard errors from (X'Omega^{-1}X)^{-1}
    let se_scale = rss / df_residual as f64;
    let std_errors: Vec<f64> = (0..k)
        .map(|j| (xtox_inv[[j, j]] * se_scale).abs().sqrt())
        .collect();

    // T-statistics and p-values
    let t_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|&t| t_test_p_value(t, df_residual as f64))
        .collect();

    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Variable names ("(Intercept)" first for the models with an intercept)
    let variables: Vec<String> = dm.column_names.clone();

    let coef_arr: Array1<f64> = beta.clone();
    let se_arr: Array1<f64> = Array1::from_vec(std_errors.clone());
    // vcov ≈ se_scale * (X'Ω⁻¹X)⁻¹  (the diagonal of which the SEs are derived from)
    let vcov: ndarray::Array2<f64> = &xtox_inv * se_scale;

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
        n_obs: n,
        n_groups,
        n_periods,
        df_residual,
        rss,
        sigma,
        error_cov_dim: t_dim,
        warnings,
        coef_arr,
        se_arr,
        residuals,
        vcov,
        missing,
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

    /// Pooled GLS on Grunfeld 1935-1944 (N = T = 10) vs `plm::pggls`
    /// (plm 2.6.7), which agrees with the pre-T0-WP14 code to all printed digits:
    ///
    /// ```r
    /// g <- read.csv("validation/datasets/grunfeld.csv")
    /// d <- pdata.frame(subset(g, year <= 1944), index = c("firm", "year"))
    /// coef(pggls(inv ~ value + capital, data = d, model = "pooling"))
    /// # -13.7465321982 0.0959140825 0.1596389388
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
        let plm = [-13.7465321982, 0.0959140825, 0.1596389388];
        for (got, want) in r.coefficients.iter().zip(plm) {
            assert!(
                (got - want).abs() < 1e-8,
                "{:?} vs plm {plm:?}",
                r.coefficients
            );
        }
    }

    /// Characterisation test: the code before the T0-WP14 resource bounds gives
    /// the same numbers, and later refactors must not move them. Each fit uses a
    /// panel whose T x T Omega has full rank, so the pinned values do not depend
    /// on rounding-level pivots inside the inverse.
    ///
    /// FixedEffects is not pinned: under the current algorithm its Omega is
    /// singular along the vector of ones (the within residuals sum to zero in
    /// every entity), so its SEs are rounding noise of order 1e-9 (ECON-33,
    /// Tier 4).
    #[test]
    fn test_pinned_panel_gls_outputs() {
        let fit = |ds: &Dataset, model| {
            run_panel_gls(
                ds,
                "inv",
                &["value", "capital"],
                "firm",
                "year",
                Some(model),
            )
            .unwrap()
        };
        let mut v = Vec::new();
        let mut push = |r: &PanelGlsResult| {
            v.extend(&r.coefficients);
            v.extend(&r.std_errors);
            v.push(r.rss);
            v.push(r.r_squared);
        };

        // Pooling on Grunfeld 1935-1944 (N = T = 10): well conditioned, and
        // its coefficients match plm (see the test above).
        let r = fit(&grunfeld_until(1944), PanelGlsModel::Pooling);
        assert_eq!((r.n_groups, r.n_periods), (10, 10));
        push(&r);

        // FirstDifference on Grunfeld 1935-1939 (N = 10, T = 5). The 40
        // differenced residuals are matched to the first 40 data rows, so they
        // cover 8 of the 10 firms (the residual-truncation path); the 5 x 5
        // Omega still has full rank (smallest eigenvalue about 56, condition
        // number about 50).
        let r = fit(&grunfeld_until(1939), PanelGlsModel::FirstDifference);
        assert_eq!((r.n_groups, r.n_periods), (10, 5));
        push(&r);

        assert_pinned(
            "panel GLS",
            &v,
            &[
                -13.746532198161622,
                0.09591408245586196,
                0.15963893877650104,
                110.35059790229461,
                0.05939444350853668,
                0.32641053705590134,
                498646.34093838633,
                0.7644231283694306,
                15.81979032586473,
                0.07743052040539032,
                -0.04783883473444184,
                446.5929154806963,
                0.18013711466451304,
                2.881466704862209,
                228999.72960693543,
                0.6970403766260955,
            ],
        );
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
            assert_eq!((r.n_obs, r.n_groups, r.n_periods), (97, 10, 10), "{m:?}");
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

    /// The differenced intercept is a column of zeros: the first step names it.
    #[test]
    fn test_validate_panel_gls_fd_reports_aliased_first_step() {
        let (_case, ds) = crate::test_support::load_case("pggls_sim");
        let r = run_panel_gls(
            &ds,
            "y",
            &["x1", "x2"],
            "id",
            "t",
            Some(PanelGlsModel::FirstDifference),
        )
        .unwrap();
        assert!(
            r.warnings.iter().any(|w| w
                == "first-step OLS: Coefficients not defined because of singularities: \
                    (Intercept)"),
            "{:?}",
            r.warnings
        );
    }

    /// plm::pggls(model = "pooling") on pggls_sim (N = 40, T = 5).
    #[test]
    fn test_validate_pggls_pooling_sim() {
        use p2a_validation::{Check, Tol};
        let (case, ds) = crate::test_support::load_case("pggls_sim");
        let r = run_pooled_gls(&ds, "y", &["x1", "x2"], "id", "t").unwrap();
        // deviation ECON-33: pooling_se (SEs scaled by rss/df twice)
        Check::new(&case)
            .named(
                "pooling_coef",
                &r.variables,
                &r.coefficients,
                Tol::coef(200),
            )
            .scalar("pooling_rss", r.rss, Tol::stat(200))
            .scalar("pooling_r2", r.r_squared, Tol::stat(200))
            .scalar("pooling_df_resid", r.df_residual as f64, Tol::exact())
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
        assert_eq!((r.n_obs, r.n_groups, r.n_periods), (200, 40, 5));
        // deviation ECON-33: within_*/fd_* coefficients and SEs, fd_df_resid (GLS step
        // ignores the transformation)
        let w = run_fegls(&ds, "y", &["x1", "x2"], "id", "t").unwrap();
        Check::new(&case)
            .scalar("within_df_resid", w.df_residual as f64, Tol::exact())
            .finish();
        assert_eq!(w.n_obs, 200);
    }
}
