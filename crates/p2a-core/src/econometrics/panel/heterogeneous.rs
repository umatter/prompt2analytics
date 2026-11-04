//! Variable Coefficients Models (pvcm) and Mean Group (pmg) estimators.
//!
//! # Mathematical Background
//!
//! For each entity i: y_it = X_it β_i + ε_it
//!
//! For the random model: β_i = β + u_i, where u_i ~ N(0, Δ)
//!
//! The GLS estimator is:
//!   β̂_GLS = (Σ W_i)^(-1) Σ W_i β̂_i
//!   where W_i = [Var(β̂_i) + Δ̂]^(-1)
//!
//! # References
//!
//! - Swamy, P.A.V.B. (1970). Efficient inference in a random coefficient
//!   regression model. *Econometrica*, 38(2), 311-323.
//!
//! - Pesaran, M.H., & Smith, R. (1995). Estimating long-run relationships from
//!   dynamic heterogeneous panels. *Journal of Econometrics*, 68(1), 79-113.
//!
//! R equivalent: `plm::pvcm()`, `plm::pmg()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::data::Dataset;
use crate::data::columns::{ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use crate::linalg::solve::{SingularPolicy, least_squares, sym_inverse};
use crate::resource;
use crate::traits::convergence::{ResultNotes, notes_text};
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, t_test_p_value};

/// Result from a variable coefficients model (pvcm).
///
/// Contains individual-specific coefficients and the overall GLS estimator
/// for the Swamy (1970) random coefficients model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PvcmResult {
    /// Type of pvcm estimation
    pub model_type: PvcmType,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (including intercept)
    pub variables: Vec<String>,
    /// Overall GLS coefficients (for random model)
    pub coefficients: Vec<f64>,
    /// Overall standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Individual-specific coefficients: entity_id -> coefficients
    pub individual_coefficients: HashMap<String, Vec<f64>>,
    /// Individual-specific standard errors
    pub individual_std_errors: HashMap<String, Vec<f64>>,
    /// Estimated variance of random coefficients (Delta matrix diagonal)
    pub delta: Vec<f64>,
    /// Test for coefficient homogeneity (chi-squared statistic)
    pub homogeneity_stat: f64,
    /// p-value for homogeneity test
    pub homogeneity_pvalue: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of entities
    pub n_entities: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Entity variable name
    pub entity_var: String,
    /// Entities skipped (too few rows, collinear regressors)
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

    /// Rows dropped because of a missing value in y, a regressor or the entity id
    /// (plm `na.omit`).
    #[serde(default, skip_serializing_if = "MissingSummary::is_empty")]
    pub missing: MissingSummary,
}

impl crate::traits::estimator::LinearEstimator for PvcmResult {
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
        self.df
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
}

impl ResultNotes for PvcmResult {
    fn notes_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

impl fmt::Display for PvcmResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Variable Coefficients Model (pvcm)")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Model: {:?}", self.model_type)?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Entities: {}", self.n_entities)?;
        writeln!(f)?;

        if matches!(self.model_type, PvcmType::Random) {
            writeln!(f, "Overall GLS Coefficients (Swamy estimator):")?;
            writeln!(
                f,
                "{:<20} {:>12} {:>12} {:>10} {:>10}",
                "Variable", "Coef", "Std Err", "t", "P>|t|"
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
            writeln!(f)?;
        }

        writeln!(
            f,
            "Homogeneity Test (H0: coefficients are equal across entities):"
        )?;
        writeln!(
            f,
            "  Chi-squared = {:.4}, p-value = {:.4}",
            self.homogeneity_stat, self.homogeneity_pvalue
        )?;
        if self.homogeneity_pvalue < 0.05 {
            writeln!(
                f,
                "  -> Reject H0: coefficients vary significantly across entities"
            )?;
        } else {
            writeln!(f, "  -> Fail to reject H0: coefficients may be poolable")?;
        }
        writeln!(f)?;

        writeln!(f, "Individual Coefficients (first 5 entities shown):")?;
        let mut entities: Vec<&String> = self.individual_coefficients.keys().collect();
        entities.sort();
        let mut count = 0;
        for entity in entities {
            let coeffs = &self.individual_coefficients[entity];
            if count >= 5 {
                break;
            }
            writeln!(
                f,
                "  {}: {:?}",
                entity,
                coeffs
                    .iter()
                    .map(|c| format!("{:.4}", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
            count += 1;
        }
        if self.n_entities > 5 {
            writeln!(f, "  ... ({} more entities)", self.n_entities - 5)?;
        }
        // The warnings, then the missing-value note as the last line.
        f.write_str(&notes_text(self))?;
        if let Some(note) = self.missing.note() {
            writeln!(f, "Missing values: {note}")?;
        }

        Ok(())
    }
}

/// Type of variable coefficients model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PvcmType {
    /// Within: separate OLS for each entity (fixed individual coefficients)
    Within,
    /// Random: Swamy (1970) GLS estimator (random coefficients)
    Random,
}

/// Run Variable Coefficients Model (pvcm).
///
/// Estimates models where coefficients are allowed to vary across entities.
///
/// # Model Types
///
/// - **Within**: Runs separate OLS for each entity. Returns individual-specific
///   coefficients without pooling.
/// - **Random** (Swamy 1970): Assumes coefficients are drawn from a distribution.
///   Returns GLS weighted average with weights inversely proportional to
///   variance-covariance matrices.
///
/// The entity id may be text or numeric; entities are fitted in R factor order of
/// the ids, so the Swamy sums are deterministic. Rows with a missing value (null or
/// NaN) in `y`, a regressor or the entity are dropped and reported in `missing` (plm
/// `na.omit`).
pub fn run_pvcm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    model: PvcmType,
) -> EconResult<PvcmResult> {
    // y, X (with intercept) and the entity on the rows complete in all of them
    // (plm na.omit); numeric and text ids alike
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y_col)
        .numerics(x_cols)
        .factor(entity_col)
        .na(NaPolicy::Omit)
        .extract()?;
    let y_full = cols.numeric_array(y_col);
    let entity = cols.factor(entity_col);
    let n = y_full.len();

    let design = cols.matrix(x_cols, true)?;
    let x_full = design.data;
    let var_names = design.column_names;
    let k = x_full.ncols();

    // Rows of each entity in one pass, entities in level order (R factor order)
    let n_entities = entity.n_levels();
    let mut rows_by_entity: Vec<Vec<usize>> = vec![Vec::new(); n_entities];
    for (row, &code) in entity.codes.iter().enumerate() {
        rows_by_entity[code as usize].push(row);
    }

    // Run separate OLS for each entity
    let mut individual_coefficients: HashMap<String, Vec<f64>> = HashMap::new();
    let mut individual_std_errors: HashMap<String, Vec<f64>> = HashMap::new();
    let mut all_betas: Vec<Array1<f64>> = Vec::new();
    let mut all_vcovs: Vec<Array2<f64>> = Vec::new();
    let mut all_labels: Vec<&str> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    for (level, indices) in rows_by_entity.iter().enumerate() {
        resource::checkpoint()?;
        let entity_label = &entity.levels[level];
        let n_i = indices.len();
        if n_i <= k {
            warnings.push(format!(
                "entity '{entity_label}' skipped: {n_i} observations for {k} coefficients"
            ));
            continue;
        }

        // Extract data for this entity
        let mut y_i = Array1::zeros(n_i);
        let mut x_i = Array2::zeros((n_i, k));

        for (j, &idx) in indices.iter().enumerate() {
            y_i[j] = y_full[idx];
            for col in 0..k {
                x_i[[j, col]] = x_full[[idx, col]];
            }
        }

        // OLS for entity i; collinear regressors are skipped, not pseudo-inverted
        let ls = least_squares(x_i.view(), y_i.view())?;
        if ls.rank < k {
            warnings.push(format!(
                "entity '{entity_label}' skipped: its regressors are collinear (rank {} of {k})",
                ls.rank
            ));
            continue;
        }
        let beta_i = ls.coefficients;

        // Residual variance (n_i > k above)
        let sigma2_i: f64 = ls.residuals.iter().map(|r| r * r).sum::<f64>() / (n_i - k) as f64;

        // Variance-covariance of beta_i
        let vcov_i = &ls.unscaled_cov * sigma2_i;
        let std_err_i: Vec<f64> = vcov_i.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

        individual_coefficients.insert(entity_label.clone(), beta_i.to_vec());
        individual_std_errors.insert(entity_label.clone(), std_err_i);
        all_betas.push(beta_i);
        all_vcovs.push(vcov_i);
        all_labels.push(entity_label);
    }

    let n_valid_entities = all_betas.len();
    if n_valid_entities == 0 {
        return Err(EconError::InsufficientData {
            required: k + 1,
            provided: 0,
            context: "No entities have enough observations for estimation".to_string(),
        });
    }

    // Compute results based on model type
    let (coefficients, std_errors, delta) = match model {
        PvcmType::Within => {
            // Simple average of individual coefficients
            let mut avg_coef: Array1<f64> = Array1::zeros(k);
            for beta in &all_betas {
                avg_coef = &avg_coef + beta;
            }
            avg_coef /= n_valid_entities as f64;

            // Standard error as std dev of individual coefficients
            let mut var_coef: Array1<f64> = Array1::zeros(k);
            for beta in &all_betas {
                let diff = beta - &avg_coef;
                var_coef = &var_coef + &diff.mapv(|d: f64| d * d);
            }
            var_coef /= n_valid_entities.max(1) as f64;
            let se_coef = var_coef.mapv(|v: f64| v.max(0.0).sqrt());

            (avg_coef.to_vec(), se_coef.to_vec(), var_coef.to_vec())
        }
        PvcmType::Random => {
            // Swamy (1970) GLS estimator
            // First, compute mean of individual betas
            let mut beta_bar: Array1<f64> = Array1::zeros(k);
            for beta in &all_betas {
                beta_bar = &beta_bar + beta;
            }
            beta_bar /= n_valid_entities as f64;

            // Estimate Delta (variance of random coefficients)
            // Delta = (1/(N-1)) * sum((beta_i - beta_bar)(beta_i - beta_bar)') - (1/N) * sum(Var(beta_i))
            let mut delta_mat: Array2<f64> = Array2::zeros((k, k));
            for beta in &all_betas {
                let diff = beta - &beta_bar;
                for j in 0..k {
                    for l in 0..k {
                        delta_mat[[j, l]] += diff[j] * diff[l];
                    }
                }
            }
            delta_mat /= (n_valid_entities - 1).max(1) as f64;

            // Subtract average of individual variances
            let mut avg_var: Array2<f64> = Array2::zeros((k, k));
            for vcov in &all_vcovs {
                avg_var = &avg_var + vcov;
            }
            avg_var /= n_valid_entities as f64;
            delta_mat = delta_mat - avg_var;

            // Ensure non-negative diagonal
            for j in 0..k {
                delta_mat[[j, j]] = delta_mat[[j, j]].max(0.0);
            }

            // GLS weights: W_i = (Var(beta_i) + Delta)^(-1)
            let mut sum_w: Array2<f64> = Array2::zeros((k, k));
            let mut sum_wb: Array1<f64> = Array1::zeros(k);

            for (i, beta) in all_betas.iter().enumerate() {
                resource::checkpoint()?;
                let vcov_plus_delta = &all_vcovs[i] + &delta_mat;
                let w_i = sym_inverse(
                    vcov_plus_delta.view(),
                    SingularPolicy::Error,
                    &format!("Var(b_i) + Delta for entity '{}'", all_labels[i]),
                )?
                .inverse;
                sum_w = &sum_w + &w_i;
                sum_wb = &sum_wb + &w_i.dot(beta);
            }

            // GLS estimate
            let sum_w_inv = sym_inverse(
                sum_w.view(),
                SingularPolicy::Error,
                "the Swamy weight sum (sum of (Var(b_i) + Delta)^-1)",
            )?
            .inverse;

            let beta_gls: Array1<f64> = sum_w_inv.dot(&sum_wb);
            let se_gls: Vec<f64> = sum_w_inv.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();
            let delta_diag = delta_mat.diag().to_vec();

            (beta_gls.to_vec(), se_gls, delta_diag)
        }
    };

    // Compute t-statistics and p-values
    let df = n
        .checked_sub(k)
        .ok_or_else(|| EconError::InsufficientData {
            required: k,
            provided: n,
            context: "pvcm: fewer rows than coefficients".to_string(),
        })?;
    let t_stats: Vec<f64> = coefficients
        .iter()
        .zip(std_errors.iter())
        .map(|(b, se): (&f64, &f64)| if *se > 1e-10 { b / se } else { 0.0 })
        .collect();
    let p_values: Vec<f64> = t_stats
        .iter()
        .map(|t: &f64| t_test_p_value(*t, df as f64))
        .collect();
    let significance: Vec<SignificanceLevel> = p_values
        .iter()
        .map(|&p| SignificanceLevel::from_p_value(p))
        .collect();

    // Homogeneity test: H0: all beta_i are equal
    // Test statistic: sum_i (beta_i - beta_bar)' * Var(beta_i)^(-1) * (beta_i - beta_bar)
    // Distributed as chi-squared with (N-1)*k degrees of freedom
    let mut beta_bar_test: Array1<f64> = Array1::zeros(k);
    for beta in &all_betas {
        beta_bar_test = &beta_bar_test + beta;
    }
    let beta_bar_test = beta_bar_test / n_valid_entities as f64;

    let mut homogeneity_stat = 0.0;
    for (i, beta) in all_betas.iter().enumerate() {
        let diff = beta - &beta_bar_test;
        let vcov_inv = sym_inverse(
            all_vcovs[i].view(),
            SingularPolicy::Error,
            &format!("Var(b_i) of entity '{}'", all_labels[i]),
        )?
        .inverse;
        homogeneity_stat += diff.dot(&vcov_inv.dot(&diff));
    }

    let homogeneity_df = ((n_valid_entities - 1) * k) as f64;
    let homogeneity_pvalue = chi_squared_p_value(homogeneity_stat, homogeneity_df);

    let coef_arr = Array1::from_vec(coefficients.clone());
    let se_arr = Array1::from_vec(std_errors.clone());
    // PVCM aggregates entity-level fits; the overall residual vector and
    // full vcov matrix are not retained at the aggregate level. Expose
    // diagonal-only vcov (built from std_errors) and an empty residual
    // vector via the trait, while keeping the full per-entity detail
    // available through `individual_coefficients` / `individual_std_errors`.
    let mut vcov = Array2::<f64>::zeros((coefficients.len(), coefficients.len()));
    for (i, &se) in std_errors.iter().enumerate() {
        vcov[[i, i]] = se * se;
    }
    let residuals: Array1<f64> = Array1::zeros(0);

    Ok(PvcmResult {
        model_type: model,
        dep_var: y_col.to_string(),
        variables: var_names,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        significance,
        individual_coefficients,
        individual_std_errors,
        delta,
        homogeneity_stat,
        homogeneity_pvalue,
        n_obs: n,
        n_entities,
        df,
        entity_var: entity_col.to_string(),
        warnings,
        coef_arr,
        se_arr,
        residuals,
        vcov,
        missing: cols.missing().clone(),
    })
}

/// Run Mean Group (MG) estimator for heterogeneous panels.
///
/// The Mean Group estimator computes the simple average of individual-specific
/// OLS estimates. This is a special case of pvcm with equal weights.
pub fn run_pmg(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<PvcmResult> {
    // pmg is essentially pvcm with "within" model
    run_pvcm(dataset, y_col, x_cols, entity_col, PvcmType::Within)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p2a_validation::{Check, Tol};

    /// The mean-group estimator on Grunfeld with missing cells and two missing ids,
    /// pinned to `plm::pmg(model = "mg")` on the 195 complete rows, for a numeric and
    /// a text id. Not compared: `mg_se` (p2a reports the cross-entity standard
    /// deviation, not its standard error; registry deviation ECON-34).
    #[test]
    fn test_validate_pmg_grunfeld_na() {
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        for id in ["firm_id", "firm_name"] {
            let r = run_pmg(&ds, "inv", &["value", "capital"], id).unwrap();
            Check::new(&case)
                .named("mg_coef", &r.variables, &r.coefficients, Tol::coef(195))
                .scalar("n_complete_fe", r.n_obs as f64, Tol::exact())
                .scalar("n_groups", r.n_entities as f64, Tol::exact())
                .finish();
            assert_eq!(r.missing.n_dropped, 5, "{id}");
            let text = r.to_string();
            let last = text.trim_end().lines().last().unwrap_or("");
            assert_eq!(
                last,
                format!(
                    "Missing values: 5 of 200 rows dropped because of missing values \
                     (inv: 2, value: 1, {id}: 2)"
                ),
                "{text}"
            );
        }
    }

    /// Numeric and text ids give the same Swamy fit and the same per-entity fits (in
    /// the same entity order), deterministically; the errors name the column.
    #[test]
    fn test_validate_na_pvcm_numeric_and_string_ids_agree() {
        let (case, ds) = crate::test_support::load_case("panel_grunfeld_na");
        assert_eq!(
            case.dataset,
            "validation/datasets/generated/grunfeld_na.csv"
        );
        let fit = |id: &str| {
            run_pvcm(&ds, "inv", &["value", "capital"], id, PvcmType::Random)
                .unwrap_or_else(|e| panic!("{id}: {e}"))
        };
        let (a, b) = (fit("firm_id"), fit("firm_name"));
        let tol = Tol::new(1e-12, 1e-12);
        for (x, y) in a
            .coefficients
            .iter()
            .chain(&a.std_errors)
            .zip(b.coefficients.iter().chain(&b.std_errors))
        {
            assert!(tol.allows(*x, *y), "{x} vs {y}");
        }
        let sorted = |r: &PvcmResult| {
            let mut keys: Vec<&String> = r.individual_coefficients.keys().collect();
            keys.sort();
            keys.into_iter()
                .map(|k| (k.clone(), r.individual_coefficients[k].clone()))
                .collect::<Vec<_>>()
        };
        let (ka, kb) = (sorted(&a), sorted(&b));
        assert_eq!(ka.len(), 10);
        assert_eq!(ka[0].0, "0");
        assert_eq!(kb[0].0, "F01");
        for ((_, ca), (_, cb)) in ka.iter().zip(&kb) {
            assert_eq!(ca, cb);
        }
        let again = fit("firm_id");
        assert_eq!(again.coefficients, a.coefficients);
        assert_eq!(again.std_errors, a.std_errors);

        match run_pvcm(&ds, "firm_name", &["value"], "firm_id", PvcmType::Within) {
            Err(EconError::NonNumericColumn { column }) => assert_eq!(column, "firm_name"),
            other => panic!("expected NonNumericColumn, got {other:?}"),
        }
        match run_pvcm(&ds, "inv", &["value"], "nope", PvcmType::Within) {
            Err(EconError::ColumnNotFound { column, .. }) => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
    }

    /// `pvcm_panel` with entity 3's `x2` set to `2 * x1`, and, with `cut_12`, entity 12
    /// cut to its first 3 rows.
    fn pvcm_panel_degenerate(ds: &Dataset, cut_12: bool) -> Dataset {
        use polars::prelude::*;
        let df = ds.df();
        let get = |c: &str| -> Vec<f64> {
            df.column(c)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap()
                .f64()
                .unwrap()
                .into_no_null_iter() // na-ok: test fixture without missing cells
                .collect()
        };
        let (id, t, x1, mut x2, y) = (get("id"), get("t"), get("x1"), get("x2"), get("y"));
        for i in 0..id.len() {
            if id[i] == 3.0 {
                x2[i] = 2.0 * x1[i];
            }
        }
        let keep: Vec<usize> = (0..id.len())
            .filter(|&i| !(cut_12 && id[i] == 12.0 && t[i] > 3.0))
            .collect();
        let pick = |v: &[f64]| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        let ids: Vec<i64> = keep.iter().map(|&i| id[i] as i64).collect();
        Dataset::new(
            df! {
                "id" => ids,
                "x1" => pick(&x1),
                "x2" => pick(&x2),
                "y" => pick(&y),
            }
            .unwrap(),
        )
    }

    #[test]
    fn test_validate_pvcm_skips_collinear_entity_with_warning() {
        let (_case, ds) = crate::test_support::load_case("pvcm_pmg_sim");
        let r = run_pvcm(
            &pvcm_panel_degenerate(&ds, false),
            "y",
            &["x1", "x2"],
            "id",
            PvcmType::Within,
        )
        .unwrap();
        assert_eq!(r.individual_coefficients.len(), 11);
        assert!(!r.individual_coefficients.contains_key("3"));
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "entity '3' skipped: its regressors are collinear (rank 2 of 3)"),
            "{:?}",
            r.warnings
        );
        let r = run_pvcm(
            &pvcm_panel_degenerate(&ds, true),
            "y",
            &["x1", "x2"],
            "id",
            PvcmType::Random,
        )
        .unwrap();
        assert!(
            r.warnings
                .iter()
                .any(|w| w == "entity '12' skipped: 3 observations for 3 coefficients"),
            "{:?}",
            r.warnings
        );
        let text = r.to_string();
        assert!(text.contains("Warning: entity '12' skipped"), "{text}");
    }

    /// plm::pvcm(model = "within") on pvcm_pmg_sim: every entity's coefficients and SEs.
    #[test]
    fn test_validate_pvcm_within_sim() {
        let (case, ds) = crate::test_support::load_case("pvcm_pmg_sim");
        let r = run_pvcm(&ds, "y", &["x1", "x2"], "id", PvcmType::Within).unwrap();
        let mut c = Check::new(&case);
        for i in 1..=12 {
            let id = i.to_string();
            c.named(
                &format!("within_coef_{i}"),
                &r.variables,
                &r.individual_coefficients[&id],
                Tol::coef(240),
            )
            .named(
                &format!("within_se_{i}"),
                &r.variables,
                &r.individual_std_errors[&id],
                Tol::se(240),
            );
        }
        c.scalar("n_obs", r.n_obs as f64, Tol::exact());
        c.finish();
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
    }

    /// plm::pvcm(model = "random") on pvcm_pmg_sim (Delta positive definite).
    #[test]
    fn test_validate_pvcm_random_sim() {
        let (case, ds) = crate::test_support::load_case("pvcm_pmg_sim");
        let r = run_pvcm(&ds, "y", &["x1", "x2"], "id", PvcmType::Random).unwrap();
        // deviation ECON-35: random_chisq (homogeneity test around the unweighted mean); Delta clamp when not PD
        Check::new(&case)
            .named("random_coef", &r.variables, &r.coefficients, Tol::coef(240))
            .named("random_se", &r.variables, &r.std_errors, Tol::se(240))
            .named("random_delta_diag", &r.variables, &r.delta, Tol::stat(240))
            .finish();
    }

    /// plm::pmg(model = "mg") on pvcm_pmg_sim.
    #[test]
    fn test_validate_pmg_sim() {
        let (case, ds) = crate::test_support::load_case("pvcm_pmg_sim");
        let r = run_pmg(&ds, "y", &["x1", "x2"], "id").unwrap();
        // deviation ECON-34: mg_se
        Check::new(&case)
            .named("mg_coef", &r.variables, &r.coefficients, Tol::coef(240))
            .scalar("n_obs", r.n_obs as f64, Tol::exact())
            .finish();
    }
}
