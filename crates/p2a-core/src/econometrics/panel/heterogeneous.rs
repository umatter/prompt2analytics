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
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::{safe_inverse, xtx, xty};
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

    // ── Trait-backing storage (LinearEstimator). Skipped from JSON.
    #[serde(skip, default)]
    pub coef_arr: ndarray::Array1<f64>,
    #[serde(skip, default)]
    pub se_arr: ndarray::Array1<f64>,
    #[serde(skip, default)]
    pub residuals: ndarray::Array1<f64>,
    #[serde(skip, default)]
    pub vcov: ndarray::Array2<f64>,
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
        let mut count = 0;
        for (entity, coeffs) in &self.individual_coefficients {
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
pub fn run_pvcm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
    model: PvcmType,
) -> EconResult<PvcmResult> {
    // Extract full data
    let y_full = DesignMatrix::extract_column(dataset.df(), y_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: y_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let entity_col_data =
        dataset
            .df()
            .column(entity_col)
            .map_err(|_| EconError::ColumnNotFound {
                column: entity_col.to_string(),
                available: dataset
                    .df()
                    .get_column_names()
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            })?;

    let entities: Vec<String> = entity_col_data
        .str()
        .map_err(|_| EconError::InvalidSpecification {
            message: "Entity column must be string type".to_string(),
        })?
        .into_iter()
        .map(|opt: Option<&str>| opt.unwrap_or("").to_string())
        .collect();

    let n = y_full.len();

    // Build X matrix (with intercept)
    let design = DesignMatrix::from_dataframe(dataset.df(), x_cols, true)?;
    let x_full = design.data;
    let var_names = design.column_names;
    let k = x_full.ncols();

    // Group data by entity
    let mut entity_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, ent) in entities.iter().enumerate() {
        entity_map.entry(ent.clone()).or_default().push(idx);
    }

    let n_entities = entity_map.len();

    // Run separate OLS for each entity
    let mut individual_coefficients: HashMap<String, Vec<f64>> = HashMap::new();
    let mut individual_std_errors: HashMap<String, Vec<f64>> = HashMap::new();
    let mut individual_vcov: HashMap<String, Array2<f64>> = HashMap::new();
    let mut all_betas: Vec<Array1<f64>> = Vec::new();
    let mut all_vcovs: Vec<Array2<f64>> = Vec::new();

    for (entity, indices) in &entity_map {
        let n_i = indices.len();
        if n_i <= k {
            // Not enough observations for this entity
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

        // OLS for entity i
        let xtx_i = xtx(&x_i.view());
        let (xtx_inv_i, _) = match safe_inverse(&xtx_i.view()) {
            Ok(inv) => inv,
            Err(_) => continue, // Skip singular entities
        };

        let xty_i = xty(&x_i.view(), &y_i);
        let beta_i: Array1<f64> = xtx_inv_i.dot(&xty_i);

        // Residuals and variance
        let fitted_i = x_i.dot(&beta_i);
        let resid_i = &y_i - &fitted_i;
        let df_i = n_i.saturating_sub(k);
        let sigma2_i: f64 = if df_i > 0 {
            resid_i.iter().map(|r| r * r).sum::<f64>() / df_i as f64
        } else {
            resid_i.iter().map(|r| r * r).sum::<f64>()
        };

        // Variance-covariance of beta_i
        let vcov_i = &xtx_inv_i * sigma2_i;
        let std_err_i: Vec<f64> = vcov_i.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();

        individual_coefficients.insert(entity.clone(), beta_i.to_vec());
        individual_std_errors.insert(entity.clone(), std_err_i);
        individual_vcov.insert(entity.clone(), vcov_i.clone());
        all_betas.push(beta_i);
        all_vcovs.push(vcov_i);
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
                let vcov_plus_delta = &all_vcovs[i] + &delta_mat;
                if let Ok((w_i, _)) = safe_inverse(&vcov_plus_delta.view()) {
                    sum_w = &sum_w + &w_i;
                    sum_wb = &sum_wb + &w_i.dot(beta);
                }
            }

            // GLS estimate
            let (sum_w_inv, _) =
                safe_inverse(&sum_w.view()).map_err(|_| EconError::SingularMatrix {
                    context: "GLS weight matrix in pvcm".to_string(),
                    suggestion: "Check for collinearity".to_string(),
                })?;

            let beta_gls: Array1<f64> = sum_w_inv.dot(&sum_wb);
            let se_gls: Vec<f64> = sum_w_inv.diag().mapv(|v| v.max(0.0).sqrt()).to_vec();
            let delta_diag = delta_mat.diag().to_vec();

            (beta_gls.to_vec(), se_gls, delta_diag)
        }
    };

    // Compute t-statistics and p-values
    let df = n.saturating_sub(k);
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
        if let Ok((vcov_inv, _)) = safe_inverse(&all_vcovs[i].view()) {
            let quad_form = diff.dot(&vcov_inv.dot(&diff));
            homogeneity_stat += quad_form;
        }
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
        coef_arr,
        se_arr,
        residuals,
        vcov,
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
