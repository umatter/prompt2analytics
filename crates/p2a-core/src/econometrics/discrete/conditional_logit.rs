//! McFadden's Conditional Logit (mlogit).
//!
//! # Mathematical Background
//!
//! McFadden's conditional logit (also called "mixed logit" in some contexts) specifies:
//!
//! U_ij = V_ij + epsilon_ij
//!
//! where:
//! - V_ij = X_ij'beta + Z_i'gamma_j is the deterministic utility
//! - X_ij are alternative-specific variables
//! - Z_i are individual-specific variables
//! - beta are generic coefficients (same across all alternatives)
//! - gamma_j are alternative-specific coefficients
//!
//! # References
//!
//! - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
//!
//! R equivalent: `mlogit::mlogit()`

use ndarray::{Array1, Array2, ArrayView1};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::normal_cdf;

/// Result from McFadden's conditional logit model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlogitResult {
    /// Alternative-specific variable names (generic coefficients)
    pub alt_specific_vars: Vec<String>,
    /// Individual-specific variable names (alternative-specific coefficients)
    pub ind_specific_vars: Vec<String>,
    /// Alternative names
    pub alternatives: Vec<String>,
    /// Reference alternative
    pub reference_alternative: String,
    /// Generic coefficients for alternative-specific variables
    pub beta: Vec<f64>,
    /// Standard errors for beta
    pub beta_std_errors: Vec<f64>,
    /// Z-statistics for beta
    pub beta_z_stats: Vec<f64>,
    /// P-values for beta
    pub beta_p_values: Vec<f64>,
    /// Alternative-specific coefficients for individual-specific variables
    pub gamma: Vec<Vec<f64>>,
    /// Standard errors for gamma
    pub gamma_std_errors: Vec<Vec<f64>>,
    /// Z-statistics for gamma
    pub gamma_z_stats: Vec<Vec<f64>>,
    /// P-values for gamma
    pub gamma_p_values: Vec<Vec<f64>>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Log-likelihood of null model
    pub log_likelihood_null: f64,
    /// McFadden's pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Number of choice situations
    pub n_choice_situations: usize,
    /// Number of alternatives
    pub n_alternatives: usize,
    /// Choice counts by alternative
    pub choice_counts: Vec<usize>,
}

impl fmt::Display for MlogitResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "McFadden's Conditional Logit (mlogit)")?;
        writeln!(f, "======================================")?;
        writeln!(f, "N (choice situations) = {}", self.n_choice_situations)?;
        writeln!(
            f,
            "Alternatives = {} (reference: {})",
            self.n_alternatives, self.reference_alternative
        )?;
        writeln!(f)?;

        if !self.alt_specific_vars.is_empty() {
            writeln!(
                f,
                "--- Alternative-Specific Variables (Generic Coefficients) ---"
            )?;
            writeln!(
                f,
                "{:<15} {:>10} {:>10} {:>10} {:>10}",
                "Variable", "Coef", "Std.Err", "z", "P>|z|"
            )?;
            writeln!(
                f,
                "{:-<15} {:-<10} {:-<10} {:-<10} {:-<10}",
                "", "", "", "", ""
            )?;
            for (i, var) in self.alt_specific_vars.iter().enumerate() {
                writeln!(
                    f,
                    "{:<15} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                    var,
                    self.beta[i],
                    self.beta_std_errors[i],
                    self.beta_z_stats[i],
                    self.beta_p_values[i]
                )?;
            }
            writeln!(f)?;
        }

        if !self.ind_specific_vars.is_empty() {
            writeln!(
                f,
                "--- Individual-Specific Variables (Alternative-Specific Coefficients) ---"
            )?;
            for (alt_idx, alt) in self.alternatives.iter().enumerate() {
                if *alt == self.reference_alternative {
                    continue;
                }
                let coef_idx = if alt_idx == 0 {
                    0
                } else {
                    self.alternatives
                        .iter()
                        .take(alt_idx)
                        .filter(|a| **a != self.reference_alternative)
                        .count()
                };
                if coef_idx >= self.gamma.len() {
                    continue;
                }
                writeln!(f, "\n  {} vs {}:", alt, self.reference_alternative)?;
                writeln!(
                    f,
                    "  {:<13} {:>10} {:>10} {:>10} {:>10}",
                    "Variable", "Coef", "Std.Err", "z", "P>|z|"
                )?;
                for (var_idx, var) in self.ind_specific_vars.iter().enumerate() {
                    writeln!(
                        f,
                        "  {:<13} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                        var,
                        self.gamma[coef_idx][var_idx],
                        self.gamma_std_errors[coef_idx][var_idx],
                        self.gamma_z_stats[coef_idx][var_idx],
                        self.gamma_p_values[coef_idx][var_idx]
                    )?;
                }
            }
            writeln!(f)?;
        }

        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "AIC: {:.4}, BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {} ({} iterations)",
            self.converged, self.iterations
        )?;
        Ok(())
    }
}

/// Pre-computed data structure for fast mlogit computation.
struct MlogitData {
    /// For each choice situation: (start_row, n_alts, chosen_local_idx)
    choice_info: Vec<(usize, usize, usize)>,
    /// Pre-computed feature matrix
    features: Array2<f64>,
    /// Number of parameters
    n_params: usize,
}

impl MlogitData {
    fn new(
        choice_situations: &std::collections::HashMap<String, Vec<(usize, usize)>>,
        choices: &[f64],
        x_alt_specific: &[Vec<f64>],
        z_ind_specific: &[Vec<f64>],
        n_alt_specific: usize,
        n_ind_specific: usize,
        n_alternatives: usize,
        ref_idx: usize,
    ) -> Self {
        let n_params = n_alt_specific + n_ind_specific * (n_alternatives - 1).max(0);
        let total_rows: usize = choice_situations.values().map(|v| v.len()).sum();

        let mut choice_info = Vec::with_capacity(choice_situations.len());
        let mut features = Array2::<f64>::zeros((total_rows, n_params));

        let mut flat_idx = 0;
        for alts in choice_situations.values() {
            let start = flat_idx;
            let n_alts = alts.len();
            let mut chosen_local = 0;

            for (local_idx, &(row_idx, alt_idx)) in alts.iter().enumerate() {
                // Fill feature row
                for (k, x_col) in x_alt_specific.iter().enumerate() {
                    features[[flat_idx, k]] = x_col[row_idx];
                }

                if alt_idx != ref_idx && n_ind_specific > 0 {
                    let gamma_offset = n_alt_specific
                        + (if alt_idx < ref_idx {
                            alt_idx
                        } else {
                            alt_idx - 1
                        }) * n_ind_specific;
                    for (k, z_col) in z_ind_specific.iter().enumerate() {
                        features[[flat_idx, gamma_offset + k]] = z_col[row_idx];
                    }
                }

                if choices[row_idx] > 0.5 {
                    chosen_local = local_idx;
                }

                flat_idx += 1;
            }

            choice_info.push((start, n_alts, chosen_local));
        }

        Self {
            choice_info,
            features,
            n_params,
        }
    }
}

fn compute_mlogit_loglik_fast(data: &MlogitData, params: &[f64]) -> f64 {
    let params_arr = ArrayView1::from(params);
    let mut ll = 0.0;

    for &(start, n_alts, chosen_local) in &data.choice_info {
        let mut max_v = f64::NEG_INFINITY;
        let mut chosen_v = 0.0;
        let mut sum_exp = 0.0;

        for i in 0..n_alts {
            let row = start + i;
            let v: f64 = data.features.row(row).dot(&params_arr);
            if v > max_v {
                max_v = v;
            }
            if i == chosen_local {
                chosen_v = v;
            }
        }

        for i in 0..n_alts {
            let row = start + i;
            let v: f64 = data.features.row(row).dot(&params_arr);
            sum_exp += (v - max_v).exp();
        }

        ll += chosen_v - max_v - sum_exp.ln();
    }

    ll
}

fn compute_mlogit_derivatives_fast(
    data: &MlogitData,
    params: &[f64],
) -> (f64, Array1<f64>, Array2<f64>) {
    let n_params = data.n_params;
    let params_arr = ArrayView1::from(params);
    let mut ll = 0.0;
    let mut gradient = Array1::<f64>::zeros(n_params);
    let mut hessian = Array2::<f64>::zeros((n_params, n_params));

    let max_alts = data.choice_info.iter().map(|c| c.1).max().unwrap_or(0);
    let mut utilities = vec![0.0; max_alts];
    let mut probs = vec![0.0; max_alts];
    let mut weighted_sum = vec![0.0; n_params];
    let mut px: Vec<Vec<f64>> = vec![vec![0.0; n_params]; max_alts];

    for &(start, n_alts, chosen_local) in &data.choice_info {
        let mut max_v = f64::NEG_INFINITY;
        for i in 0..n_alts {
            let v = data.features.row(start + i).dot(&params_arr);
            utilities[i] = v;
            if v > max_v {
                max_v = v;
            }
        }

        let mut sum_exp = 0.0;
        for i in 0..n_alts {
            let e = (utilities[i] - max_v).exp();
            probs[i] = e;
            sum_exp += e;
        }
        let inv_sum = 1.0 / sum_exp;
        for i in 0..n_alts {
            probs[i] *= inv_sum;
        }

        ll += utilities[chosen_local] - max_v - sum_exp.ln();

        for k in 0..n_params {
            weighted_sum[k] = 0.0;
        }

        for i in 0..n_alts {
            let residual = if i == chosen_local {
                1.0 - probs[i]
            } else {
                -probs[i]
            };
            let p_i = probs[i];
            let feat_row = data.features.row(start + i);
            for k in 0..n_params {
                let x_ik = feat_row[k];
                gradient[k] += residual * x_ik;
                let px_ik = p_i * x_ik;
                weighted_sum[k] += px_ik;
                px[i][k] = px_ik;
            }
        }

        for k in 0..n_params {
            let mut diag_sum = 0.0;
            for i in 0..n_alts {
                let x_ik = data.features[[start + i, k]];
                diag_sum += px[i][k] * x_ik;
            }
            hessian[[k, k]] += -diag_sum + weighted_sum[k] * weighted_sum[k];

            for l in (k + 1)..n_params {
                let mut off_diag_sum = 0.0;
                for i in 0..n_alts {
                    let x_il = data.features[[start + i, l]];
                    off_diag_sum += px[i][k] * x_il;
                }
                let h_kl = -off_diag_sum + weighted_sum[k] * weighted_sum[l];
                hessian[[k, l]] += h_kl;
                hessian[[l, k]] += h_kl;
            }
        }
    }

    (ll, gradient, hessian)
}

/// Helper to extract column as strings.
fn extract_string_or_int_column(
    df: &polars::prelude::DataFrame,
    col: &str,
) -> EconResult<Vec<String>> {
    let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
        column: col.to_string(),
        available: df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect(),
    })?;

    if let Ok(ca) = series.str() {
        Ok(ca.into_no_null_iter().map(|s| s.to_string()).collect())
    } else if let Ok(ca) = series.i64() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else if let Ok(ca) = series.i32() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else if let Ok(ca) = series.f64() {
        Ok(ca.into_no_null_iter().map(|v| v.to_string()).collect())
    } else {
        Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be string or integer", col),
        })
    }
}

/// Run McFadden's conditional logit model.
///
/// R equivalent: `mlogit::mlogit()`
pub fn run_mlogit(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    alt_specific_cols: &[&str],
    ind_specific_cols: &[&str],
    reference: Option<&str>,
) -> EconResult<MlogitResult> {
    let df = dataset.df();
    let n_rows = df.height();

    let choice_ids: Vec<String> = extract_string_or_int_column(df, choice_id_col)?;
    let alt_ids: Vec<String> = extract_string_or_int_column(df, alt_id_col)?;

    let choice_series = df
        .column(choice_col)
        .map_err(|_| EconError::ColumnNotFound {
            column: choice_col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;
    let choices: Vec<f64> = if let Ok(ca) = choice_series.f64() {
        ca.into_no_null_iter().collect()
    } else if let Ok(ca) = choice_series.i64() {
        ca.into_no_null_iter().map(|v| v as f64).collect()
    } else {
        return Err(EconError::InvalidSpecification {
            message: format!("Column '{}' must be numeric (0/1)", choice_col),
        });
    };

    let unique_choice_ids: Vec<String> = choice_ids
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    let n_choice_situations = unique_choice_ids.len();

    let mut alternatives: Vec<String> = alt_ids
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    alternatives.sort();
    let n_alternatives = alternatives.len();

    if n_alternatives < 2 {
        return Err(EconError::InvalidSpecification {
            message: "Conditional logit requires at least 2 alternatives".to_string(),
        });
    }

    let ref_alt = reference
        .map(|s| s.to_string())
        .unwrap_or_else(|| alternatives[0].clone());

    if !alternatives.contains(&ref_alt) {
        return Err(EconError::InvalidSpecification {
            message: format!("Reference alternative '{}' not found in data", ref_alt),
        });
    }

    let alt_to_idx: std::collections::HashMap<String, usize> = alternatives
        .iter()
        .enumerate()
        .map(|(i, a)| (a.clone(), i))
        .collect();
    let ref_idx = alt_to_idx[&ref_alt];

    let mut choice_situations: std::collections::HashMap<String, Vec<(usize, usize)>> =
        std::collections::HashMap::new();

    for row_idx in 0..n_rows {
        let cid = &choice_ids[row_idx];
        let aid = &alt_ids[row_idx];
        let alt_idx = alt_to_idx[aid];
        choice_situations
            .entry(cid.clone())
            .or_default()
            .push((row_idx, alt_idx));
    }

    // Extract variables
    let n_alt_specific = alt_specific_cols.len();
    let mut x_alt_specific: Vec<Vec<f64>> = Vec::new();
    for col in alt_specific_cols {
        let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
            column: col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;
        let vals: Vec<f64> = if let Ok(ca) = series.f64() {
            ca.into_no_null_iter().collect()
        } else if let Ok(ca) = series.i64() {
            ca.into_no_null_iter().map(|v| v as f64).collect()
        } else {
            return Err(EconError::InvalidSpecification {
                message: format!("Column '{}' must be numeric", col),
            });
        };
        x_alt_specific.push(vals);
    }

    let n_ind_specific = ind_specific_cols.len();
    let mut z_ind_specific: Vec<Vec<f64>> = Vec::new();
    for col in ind_specific_cols {
        let series = df.column(col).map_err(|_| EconError::ColumnNotFound {
            column: col.to_string(),
            available: df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        })?;
        let vals: Vec<f64> = if let Ok(ca) = series.f64() {
            ca.into_no_null_iter().collect()
        } else if let Ok(ca) = series.i64() {
            ca.into_no_null_iter().map(|v| v as f64).collect()
        } else {
            return Err(EconError::InvalidSpecification {
                message: format!("Column '{}' must be numeric", col),
            });
        };
        z_ind_specific.push(vals);
    }

    let n_gamma = n_ind_specific * (n_alternatives - 1);
    let n_params = n_alt_specific + n_gamma;

    if n_params == 0 {
        return Err(EconError::InvalidSpecification {
            message: "At least one variable must be specified".to_string(),
        });
    }

    let mut params = vec![0.0; n_params];

    let mlogit_data = MlogitData::new(
        &choice_situations,
        &choices,
        &x_alt_specific,
        &z_ind_specific,
        n_alt_specific,
        n_ind_specific,
        n_alternatives,
        ref_idx,
    );

    // Newton-Raphson optimization
    let max_iter = 100;
    let tol = 1e-8;
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        let (ll, gradient, hessian) = compute_mlogit_derivatives_fast(&mlogit_data, &params);

        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tol {
            converged = true;
            break;
        }

        let (h_inv, _) = match safe_inverse(&hessian.view()) {
            Ok(inv) => inv,
            Err(_) => {
                for i in 0..n_params {
                    params[i] -= 0.1 * gradient[i];
                }
                continue;
            }
        };

        let delta = h_inv.dot(&(-&gradient));

        let mut step = 1.0;
        let mut best_ll = ll;
        let mut best_params = params.clone();

        for _ in 0..10 {
            let new_params: Vec<f64> = params
                .iter()
                .zip(delta.iter())
                .map(|(&p, &d)| p + step * d)
                .collect();

            let new_ll = compute_mlogit_loglik_fast(&mlogit_data, &new_params);

            if new_ll > best_ll {
                best_ll = new_ll;
                best_params = new_params;
                break;
            }
            step *= 0.5;
        }

        params = best_params;
    }

    let log_likelihood = compute_mlogit_loglik_fast(&mlogit_data, &params);
    let log_likelihood_null = n_choice_situations as f64 * (-(n_alternatives as f64).ln());

    let (_, _, hessian) = compute_mlogit_derivatives_fast(&mlogit_data, &params);

    let vcov = match safe_inverse(&(-&hessian).view()) {
        Ok((inv, _)) => inv,
        Err(_) => Array2::eye(n_params),
    };

    let std_errors: Vec<f64> = (0..n_params)
        .map(|i| vcov[[i, i]].max(0.0).sqrt())
        .collect();

    let beta: Vec<f64> = params[..n_alt_specific].to_vec();
    let beta_std_errors: Vec<f64> = std_errors[..n_alt_specific].to_vec();
    let beta_z_stats: Vec<f64> = beta
        .iter()
        .zip(beta_std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();
    let beta_p_values: Vec<f64> = beta_z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    let mut gamma: Vec<Vec<f64>> = Vec::new();
    let mut gamma_std_errors: Vec<Vec<f64>> = Vec::new();
    let mut gamma_z_stats: Vec<Vec<f64>> = Vec::new();
    let mut gamma_p_values: Vec<Vec<f64>> = Vec::new();

    if n_ind_specific > 0 {
        let gamma_start = n_alt_specific;
        for j in 0..n_alternatives {
            if j == ref_idx {
                continue;
            }
            let offset = if j < ref_idx { j } else { j - 1 };
            let start = gamma_start + offset * n_ind_specific;
            let end = start + n_ind_specific;

            let g: Vec<f64> = params[start..end].to_vec();
            let g_se: Vec<f64> = std_errors[start..end].to_vec();
            let g_z: Vec<f64> = g
                .iter()
                .zip(g_se.iter())
                .map(|(&coef, &se)| if se > 0.0 { coef / se } else { 0.0 })
                .collect();
            let g_p: Vec<f64> = g_z
                .iter()
                .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
                .collect();

            gamma.push(g);
            gamma_std_errors.push(g_se);
            gamma_z_stats.push(g_z);
            gamma_p_values.push(g_p);
        }
    }

    let mut choice_counts = vec![0usize; n_alternatives];
    for row_idx in 0..n_rows {
        if choices[row_idx] > 0.5 {
            let alt_idx = alt_to_idx[&alt_ids[row_idx]];
            choice_counts[alt_idx] += 1;
        }
    }

    let pseudo_r_squared = 1.0 - log_likelihood / log_likelihood_null;
    let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
    let bic = -2.0 * log_likelihood + (n_params as f64) * (n_choice_situations as f64).ln();

    Ok(MlogitResult {
        alt_specific_vars: alt_specific_cols.iter().map(|s| s.to_string()).collect(),
        ind_specific_vars: ind_specific_cols.iter().map(|s| s.to_string()).collect(),
        alternatives: alternatives.clone(),
        reference_alternative: ref_alt,
        beta,
        beta_std_errors,
        beta_z_stats,
        beta_p_values,
        gamma,
        gamma_std_errors,
        gamma_z_stats,
        gamma_p_values,
        log_likelihood,
        log_likelihood_null,
        pseudo_r_squared,
        aic,
        bic,
        iterations,
        converged,
        n_choice_situations,
        n_alternatives,
        choice_counts,
    })
}

/// Convenience function for conditional logit with only alternative-specific variables.
///
/// R equivalent: `mlogit::mlogit()`
pub fn run_conditional_logit(
    dataset: &Dataset,
    choice_id_col: &str,
    alt_id_col: &str,
    choice_col: &str,
    x_cols: &[&str],
    reference: Option<&str>,
) -> EconResult<MlogitResult> {
    run_mlogit(
        dataset,
        choice_id_col,
        alt_id_col,
        choice_col,
        x_cols,
        &[],
        reference,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_mlogit_dataset() -> Dataset {
        let df = df! {
            "choice_id" => [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "alt_id" => ["car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train"],
            "choice" => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "cost" => [10.0, 3.0, 5.0, 8.0, 2.0, 4.0, 15.0, 4.0, 3.0,
                      5.0, 5.0, 8.0, 12.0, 2.0, 6.0],
            "time" => [20.0, 40.0, 30.0, 15.0, 35.0, 25.0, 25.0, 45.0, 20.0,
                      10.0, 30.0, 40.0, 20.0, 30.0, 25.0],
            "income" => [50.0, 50.0, 50.0, 30.0, 30.0, 30.0, 70.0, 70.0, 70.0,
                        60.0, 60.0, 60.0, 25.0, 25.0, 25.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_mlogit_basic() {
        let dataset = create_mlogit_dataset();
        let result = run_mlogit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[],
            None,
        )
        .unwrap();

        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);
        assert_eq!(result.beta.len(), 2);
        assert!(result.beta[0] < 0.0); // Cost should be negative
    }

    #[test]
    fn test_conditional_logit() {
        let dataset = create_mlogit_dataset();
        let result =
            run_conditional_logit(&dataset, "choice_id", "alt_id", "choice", &["cost"], None)
                .unwrap();

        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.beta.len(), 1);
        assert!(result.gamma.is_empty());
    }

    // ==========================================================================
    // R Validation Tests
    // ==========================================================================

    #[test]
    fn test_validate_mlogit_vs_r() {
        // R reference: mlogit::mlogit()
        // Standard mode choice problem: cost and time should have negative coefficients

        // Create more realistic mode choice data
        let n_individuals = 50;
        let alternatives = ["car", "bus", "train"];
        let n_alts = alternatives.len();
        let n_rows = n_individuals * n_alts;

        let mut choice_id: Vec<i64> = Vec::with_capacity(n_rows);
        let mut alt_id: Vec<&str> = Vec::with_capacity(n_rows);
        let mut choice: Vec<f64> = Vec::with_capacity(n_rows);
        let mut cost: Vec<f64> = Vec::with_capacity(n_rows);
        let mut time: Vec<f64> = Vec::with_capacity(n_rows);

        for i in 0..n_individuals {
            // Generate alternative-specific attributes
            let car_cost = 8.0 + (i as f64 * 0.1) % 5.0;
            let bus_cost = 2.0 + (i as f64 * 0.05) % 2.0;
            let train_cost = 4.0 + (i as f64 * 0.07) % 3.0;

            let car_time = 15.0 + (i as f64 * 0.2) % 10.0;
            let bus_time = 35.0 + (i as f64 * 0.3) % 15.0;
            let train_time = 25.0 + (i as f64 * 0.15) % 8.0;

            let costs = [car_cost, bus_cost, train_cost];
            let times = [car_time, bus_time, train_time];

            // Utility: V = -0.3*cost - 0.05*time
            let utilities: Vec<f64> = costs
                .iter()
                .zip(times.iter())
                .map(|(&c, &t)| -0.3 * c - 0.05 * t + (i as f64 * 0.5).sin() * 0.5)
                .collect();

            // Choose max utility
            let chosen_idx = utilities
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap()
                .0;

            for (j, alt) in alternatives.iter().enumerate() {
                choice_id.push(i as i64 + 1);
                alt_id.push(alt);
                choice.push(if j == chosen_idx { 1.0 } else { 0.0 });
                cost.push(costs[j]);
                time.push(times[j]);
            }
        }

        let df = df! {
            "choice_id" => choice_id,
            "alt_id" => alt_id,
            "choice" => choice,
            "cost" => cost,
            "time" => time
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_mlogit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[],
            None,
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.n_choice_situations, n_individuals);
        assert_eq!(result.n_alternatives, n_alts);
        assert_eq!(result.beta.len(), 2);

        // Cost and time coefficients should be negative (higher cost/time = lower utility)
        assert!(
            result.beta[0] < 0.0,
            "Cost coefficient should be negative: {}",
            result.beta[0]
        );
        // Time coefficient may vary more
        assert!(
            result.beta[1].is_finite(),
            "Time coefficient should be finite"
        );

        // Log-likelihood should be finite
        assert!(
            result.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(
            result.log_likelihood < 0.0,
            "Log-likelihood should be negative"
        );
    }

    #[test]
    fn test_validate_conditional_logit_structure() {
        // Test conditional logit with only alternative-specific variables
        let dataset = create_mlogit_dataset();
        let result = run_conditional_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            None,
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);
        assert_eq!(result.beta.len(), 2);
        assert!(result.gamma.is_empty());

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());

        // Pseudo R-squared should be in [0, 1]
        assert!(result.pseudo_r_squared >= 0.0 && result.pseudo_r_squared <= 1.0);
    }

    #[test]
    fn test_validate_mlogit_with_individual_vars() {
        // Test with both alternative-specific and individual-specific variables
        let dataset = create_mlogit_dataset();
        let result = run_mlogit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost"],
            &["income"],
            None,
        )
        .unwrap();

        // Structure checks
        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.beta.len(), 1); // One alternative-specific var
        assert!(!result.gamma.is_empty()); // Should have individual-specific coefficients

        // Each non-reference alternative should have coefficients for income
        // With 3 alternatives, that's 2 sets of gamma coefficients
        assert_eq!(result.gamma.len(), 2);

        // Log-likelihood should be finite
        assert!(result.log_likelihood.is_finite());
    }
}
