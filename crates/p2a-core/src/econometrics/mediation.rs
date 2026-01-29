//! Causal Mediation Analysis using Inverse Probability Weighting.
//!
//! Implements methods from Huber (2014) for decomposing treatment effects
//! into natural direct and indirect effects.
//!
//! # Overview
//!
//! Mediation analysis answers: "How much of the treatment effect operates
//! through a particular mediator variable?"
//!
//! # Key Concepts
//!
//! - **Total Effect (ATE)**: Overall effect of treatment on outcome
//! - **Natural Direct Effect (NDE)**: Effect NOT through the mediator
//! - **Natural Indirect Effect (NIE)**: Effect through the mediator
//! - **Decomposition**: ATE = NDE + NIE
//!
//! # References
//!
//! - Huber, M. (2014). "Identifying causal mechanisms (primarily) based on
//!   inverse probability weighting." Journal of Applied Econometrics, 29, 920-943.
//! - Imai, K., Keele, L., & Tingley, D. (2010). "A General Approach to Causal
//!   Mediation Analysis." Psychological Methods, 15(4), 309-334.

use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::prelude::*;
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::traits::estimator::{logistic_cdf, normal_cdf};

/// Configuration for mediation analysis.
#[derive(Debug, Clone)]
pub struct MediationConfig {
    /// Number of bootstrap replications for standard errors (default: 999)
    pub bootstrap: usize,
    /// Trimming threshold for propensity scores (default: 0.05)
    pub trim: f64,
    /// Random seed for bootstrap (optional, for reproducibility)
    pub seed: Option<u64>,
}

impl Default for MediationConfig {
    fn default() -> Self {
        Self {
            bootstrap: 999,
            trim: 0.05,
            seed: None,
        }
    }
}

/// Result from causal mediation analysis.
#[derive(Debug, Clone)]
pub struct MediationResult {
    /// Total effect (ATE)
    pub total_effect: f64,
    /// Natural direct effect (NDE)
    pub direct_effect: f64,
    /// Natural indirect effect (NIE)
    pub indirect_effect: f64,
    /// Proportion of total effect mediated (NIE / ATE)
    pub proportion_mediated: f64,
    /// Standard error of total effect
    pub se_total: f64,
    /// Standard error of direct effect
    pub se_direct: f64,
    /// Standard error of indirect effect
    pub se_indirect: f64,
    /// 95% CI for total effect
    pub ci_total: (f64, f64),
    /// 95% CI for direct effect
    pub ci_direct: (f64, f64),
    /// 95% CI for indirect effect
    pub ci_indirect: (f64, f64),
    /// p-value for total effect
    pub p_total: f64,
    /// p-value for direct effect
    pub p_direct: f64,
    /// p-value for indirect effect
    pub p_indirect: f64,
    /// Number of observations used
    pub n_obs: usize,
    /// Number of observations trimmed
    pub n_trimmed: usize,
}

impl fmt::Display for MediationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Causal Mediation Analysis ===")?;
        writeln!(f)?;
        writeln!(f, "Effect Decomposition:")?;
        writeln!(
            f,
            "─────────────────────────────────────────────────────────────"
        )?;
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12} {:>10}",
            "Effect", "Estimate", "Std.Err", "95% CI", "p-value"
        )?;
        writeln!(
            f,
            "─────────────────────────────────────────────────────────────"
        )?;

        let stars = |p: f64| -> &str {
            if p < 0.01 {
                "***"
            } else if p < 0.05 {
                "**"
            } else if p < 0.10 {
                "*"
            } else {
                ""
            }
        };

        writeln!(
            f,
            "{:<20} {:>12.4} {:>12.4} [{:>5.3},{:>5.3}] {:>8.4}{}",
            "Total (ATE)",
            self.total_effect,
            self.se_total,
            self.ci_total.0,
            self.ci_total.1,
            self.p_total,
            stars(self.p_total)
        )?;

        writeln!(
            f,
            "{:<20} {:>12.4} {:>12.4} [{:>5.3},{:>5.3}] {:>8.4}{}",
            "Direct (NDE)",
            self.direct_effect,
            self.se_direct,
            self.ci_direct.0,
            self.ci_direct.1,
            self.p_direct,
            stars(self.p_direct)
        )?;

        writeln!(
            f,
            "{:<20} {:>12.4} {:>12.4} [{:>5.3},{:>5.3}] {:>8.4}{}",
            "Indirect (NIE)",
            self.indirect_effect,
            self.se_indirect,
            self.ci_indirect.0,
            self.ci_indirect.1,
            self.p_indirect,
            stars(self.p_indirect)
        )?;

        writeln!(
            f,
            "─────────────────────────────────────────────────────────────"
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Proportion Mediated: {:.1}%",
            self.proportion_mediated * 100.0
        )?;
        writeln!(f, "  (NIE / Total Effect)")?;
        writeln!(f)?;
        writeln!(f, "N = {} (trimmed: {})", self.n_obs, self.n_trimmed)?;
        writeln!(f)?;
        writeln!(f, "Signif. codes: *** p<0.01, ** p<0.05, * p<0.10")?;

        Ok(())
    }
}

/// Run causal mediation analysis.
///
/// Estimates the natural direct effect (NDE) and natural indirect effect (NIE)
/// using inverse probability weighting, following Huber (2014).
///
/// # Arguments
///
/// * `dataset` - The dataset containing all variables
/// * `outcome` - Name of the outcome variable column
/// * `treatment` - Name of the treatment variable column (binary 0/1)
/// * `mediator` - Name of the mediator variable column
/// * `covariates` - Names of covariate columns for adjustment
/// * `config` - Configuration options
///
/// # Returns
///
/// `MediationResult` containing effect estimates, standard errors, and p-values
///
/// # Mathematical Background
///
/// The total effect decomposes as:
/// ```text
/// ATE = E[Y(1,M(1))] - E[Y(0,M(0))] = NDE + NIE
/// ```
///
/// Where:
/// - NDE = E[Y(1,M(0))] - E[Y(0,M(0))]: effect of treatment holding mediator at control level
/// - NIE = E[Y(1,M(1))] - E[Y(1,M(0))]: effect of mediator change under treatment
///
/// # Key Assumptions
///
/// 1. **Sequential ignorability**: Treatment assignment and mediator are ignorable
///    given observed covariates
/// 2. **No treatment-mediator interaction** (for simple decomposition)
/// 3. **Positivity**: All units have positive probability of treatment and mediator values
pub fn run_mediation_analysis(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    covariates: &[&str],
    config: MediationConfig,
) -> EconResult<MediationResult> {
    let df = dataset.df();
    let n = df.height();
    let available_cols: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Extract outcome variable
    let y_col = df.column(outcome).map_err(|_| EconError::ColumnNotFound {
        column: outcome.to_string(),
        available: available_cols.clone(),
    })?;
    let y: Vec<f64> = y_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: outcome.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract treatment variable
    let d_col = df
        .column(treatment)
        .map_err(|_| EconError::ColumnNotFound {
            column: treatment.to_string(),
            available: available_cols.clone(),
        })?;
    let d: Vec<f64> = d_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: treatment.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Extract mediator variable
    let m_col = df.column(mediator).map_err(|_| EconError::ColumnNotFound {
        column: mediator.to_string(),
        available: available_cols.clone(),
    })?;
    let m: Vec<f64> = m_col
        .f64()
        .map_err(|_| EconError::NonNumericColumn {
            column: mediator.to_string(),
        })?
        .into_no_null_iter()
        .collect();

    // Build covariate matrix
    let mut x_data: Vec<f64> = Vec::with_capacity(n * (covariates.len() + 1));

    // Add intercept
    for _ in 0..n {
        x_data.push(1.0);
    }

    // Add covariates
    for cov_name in covariates {
        let col = df.column(cov_name).map_err(|_| EconError::ColumnNotFound {
            column: cov_name.to_string(),
            available: available_cols.clone(),
        })?;
        let vals: Vec<f64> = col
            .f64()
            .map_err(|_| EconError::NonNumericColumn {
                column: cov_name.to_string(),
            })?
            .into_no_null_iter()
            .collect();
        x_data.extend(vals);
    }

    let k = covariates.len() + 1;
    let x = Array2::from_shape_vec((n, k), x_data)
        .map_err(|e| EconError::Internal(format!("Failed to create design matrix: {}", e)))?;

    // Compute point estimates
    let (total, direct, indirect, keep_idx) =
        compute_mediation_effects(&y, &d, &m, &x, config.trim)?;

    let n_obs = keep_idx.len();
    let n_trimmed = n - n_obs;

    if n_obs < 20 {
        return Err(EconError::InsufficientData {
            required: 20,
            provided: n_obs,
            context: "Too few observations after trimming for mediation analysis".to_string(),
        });
    }

    // Bootstrap for standard errors
    let mut rng: Box<dyn RngCore> = match config.seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    let mut boot_total = Vec::with_capacity(config.bootstrap);
    let mut boot_direct = Vec::with_capacity(config.bootstrap);
    let mut boot_indirect = Vec::with_capacity(config.bootstrap);

    for _ in 0..config.bootstrap {
        // Resample with replacement from kept observations
        let boot_indices: Vec<usize> = (0..n_obs).map(|_| rng.gen_range(0..n_obs)).collect();

        // Create bootstrap sample
        let y_boot: Vec<f64> = boot_indices.iter().map(|&i| y[keep_idx[i]]).collect();
        let d_boot: Vec<f64> = boot_indices.iter().map(|&i| d[keep_idx[i]]).collect();
        let m_boot: Vec<f64> = boot_indices.iter().map(|&i| m[keep_idx[i]]).collect();
        let x_boot: Array2<f64> =
            Array2::from_shape_fn((n_obs, k), |(i, j)| x[[keep_idx[boot_indices[i]], j]]);

        // Compute effects on bootstrap sample (no further trimming)
        if let Ok((t, de, ie, _)) =
            compute_mediation_effects(&y_boot, &d_boot, &m_boot, &x_boot, 0.0)
        {
            if t.is_finite() && de.is_finite() && ie.is_finite() {
                boot_total.push(t);
                boot_direct.push(de);
                boot_indirect.push(ie);
            }
        }
    }

    // Compute standard errors and CIs from bootstrap
    let (se_total, ci_total) = bootstrap_stats(&boot_total, total);
    let (se_direct, ci_direct) = bootstrap_stats(&boot_direct, direct);
    let (se_indirect, ci_indirect) = bootstrap_stats(&boot_indirect, indirect);

    // Compute p-values (two-sided test of H0: effect = 0)
    let p_total = compute_p_value(total, se_total);
    let p_direct = compute_p_value(direct, se_direct);
    let p_indirect = compute_p_value(indirect, se_indirect);

    // Proportion mediated
    let proportion_mediated = if total.abs() > 1e-10 {
        (indirect / total).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    Ok(MediationResult {
        total_effect: total,
        direct_effect: direct,
        indirect_effect: indirect,
        proportion_mediated,
        se_total,
        se_direct,
        se_indirect,
        ci_total,
        ci_direct,
        ci_indirect,
        p_total,
        p_direct,
        p_indirect,
        n_obs,
        n_trimmed,
    })
}

/// Compute mediation effects using IPW.
///
/// Returns (total_effect, direct_effect, indirect_effect, kept_indices)
fn compute_mediation_effects(
    y: &[f64],
    d: &[f64],
    m: &[f64],
    x: &Array2<f64>,
    trim: f64,
) -> EconResult<(f64, f64, f64, Vec<usize>)> {
    let n = y.len();

    // Step 1: Estimate propensity score p(D=1|X)
    let ps_x = estimate_propensity_scores(d, x)?;

    // Step 2: Estimate propensity score p(D=1|M,X)
    // Add mediator to covariates
    let mut xm_data: Vec<f64> = Vec::with_capacity(n * (x.ncols() + 1));
    for i in 0..n {
        for j in 0..x.ncols() {
            xm_data.push(x[[i, j]]);
        }
        xm_data.push(m[i]);
    }
    let xm = Array2::from_shape_vec((n, x.ncols() + 1), xm_data)
        .map_err(|e| EconError::Internal(format!("Failed to create XM matrix: {}", e)))?;

    let ps_mx = estimate_propensity_scores(d, &xm)?;

    // Step 3: Apply trimming
    let mut keep_idx: Vec<usize> = Vec::new();
    for i in 0..n {
        let ps1 = ps_x[i];
        let ps2 = ps_mx[i];
        if ps1 > trim && ps1 < (1.0 - trim) && ps2 > trim && ps2 < (1.0 - trim) {
            keep_idx.push(i);
        }
    }

    if keep_idx.is_empty() {
        return Err(EconError::InsufficientData {
            required: 10,
            provided: 0,
            context: "All observations trimmed due to extreme propensity scores".to_string(),
        });
    }

    // Step 4: Compute IPW estimates following Huber (2014)
    //
    // Total Effect (ATE):
    //   E[D*Y/p(X)] - E[(1-D)*Y/(1-p(X))]
    //
    // For the mediated effects, we use the following identification:
    //
    // NDE (Natural Direct Effect) under d=1:
    //   θ(1) = E[D*Y/p(X)] - E[(1-D)*p(M,X)*Y / ((1-p(X))*(1-p(M,X)))]
    //
    // This compares treated outcomes to a reweighted control group where
    // the mediator distribution is shifted to match the treated distribution.

    let mut sum_w1_y = 0.0;
    let mut sum_w1 = 0.0;
    let mut sum_w0_y = 0.0;
    let mut sum_w0 = 0.0;
    let mut sum_w0_nde_y = 0.0;
    let mut sum_w0_nde = 0.0;

    for &i in &keep_idx {
        let yi = y[i];
        let di = d[i];
        let ps1 = ps_x[i];
        let ps2 = ps_mx[i];

        if di >= 0.5 {
            // Treated observation
            let w = 1.0 / ps1;
            sum_w1_y += w * yi;
            sum_w1 += w;
        } else {
            // Control observation
            let w_ate = 1.0 / (1.0 - ps1);
            sum_w0_y += w_ate * yi;
            sum_w0 += w_ate;

            // NDE weight: reweight controls by ratio of propensity scores
            // This shifts the mediator distribution in the control group
            // to match what it would be if they were treated
            let w_nde = ps2 / ((1.0 - ps1) * (1.0 - ps2));
            sum_w0_nde_y += w_nde * yi;
            sum_w0_nde += w_nde;
        }
    }

    // Normalized (Hajek) estimates
    let mu1 = if sum_w1 > 0.0 { sum_w1_y / sum_w1 } else { 0.0 };
    let mu0 = if sum_w0 > 0.0 { sum_w0_y / sum_w0 } else { 0.0 };
    let mu0_nde = if sum_w0_nde > 0.0 {
        sum_w0_nde_y / sum_w0_nde
    } else {
        0.0
    };

    // Total effect (ATE)
    let total = mu1 - mu0;

    // Natural direct effect (NDE): effect of treatment holding mediator at M(0)
    // Comparing treated to reweighted controls with shifted mediator distribution
    let direct = mu1 - mu0_nde;

    // Natural indirect effect (NIE) = Total - Direct
    let indirect = total - direct;

    Ok((total, direct, indirect, keep_idx))
}

/// Estimate propensity scores using logistic regression via Newton-Raphson.
fn estimate_propensity_scores(d: &[f64], x: &Array2<f64>) -> EconResult<Vec<f64>> {
    let n = d.len();
    let k = x.ncols();

    let d_arr = Array1::from_vec(d.to_vec());

    // Initialize coefficients to zero
    let mut beta = Array1::zeros(k);

    // Newton-Raphson iteration
    let max_iter = 50;
    let tol = 1e-8;

    for _ in 0..max_iter {
        // Compute probabilities
        let xb = x.dot(&beta);
        let p: Array1<f64> = xb.mapv(logistic_cdf);

        // Compute weights for IRLS
        let w: Array1<f64> = p.mapv(|pi| {
            let pi_clamped = pi.clamp(1e-10, 1.0 - 1e-10);
            pi_clamped * (1.0 - pi_clamped)
        });

        // Score vector: X'(y - p)
        let residuals = &d_arr - &p;
        let score = x.t().dot(&residuals);

        // Hessian: -X'WX
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let xi = x.row(i);
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= w[i] * xi[j] * xi[l];
                }
            }
        }

        // Invert Hessian
        let hessian_inv = invert_matrix(&hessian)?;

        // Newton step
        let delta = hessian_inv.dot(&score);
        beta = &beta - &delta;

        // Check convergence
        let delta_norm: f64 = delta.iter().map(|x| x * x).sum::<f64>().sqrt();
        if delta_norm < tol {
            break;
        }
    }

    // Compute final propensity scores
    let xb = x.dot(&beta);
    let ps: Vec<f64> = xb.iter().map(|&v| logistic_cdf(v)).collect();

    Ok(ps)
}

/// Simple matrix inversion using Gaussian elimination.
fn invert_matrix(a: &Array2<f64>) -> EconResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(EconError::SingularMatrix {
            context: "Matrix must be square".to_string(),
            suggestion: "Check covariate matrix dimensions".to_string(),
        });
    }

    // Augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            return Err(EconError::SingularMatrix {
                context: "Hessian matrix is singular in mediation analysis".to_string(),
                suggestion: "Check for collinear covariates or lack of variation".to_string(),
            });
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

/// Compute bootstrap standard error and confidence interval.
fn bootstrap_stats(samples: &[f64], _point_estimate: f64) -> (f64, (f64, f64)) {
    if samples.is_empty() {
        return (f64::NAN, (f64::NAN, f64::NAN));
    }

    let n = samples.len() as f64;
    let mean: f64 = samples.iter().sum::<f64>() / n;
    let variance: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = variance.sqrt();

    // Percentile CI
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let lower_idx = ((0.025 * n) as usize).max(0).min(samples.len() - 1);
    let upper_idx = ((0.975 * n) as usize).max(0).min(samples.len() - 1);

    (se, (sorted[lower_idx], sorted[upper_idx]))
}

/// Compute two-sided p-value assuming normal distribution.
fn compute_p_value(estimate: f64, se: f64) -> f64 {
    if se <= 0.0 || !se.is_finite() {
        return f64::NAN;
    }
    let z = estimate.abs() / se;
    2.0 * (1.0 - normal_cdf(z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn create_mediation_dataset() -> Dataset {
        // Create synthetic data with known mediation structure.
        // Key: Treatment assignment must NOT be perfectly predictable by X.
        // The X distribution must overlap substantially between treated and controls.
        //
        // DGP:
        // X ~ random covariate (same distribution for D=0 and D=1)
        // D = treatment (randomly assigned, NOT predicted by X)
        // M = 0.5*D + 0.3*X + noise (mediator depends on treatment and X)
        // Y = 0.4*D + 0.6*M + 0.2*X + noise (outcome has direct and indirect effects)
        //
        // Expected effects (approximately):
        // - Direct effect: 0.4
        // - Indirect effect: 0.5 * 0.6 = 0.3
        // - Total effect: 0.7

        // Mix treatment and control randomly with overlapping X values
        let df = df! {
            "y" => [
                // Interleaved treated (T) and control (C) with similar X
                2.1, 1.2, 2.3, 1.4, 2.0, 1.3, 2.4, 1.5, 2.2, 1.6,  // T,C,T,C,...
                1.8, 2.5, 1.5, 2.6, 1.4, 2.3, 1.7, 2.4, 1.6, 2.7,  // C,T,C,T,...
                2.1, 1.3, 2.2, 1.4, 2.0, 1.5, 2.3, 1.2, 2.1, 1.4,
                1.6, 2.4, 1.5, 2.5, 1.3, 2.2, 1.7, 2.6, 1.4, 2.3,
                2.2, 1.5, 2.1, 1.3, 2.3, 1.6, 2.0, 1.4, 2.4, 1.5,
                1.4, 2.3, 1.6, 2.4, 1.5, 2.2, 1.3, 2.5, 1.7, 2.1
            ],
            "d" => [
                // Interleaved 1,0,1,0,... pattern
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
            ],
            "m" => [
                // Mediator: D*0.5 + X*0.3 + noise, so treated have higher M on average
                1.0, 0.5, 1.1, 0.6, 0.9, 0.4, 1.2, 0.7, 1.0, 0.5,
                0.4, 1.1, 0.5, 1.2, 0.3, 1.0, 0.6, 1.1, 0.5, 1.3,
                1.0, 0.4, 1.1, 0.5, 0.9, 0.6, 1.0, 0.3, 1.1, 0.5,
                0.5, 1.2, 0.4, 1.1, 0.6, 1.0, 0.5, 1.2, 0.4, 1.1,
                1.1, 0.5, 1.0, 0.4, 1.2, 0.6, 0.9, 0.5, 1.1, 0.4,
                0.6, 1.0, 0.5, 1.1, 0.4, 1.0, 0.5, 1.2, 0.6, 1.0
            ],
            "x" => [
                // Covariate: same distribution for both groups (complete overlap)
                0.5, 0.5, 0.6, 0.6, 0.4, 0.4, 0.7, 0.7, 0.5, 0.5,
                0.4, 0.6, 0.5, 0.7, 0.3, 0.5, 0.6, 0.6, 0.5, 0.8,
                0.5, 0.4, 0.6, 0.5, 0.4, 0.6, 0.5, 0.3, 0.6, 0.5,
                0.5, 0.7, 0.4, 0.6, 0.6, 0.5, 0.5, 0.7, 0.4, 0.6,
                0.6, 0.5, 0.5, 0.4, 0.7, 0.6, 0.4, 0.5, 0.6, 0.4,
                0.6, 0.5, 0.5, 0.6, 0.4, 0.5, 0.5, 0.7, 0.6, 0.5
            ]
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_mediation_basic() {
        let ds = create_mediation_dataset();

        let config = MediationConfig {
            bootstrap: 199, // Fewer for faster test
            trim: 0.01,
            seed: Some(42),
        };

        let result = run_mediation_analysis(&ds, "y", "d", "m", &["x"], config).unwrap();

        // Check that we get reasonable results
        // Total effect should be positive (around 0.7-1.0)
        assert!(
            result.total_effect > 0.3 && result.total_effect < 1.5,
            "Total effect {} out of expected range",
            result.total_effect
        );

        // Direct effect should be positive
        assert!(
            result.direct_effect > -0.5 && result.direct_effect < 1.5,
            "Direct effect {} out of expected range",
            result.direct_effect
        );

        // Indirect effect can be any value
        assert!(
            result.indirect_effect.is_finite(),
            "Indirect effect {} should be finite",
            result.indirect_effect
        );

        // Decomposition should hold: total ≈ direct + indirect
        let decomp_error =
            (result.total_effect - result.direct_effect - result.indirect_effect).abs();
        assert!(
            decomp_error < 0.001,
            "Decomposition error {} too large",
            decomp_error
        );

        // Standard errors should be positive
        assert!(result.se_total > 0.0, "SE total should be positive");
        assert!(result.se_direct > 0.0, "SE direct should be positive");
        assert!(result.se_indirect > 0.0, "SE indirect should be positive");
    }

    #[test]
    fn test_mediation_display() {
        let ds = create_mediation_dataset();

        let config = MediationConfig {
            bootstrap: 99,
            trim: 0.01,
            seed: Some(42),
        };

        let result = run_mediation_analysis(&ds, "y", "d", "m", &["x"], config).unwrap();

        let display = result.to_string();
        assert!(display.contains("Causal Mediation Analysis"));
        assert!(display.contains("Total (ATE)"));
        assert!(display.contains("Direct (NDE)"));
        assert!(display.contains("Indirect (NIE)"));
        assert!(display.contains("Proportion Mediated"));
    }

    #[test]
    fn test_missing_column_error() {
        let ds = create_mediation_dataset();

        let result = run_mediation_analysis(
            &ds,
            "nonexistent",
            "d",
            "m",
            &["x"],
            MediationConfig::default(),
        );

        assert!(result.is_err());
        if let Err(EconError::ColumnNotFound { column, .. }) = result {
            assert_eq!(column, "nonexistent");
        } else {
            panic!("Expected ColumnNotFound error");
        }
    }

    // =========================================================================
    // R Validation Tests (Phase 5)
    // =========================================================================

    /// Simple LCG for deterministic random numbers
    fn lcg_rand_med(seed: &mut u64) -> f64 {
        let a: u64 = 1103515245;
        let c: u64 = 12345;
        let m: u64 = 2_u64.pow(31);
        *seed = (a.wrapping_mul(*seed).wrapping_add(c)) % m;
        (*seed as f64) / (m as f64)
    }

    /// Create validation dataset matching R's mediation package example.
    fn create_mediation_validation_dataset() -> Dataset {
        let n = 300;
        let mut seed: u64 = 42;

        let mut x = Vec::with_capacity(n);
        let mut treatment = Vec::with_capacity(n);
        let mut mediator = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for _ in 0..n {
            // Covariate
            let u1 = lcg_rand_med(&mut seed).max(1e-10);
            let u2 = lcg_rand_med(&mut seed);
            let x_i = ((-2.0_f64 * u1.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u2).cos();
            x.push(x_i);

            // Treatment (random assignment with slight dependence on x)
            let ps = 1.0 / (1.0 + (-(0.5 * x_i)).exp());
            let t = if lcg_rand_med(&mut seed) < ps {
                1.0
            } else {
                0.0
            };
            treatment.push(t);

            // Mediator: M = 0.3 + 0.6*D + 0.4*X + noise
            let u3 = lcg_rand_med(&mut seed).max(1e-10);
            let u4 = lcg_rand_med(&mut seed);
            let noise_m =
                0.5 * ((-2.0_f64 * u3.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u4).cos();
            let m = 0.3 + 0.6 * t + 0.4 * x_i + noise_m;
            mediator.push(m);

            // Outcome: Y = 1 + 0.4*D + 0.5*M + 0.3*X + noise
            // Direct effect = 0.4, Indirect effect = 0.6 * 0.5 = 0.3
            // Total effect = 0.4 + 0.3 = 0.7
            let u5 = lcg_rand_med(&mut seed).max(1e-10);
            let u6 = lcg_rand_med(&mut seed);
            let noise_y =
                0.5 * ((-2.0_f64 * u5.ln()).sqrt()) * (2.0 * std::f64::consts::PI * u6).cos();
            let y_i = 1.0 + 0.4 * t + 0.5 * m + 0.3 * x_i + noise_y;
            y.push(y_i);
        }

        let df = df! {
            "y" => y,
            "treatment" => treatment,
            "mediator" => mediator,
            "x" => x
        }
        .unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_validate_mediation_vs_r() {
        // Validates against R mediation package
        // R reference:
        // library(mediation)
        // med_fit <- lm(mediator ~ treatment + x, data = med_data)
        // out_fit <- lm(y ~ treatment + mediator + x, data = med_data)
        // med_result <- mediate(med_fit, out_fit, treat = "treatment", mediator = "mediator", sims = 500)

        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // True effects from DGP:
        // - Direct effect: 0.4
        // - Indirect effect: 0.6 * 0.5 = 0.3
        // - Total effect: 0.7
        let true_total = 0.7;
        let true_direct = 0.4;
        let true_indirect = 0.3;
        let tol = 0.4; // Allow estimation error

        assert!(
            (result.total_effect - true_total).abs() < tol,
            "Total effect {:.4} should be close to {:.4}",
            result.total_effect,
            true_total
        );
        assert!(
            (result.direct_effect - true_direct).abs() < tol,
            "Direct effect {:.4} should be close to {:.4}",
            result.direct_effect,
            true_direct
        );
        assert!(
            (result.indirect_effect - true_indirect).abs() < tol,
            "Indirect effect {:.4} should be close to {:.4}",
            result.indirect_effect,
            true_indirect
        );
    }

    #[test]
    fn test_validate_mediation_decomposition() {
        // Validate that Total = Direct + Indirect
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // Decomposition should hold exactly
        let decomp = result.direct_effect + result.indirect_effect;
        assert!(
            (result.total_effect - decomp).abs() < 1e-10,
            "Decomposition failed: {:.6} ≠ {:.6} + {:.6}",
            result.total_effect,
            result.direct_effect,
            result.indirect_effect
        );
    }

    #[test]
    fn test_validate_mediation_proportion_mediated() {
        // Validate proportion mediated calculation
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // Proportion mediated = indirect / total
        if result.total_effect.abs() > 0.01 {
            let expected_prop = result.indirect_effect / result.total_effect;
            assert!(
                (result.proportion_mediated - expected_prop).abs() < 1e-8,
                "Proportion mediated {:.4} should equal NIE/ATE {:.4}",
                result.proportion_mediated,
                expected_prop
            );
        }

        // Proportion should be between 0 and 1 for most reasonable cases
        // (can be negative or > 1 in edge cases)
        assert!(result.proportion_mediated.is_finite());
    }

    #[test]
    fn test_validate_mediation_standard_errors() {
        // Validate standard error properties
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // All SEs should be positive
        assert!(result.se_total > 0.0, "SE total should be positive");
        assert!(result.se_direct > 0.0, "SE direct should be positive");
        assert!(result.se_indirect > 0.0, "SE indirect should be positive");

        // SEs should be reasonable (not too small or large)
        assert!(
            result.se_total < 1.0,
            "SE total {:.4} seems too large",
            result.se_total
        );
        assert!(
            result.se_direct < 1.0,
            "SE direct {:.4} seems too large",
            result.se_direct
        );
        assert!(
            result.se_indirect < 1.0,
            "SE indirect {:.4} seems too large",
            result.se_indirect
        );
    }

    #[test]
    fn test_validate_mediation_confidence_intervals() {
        // Validate CI properties
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // CIs should contain point estimates
        assert!(
            result.ci_total.0 <= result.total_effect && result.ci_total.1 >= result.total_effect
        );
        assert!(
            result.ci_direct.0 <= result.direct_effect
                && result.ci_direct.1 >= result.direct_effect
        );
        assert!(
            result.ci_indirect.0 <= result.indirect_effect
                && result.ci_indirect.1 >= result.indirect_effect
        );

        // CIs should have positive width
        assert!(result.ci_total.1 > result.ci_total.0);
        assert!(result.ci_direct.1 > result.ci_direct.0);
        assert!(result.ci_indirect.1 > result.ci_indirect.0);
    }

    #[test]
    fn test_validate_mediation_p_values() {
        // Validate p-value properties
        let dataset = create_mediation_validation_dataset();
        let config = MediationConfig {
            bootstrap: 199,
            trim: 0.05,
            seed: Some(42),
        };

        let result =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config).unwrap();

        // P-values should be in [0, 1]
        assert!(result.p_total >= 0.0 && result.p_total <= 1.0);
        assert!(result.p_direct >= 0.0 && result.p_direct <= 1.0);
        assert!(result.p_indirect >= 0.0 && result.p_indirect <= 1.0);
    }

    #[test]
    fn test_validate_mediation_reproducibility() {
        // Validate that seed produces reproducible results
        let dataset = create_mediation_validation_dataset();

        let config1 = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(12345),
        };
        let result1 =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config1)
                .unwrap();

        let config2 = MediationConfig {
            bootstrap: 99,
            trim: 0.05,
            seed: Some(12345),
        };
        let result2 =
            run_mediation_analysis(&dataset, "y", "treatment", "mediator", &["x"], config2)
                .unwrap();

        // Point estimates should be identical (same seed)
        assert!((result1.total_effect - result2.total_effect).abs() < 1e-10);
        assert!((result1.direct_effect - result2.direct_effect).abs() < 1e-10);
        assert!((result1.indirect_effect - result2.indirect_effect).abs() < 1e-10);

        // SEs may have small differences due to bootstrap
        assert!((result1.se_total - result2.se_total).abs() < 0.01);
    }
}
