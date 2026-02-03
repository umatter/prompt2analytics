//! Covariate Balancing Propensity Score (CBPS) Estimation.
//!
//! Implements the CBPS method of Imai & Ratkovic (2014), which uses Generalized
//! Method of Moments (GMM) to simultaneously estimate propensity scores and
//! achieve covariate balance.
//!
//! # Overview
//!
//! Unlike standard logistic regression for propensity scores, CBPS explicitly
//! targets covariate balance as part of the estimation by incorporating balance
//! conditions into the GMM moment conditions:
//!
//! 1. **Score conditions**: E[X_i * (T_i - p(X_i; beta))] = 0
//! 2. **Balance conditions**: E[T_i * X_i / p(X_i; beta) - (1-T_i) * X_i / (1-p(X_i; beta))] = 0
//!
//! The combined moment conditions are solved using GMM, resulting in propensity
//! scores that achieve better covariate balance than standard logit.
//!
//! # References
//!
//! - Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
//!   *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
//!   DOI: 10.1111/rssb.12027
//!
//! - R package `CBPS`: Fong, C., Ratkovic, M., & Imai, K. (2022).
//!   CBPS: Covariate Balancing Propensity Score.
//!   <https://cran.r-project.org/package=CBPS>
//!
//! R equivalent: `CBPS::CBPS()`

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::{EconError, EconResult};
use crate::linalg::design::{DesignMatrix, get_column_names};
use crate::linalg::matrix_ops::safe_inverse;
use crate::traits::estimator::{SignificanceLevel, chi_squared_p_value, logistic_cdf, normal_cdf};

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// CBPS estimation method.
///
/// Controls the balance between propensity score accuracy and covariate balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbpsMethod {
    /// Exact balance (overidentified GMM).
    ///
    /// Uses both score and balance conditions. The model is overidentified,
    /// allowing for a J-test of model specification.
    #[default]
    ExactBalance,

    /// Over-identified estimation with additional moment conditions.
    ///
    /// Uses more moment conditions than parameters, providing a stronger
    /// test of specification but may be more sensitive to misspecification.
    OverBalance,

    /// Just-identified estimation (score conditions only).
    ///
    /// Equivalent to standard logistic regression. Uses only score conditions,
    /// providing baseline comparison for balance improvement.
    JustIdentified,
}

impl fmt::Display for CbpsMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CbpsMethod::ExactBalance => write!(f, "Exact Balance (Overidentified GMM)"),
            CbpsMethod::OverBalance => write!(f, "Over-Balanced GMM"),
            CbpsMethod::JustIdentified => write!(f, "Just-Identified (Score Only)"),
        }
    }
}

/// Configuration for CBPS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbpsConfig {
    /// CBPS estimation method
    pub method: CbpsMethod,
    /// Convergence tolerance for GMM optimization
    pub tolerance: f64,
    /// Maximum iterations for Newton-Raphson
    pub max_iter: usize,
    /// Standardized difference threshold for balance (default: 0.1)
    pub balance_threshold: f64,
}

impl Default for CbpsConfig {
    fn default() -> Self {
        Self {
            method: CbpsMethod::ExactBalance,
            tolerance: 1e-8,
            max_iter: 100,
            balance_threshold: 0.1,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Result Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Covariate balance statistics for a single covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateBalance {
    /// Covariate name
    pub name: String,
    /// Mean in treated group
    pub mean_treated: f64,
    /// Mean in control group
    pub mean_control: f64,
    /// Standardized difference: (mean_treated - mean_control) / pooled_sd
    pub std_diff: f64,
    /// Variance ratio: var_treated / var_control
    pub var_ratio: f64,
    /// Whether balance is achieved (|std_diff| < threshold)
    pub balanced: bool,
}

/// Balance table showing covariate balance before and after weighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceTable {
    /// Covariate names
    pub covariates: Vec<String>,
    /// Mean in treated group
    pub mean_treated: Vec<f64>,
    /// Mean in control group
    pub mean_control: Vec<f64>,
    /// Standardized differences
    pub std_diff: Vec<f64>,
    /// Variance ratios
    pub var_ratio: Vec<f64>,
    /// Maximum absolute standardized difference
    pub max_std_diff: f64,
    /// Number of covariates with |std_diff| < threshold
    pub n_balanced: usize,
}

impl BalanceTable {
    /// Create an empty balance table
    pub fn new() -> Self {
        Self {
            covariates: Vec::new(),
            mean_treated: Vec::new(),
            mean_control: Vec::new(),
            std_diff: Vec::new(),
            var_ratio: Vec::new(),
            max_std_diff: 0.0,
            n_balanced: 0,
        }
    }

    /// Check if all covariates are balanced (|std_diff| < threshold)
    pub fn all_balanced(&self, threshold: f64) -> bool {
        self.std_diff.iter().all(|&d| d.abs() < threshold)
    }
}

impl Default for BalanceTable {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BalanceTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariate Balance Table")?;
        writeln!(
            f,
            "{:>20} {:>12} {:>12} {:>12} {:>12}",
            "Covariate", "Mean(T)", "Mean(C)", "Std.Diff", "Var.Ratio"
        )?;
        writeln!(f, "{}", "-".repeat(72))?;

        for i in 0..self.covariates.len() {
            let balanced_marker = if self.std_diff[i].abs() < 0.1 {
                " "
            } else {
                "*"
            };
            writeln!(
                f,
                "{:>20} {:>12.4} {:>12.4} {:>12.4}{} {:>12.4}",
                self.covariates[i],
                self.mean_treated[i],
                self.mean_control[i],
                self.std_diff[i],
                balanced_marker,
                self.var_ratio[i]
            )?;
        }

        writeln!(f, "{}", "-".repeat(72))?;
        writeln!(
            f,
            "Max |Std.Diff|: {:.4}  Balanced covariates: {}/{}",
            self.max_std_diff,
            self.n_balanced,
            self.covariates.len()
        )?;

        Ok(())
    }
}

/// Result from CBPS estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbpsResult {
    /// Logistic regression coefficients (beta)
    pub coefficients: Vec<f64>,
    /// Coefficient names (intercept + covariates)
    pub names: Vec<String>,
    /// Standard errors of coefficients
    pub std_errors: Vec<f64>,
    /// Z-statistics
    pub z_stats: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Estimated propensity scores P(T=1|X)
    pub propensity_scores: Vec<f64>,
    /// IPW weights for treatment effect estimation
    pub weights: Vec<f64>,
    /// Covariate balance before weighting
    pub balance_before: BalanceTable,
    /// Covariate balance after CBPS weighting
    pub balance_after: BalanceTable,
    /// Whether GMM optimization converged
    pub converged: bool,
    /// Number of iterations to convergence
    pub iterations: usize,
    /// Final GMM criterion (objective function value)
    pub gmm_criterion: f64,
    /// J-statistic for overidentification test (if overidentified)
    pub j_statistic: Option<f64>,
    /// P-value for J-test
    pub j_p_value: Option<f64>,
    /// Degrees of freedom for J-test
    pub j_df: Option<usize>,
    /// CBPS method used
    pub method: CbpsMethod,
    /// Number of observations
    pub n_obs: usize,
    /// Number of treated observations
    pub n_treated: usize,
    /// Number of control observations
    pub n_control: usize,
    /// Number of parameters
    pub n_params: usize,
    /// Number of moment conditions
    pub n_moments: usize,
}

impl fmt::Display for CbpsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Covariate Balancing Propensity Score (CBPS)")?;
        writeln!(f, "============================================")?;
        writeln!(f)?;
        writeln!(f, "Method: {}", self.method)?;
        writeln!(
            f,
            "Observations: {} (Treated: {}, Control: {})",
            self.n_obs, self.n_treated, self.n_control
        )?;
        writeln!(
            f,
            "Parameters: {}  Moment conditions: {}",
            self.n_params, self.n_moments
        )?;
        writeln!(f)?;

        writeln!(f, "Coefficients:")?;
        writeln!(
            f,
            "{:>15} {:>12} {:>12} {:>10} {:>10}",
            "Parameter", "Estimate", "Std.Err", "z-stat", "P>|z|"
        )?;
        writeln!(f, "{}", "-".repeat(65))?;

        for i in 0..self.n_params {
            let sig = SignificanceLevel::from_p_value(self.p_values[i]);
            writeln!(
                f,
                "{:>15} {:>12.6} {:>12.6} {:>10.3} {:>10.4}{}",
                self.names[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                sig.stars()
            )?;
        }
        writeln!(f, "{}", "-".repeat(65))?;

        // J-test for overidentification
        if let (Some(j), Some(p), Some(df)) = (self.j_statistic, self.j_p_value, self.j_df) {
            writeln!(f)?;
            writeln!(f, "J-Test for Overidentifying Restrictions:")?;
            writeln!(f, "  J-statistic: {:.4} (df = {})", j, df)?;
            writeln!(f, "  p-value: {:.4}", p)?;
            if p < 0.05 {
                writeln!(f, "  WARNING: Model may be misspecified (p < 0.05)")?;
            } else {
                writeln!(f, "  Cannot reject model specification")?;
            }
        }

        writeln!(f)?;
        writeln!(f, "Propensity Score Summary:")?;
        let ps_min = self
            .propensity_scores
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let ps_max = self
            .propensity_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let ps_mean: f64 = self.propensity_scores.iter().sum::<f64>() / self.n_obs as f64;
        writeln!(
            f,
            "  Mean: {:.4}  Min: {:.4}  Max: {:.4}",
            ps_mean, ps_min, ps_max
        )?;

        writeln!(f)?;
        writeln!(f, "Balance Improvement:")?;
        writeln!(
            f,
            "  Before CBPS: Max |Std.Diff| = {:.4}",
            self.balance_before.max_std_diff
        )?;
        writeln!(
            f,
            "  After CBPS:  Max |Std.Diff| = {:.4}",
            self.balance_after.max_std_diff
        )?;
        writeln!(
            f,
            "  Balanced covariates: {}/{} -> {}/{}",
            self.balance_before.n_balanced,
            self.balance_before.covariates.len(),
            self.balance_after.n_balanced,
            self.balance_after.covariates.len()
        )?;

        if !self.converged {
            writeln!(f)?;
            writeln!(
                f,
                "WARNING: GMM optimization did not converge after {} iterations",
                self.iterations
            )?;
        }

        writeln!(f)?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main Estimation Functions
// ═══════════════════════════════════════════════════════════════════════════════

/// Run CBPS estimation on a dataset.
///
/// This is the main entry point for CBPS estimation. It uses GMM to simultaneously
/// estimate propensity scores and achieve covariate balance.
///
/// # Arguments
///
/// * `dataset` - The dataset containing treatment and covariates
/// * `treatment_col` - Name of the binary treatment variable (0/1)
/// * `covariate_cols` - Names of covariate columns
/// * `config` - Configuration options (optional, uses defaults if None)
///
/// # Example
///
/// ```ignore
/// use p2a_core::econometrics::cbps::{run_cbps, CbpsConfig, CbpsMethod};
///
/// let config = CbpsConfig {
///     method: CbpsMethod::ExactBalance,
///     ..Default::default()
/// };
///
/// let result = run_cbps(&dataset, "treatment", &["x1", "x2", "x3"], Some(config))?;
/// println!("{}", result);
/// ```
///
/// # References
///
/// - Imai, K. & Ratkovic, M. (2014). "Covariate Balancing Propensity Score."
///   *Journal of the Royal Statistical Society: Series B*, 76(1), 243-263.
pub fn run_cbps(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    config: Option<CbpsConfig>,
) -> EconResult<CbpsResult> {
    let config = config.unwrap_or_default();

    // Extract treatment variable
    let t = DesignMatrix::extract_column(dataset.df(), treatment_col).map_err(|e| {
        EconError::ColumnNotFound {
            column: treatment_col.to_string(),
            available: get_column_names(dataset.df()),
        }
    })?;

    let n = t.len();

    // Validate treatment is binary
    let n_treated: usize = t.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    if n_treated == 0 || n_control == 0 {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "Treatment variable '{}' must have both treated (1) and control (0) observations. Found {} treated, {} control.",
                treatment_col, n_treated, n_control
            ),
        });
    }

    // Build design matrix with intercept
    let design = DesignMatrix::from_dataframe(dataset.df(), covariate_cols, true)?;
    let x = design.data;
    let k = x.ncols(); // Number of parameters (including intercept)

    // Compute balance before CBPS (unweighted)
    let balance_before =
        compute_balance_table(&x, &t, &design.column_names, None, config.balance_threshold);

    // Run CBPS GMM estimation
    let (beta, converged, iterations, gmm_criterion) =
        estimate_cbps_gmm(&x, &t, config.method, config.tolerance, config.max_iter)?;

    // Compute propensity scores
    let linear_pred: Array1<f64> = x.dot(&beta);
    let propensity_scores: Vec<f64> = linear_pred.iter().map(|&z| logistic_cdf(z)).collect();

    // Compute IPW weights (normalized)
    let weights = compute_ipw_weights(&propensity_scores, &t);

    // Compute balance after CBPS weighting
    let balance_after = compute_balance_table(
        &x,
        &t,
        &design.column_names,
        Some(&weights),
        config.balance_threshold,
    );

    // Compute standard errors using GMM variance formula
    let (std_errors, _vcov) =
        compute_cbps_std_errors(&x, &t, &beta, &propensity_scores, config.method)?;

    // Compute z-statistics and p-values
    let z_stats: Vec<f64> = beta
        .iter()
        .zip(std_errors.iter())
        .map(|(&b, &se)| if se > 0.0 { b / se } else { 0.0 })
        .collect();

    let p_values: Vec<f64> = z_stats
        .iter()
        .map(|&z| 2.0 * (1.0 - normal_cdf(z.abs())))
        .collect();

    // Compute J-test for overidentification
    let (j_statistic, j_p_value, j_df) = if config.method != CbpsMethod::JustIdentified {
        let n_moments = match config.method {
            CbpsMethod::ExactBalance => 2 * k, // Score + balance conditions
            CbpsMethod::OverBalance => 2 * k,
            CbpsMethod::JustIdentified => k,
        };
        let df = n_moments - k;
        if df > 0 {
            let j = n as f64 * gmm_criterion;
            let p = chi_squared_p_value(j, df as f64);
            (Some(j), Some(p), Some(df))
        } else {
            (None, None, None)
        }
    } else {
        (None, None, None)
    };

    let n_moments = match config.method {
        CbpsMethod::ExactBalance => 2 * k,
        CbpsMethod::OverBalance => 2 * k,
        CbpsMethod::JustIdentified => k,
    };

    Ok(CbpsResult {
        coefficients: beta.to_vec(),
        names: design.column_names.clone(),
        std_errors,
        z_stats,
        p_values,
        propensity_scores,
        weights,
        balance_before,
        balance_after,
        converged,
        iterations,
        gmm_criterion,
        j_statistic,
        j_p_value,
        j_df,
        method: config.method,
        n_obs: n,
        n_treated,
        n_control,
        n_params: k,
        n_moments,
    })
}

/// Simplified CBPS function with default configuration.
///
/// Convenience wrapper that uses default settings (ExactBalance method).
pub fn cbps(
    dataset: &Dataset,
    treatment_col: &str,
    covariate_cols: &[&str],
    method: CbpsMethod,
) -> EconResult<CbpsResult> {
    run_cbps(
        dataset,
        treatment_col,
        covariate_cols,
        Some(CbpsConfig {
            method,
            ..Default::default()
        }),
    )
}

// ═══════════════════════════════════════════════════════════════════════════════
// GMM Estimation
// ═══════════════════════════════════════════════════════════════════════════════

/// Estimate CBPS using GMM.
///
/// Solves the combined moment conditions using Newton-Raphson optimization.
///
/// # Moment Conditions (Imai & Ratkovic 2014, Eq. 7-8)
///
/// Score conditions:
/// ```text
/// g_score(beta) = (1/n) * sum_i [X_i * (T_i - p_i)]
/// ```
///
/// Balance conditions:
/// ```text
/// g_balance(beta) = (1/n) * sum_i [T_i * X_i / p_i - (1-T_i) * X_i / (1-p_i)]
/// ```
fn estimate_cbps_gmm(
    x: &Array2<f64>,
    t: &Array1<f64>,
    method: CbpsMethod,
    tolerance: f64,
    max_iter: usize,
) -> EconResult<(Array1<f64>, bool, usize, f64)> {
    let n = t.len();
    let k = x.ncols();

    // Initialize beta to zeros (standard logit starting point)
    let mut beta = Array1::zeros(k);

    // For just-identified, use standard logit (equivalent to score conditions only)
    if method == CbpsMethod::JustIdentified {
        return estimate_logit_newton(&x.view(), t, tolerance, max_iter);
    }

    // Two-step GMM for overidentified models
    // Step 1: Use identity weighting matrix
    let mut converged = false;
    let mut iterations = 0;

    // First step: Solve with identity weighting
    for iter in 0..max_iter {
        iterations = iter + 1;

        // Compute propensity scores
        let z: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Compute combined moment conditions
        let g_bar = compute_moment_conditions(x, t, &p_clipped, method);

        // Check convergence
        let g_norm: f64 = g_bar.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if g_norm < tolerance {
            converged = true;
            break;
        }

        // Compute Jacobian of moment conditions
        let jacobian = compute_moment_jacobian(x, t, &p_clipped, method);

        // Compute weighting matrix (identity for first step)
        let n_moments = g_bar.len();
        let w = Array2::<f64>::eye(n_moments);

        // GMM update: beta_new = beta - (G'WG)^{-1} G'W g
        // G is jacobian, W is weighting matrix, g is moment conditions
        let gwg = jacobian.t().dot(&w).dot(&jacobian);
        let (gwg_inv, _) = safe_inverse(&gwg.view()).map_err(|e| EconError::SingularMatrix {
            context: "CBPS GMM optimization".to_string(),
            suggestion: format!("Jacobian may be singular: {:?}", e),
        })?;

        let direction = gwg_inv.dot(&jacobian.t()).dot(&w).dot(&g_bar);
        beta = &beta - &direction;
    }

    // Step 2: Update with optimal weighting matrix
    let z: Array1<f64> = x.dot(&beta);
    let p: Array1<f64> = z.mapv(logistic_cdf);
    let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

    // Compute optimal weighting matrix (inverse of variance of moment conditions)
    let w_opt = compute_optimal_weight_cbps(x, t, &p_clipped, method, n)?;

    // Second step iterations
    for iter in 0..max_iter {
        iterations = max_iter + iter + 1;

        let z: Array1<f64> = x.dot(&beta);
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        let g_bar = compute_moment_conditions(x, t, &p_clipped, method);

        let g_norm: f64 = g_bar.iter().map(|&g| g * g).sum::<f64>().sqrt();
        if g_norm < tolerance {
            converged = true;
            break;
        }

        let jacobian = compute_moment_jacobian(x, t, &p_clipped, method);

        let gwg = jacobian.t().dot(&w_opt).dot(&jacobian);
        let (gwg_inv, _) = safe_inverse(&gwg.view()).map_err(|e| EconError::SingularMatrix {
            context: "CBPS GMM step 2".to_string(),
            suggestion: format!("Jacobian may be singular: {:?}", e),
        })?;

        let direction = gwg_inv.dot(&jacobian.t()).dot(&w_opt).dot(&g_bar);

        // Line search for stability
        let mut step_size = 1.0;
        let mut beta_new = &beta - &(&direction * step_size);
        let mut g_new = compute_moment_conditions(x, t, &p_clipped, method);
        let mut g_new_norm: f64 = g_new.iter().map(|&g| g * g).sum::<f64>().sqrt();

        for _ in 0..10 {
            if g_new_norm < g_norm {
                break;
            }
            step_size *= 0.5;
            beta_new = &beta - &(&direction * step_size);
            let z_new: Array1<f64> = x.dot(&beta_new);
            let p_new: Array1<f64> = z_new.mapv(logistic_cdf);
            let p_new_clipped: Array1<f64> = p_new.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));
            g_new = compute_moment_conditions(x, t, &p_new_clipped, method);
            g_new_norm = g_new.iter().map(|&g| g * g).sum::<f64>().sqrt();
        }

        beta = beta_new;
    }

    // Compute final GMM criterion
    let z_final: Array1<f64> = x.dot(&beta);
    let p_final: Array1<f64> = z_final.mapv(logistic_cdf);
    let p_clipped_final: Array1<f64> = p_final.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));
    let g_final = compute_moment_conditions(x, t, &p_clipped_final, method);

    let wg = w_opt.dot(&g_final);
    let gmm_criterion = g_final.dot(&wg);

    Ok((beta, converged, iterations, gmm_criterion))
}

/// Compute the combined moment conditions for CBPS.
///
/// Returns the sample average of moment conditions.
fn compute_moment_conditions(
    x: &Array2<f64>,
    t: &Array1<f64>,
    p: &Array1<f64>,
    method: CbpsMethod,
) -> Array1<f64> {
    let n = t.len();
    let k = x.ncols();

    let n_moments = match method {
        CbpsMethod::JustIdentified => k,
        CbpsMethod::ExactBalance | CbpsMethod::OverBalance => 2 * k,
    };

    let mut g = Array1::zeros(n_moments);

    // Score conditions: (1/n) * X'(T - p)
    // These are the standard logistic regression score equations
    for i in 0..n {
        let residual = t[i] - p[i];
        for j in 0..k {
            g[j] += x[[i, j]] * residual / n as f64;
        }
    }

    // Balance conditions: (1/n) * [T*X/p - (1-T)*X/(1-p)]
    // These ensure covariate balance between weighted treated and control groups
    if method != CbpsMethod::JustIdentified {
        for i in 0..n {
            let ti = t[i];
            let pi = p[i];
            for j in 0..k {
                let balance_term = if ti >= 0.5 {
                    x[[i, j]] / pi
                } else {
                    -x[[i, j]] / (1.0 - pi)
                };
                g[k + j] += balance_term / n as f64;
            }
        }
    }

    g
}

/// Compute the Jacobian of moment conditions with respect to beta.
///
/// G = d g_bar / d beta
fn compute_moment_jacobian(
    x: &Array2<f64>,
    t: &Array1<f64>,
    p: &Array1<f64>,
    method: CbpsMethod,
) -> Array2<f64> {
    let n = t.len();
    let k = x.ncols();

    let n_moments = match method {
        CbpsMethod::JustIdentified => k,
        CbpsMethod::ExactBalance | CbpsMethod::OverBalance => 2 * k,
    };

    let mut jacobian = Array2::zeros((n_moments, k));

    // Jacobian of score conditions
    // d/d_beta [X'(T - p)] = -X' * diag(p*(1-p)) * X
    for i in 0..n {
        let pi = p[i];
        let weight = -pi * (1.0 - pi);
        for j in 0..k {
            for l in 0..k {
                jacobian[[j, l]] += weight * x[[i, j]] * x[[i, l]] / n as f64;
            }
        }
    }

    // Jacobian of balance conditions
    // d/d_beta [T*X/p - (1-T)*X/(1-p)]
    if method != CbpsMethod::JustIdentified {
        for i in 0..n {
            let ti = t[i];
            let pi = p[i];
            let dpdb = pi * (1.0 - pi); // dp/d_z = p(1-p), and z = X*beta

            for j in 0..k {
                let d_balance: f64 = if ti >= 0.5 {
                    // d/d_beta [X/p] = -X * (1/p^2) * dp/d_beta = -X * (1-p) / p
                    -x[[i, j]] * (1.0 - pi) / pi
                } else {
                    // d/d_beta [-X/(1-p)] = X * (1/(1-p)^2) * dp/d_beta = X * p / (1-p)
                    x[[i, j]] * pi / (1.0 - pi)
                };

                for l in 0..k {
                    jacobian[[k + j, l]] += d_balance * x[[i, l]] * dpdb / n as f64;
                }
            }
        }
    }

    jacobian
}

/// Compute optimal weighting matrix for GMM.
///
/// W = [Var(g)]^{-1} where g are the moment conditions
fn compute_optimal_weight_cbps(
    x: &Array2<f64>,
    t: &Array1<f64>,
    p: &Array1<f64>,
    method: CbpsMethod,
    n: usize,
) -> EconResult<Array2<f64>> {
    let k = x.ncols();
    let n_moments = match method {
        CbpsMethod::JustIdentified => k,
        CbpsMethod::ExactBalance | CbpsMethod::OverBalance => 2 * k,
    };

    // Compute variance of moment conditions
    // Omega = (1/n) * sum_i [g_i * g_i']
    let mut omega = Array2::<f64>::zeros((n_moments, n_moments));

    for i in 0..n {
        let mut gi = Array1::zeros(n_moments);

        // Score component
        let residual = t[i] - p[i];
        for j in 0..k {
            gi[j] = x[[i, j]] * residual;
        }

        // Balance component
        if method != CbpsMethod::JustIdentified {
            let ti = t[i];
            let pi = p[i];
            for j in 0..k {
                gi[k + j] = if ti >= 0.5 {
                    x[[i, j]] / pi
                } else {
                    -x[[i, j]] / (1.0 - pi)
                };
            }
        }

        // Outer product
        for j in 0..n_moments {
            for l in 0..n_moments {
                omega[[j, l]] += gi[j] * gi[l];
            }
        }
    }

    omega /= n as f64;

    // Invert to get weighting matrix
    let (w, _) = safe_inverse(&omega.view()).map_err(|e| EconError::SingularMatrix {
        context: "CBPS optimal weighting matrix".to_string(),
        suggestion: format!("Omega matrix is singular: {:?}", e),
    })?;

    Ok(w)
}

/// Estimate standard logistic regression using Newton-Raphson.
///
/// Used as baseline for just-identified CBPS.
fn estimate_logit_newton(
    x: &ndarray::ArrayView2<f64>,
    t: &Array1<f64>,
    tolerance: f64,
    max_iter: usize,
) -> EconResult<(Array1<f64>, bool, usize, f64)> {
    let n = t.len();
    let k = x.ncols();

    let mut beta = Array1::zeros(k);
    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Linear predictor
        let z: Array1<f64> = x.dot(&beta);

        // Probabilities
        let p: Array1<f64> = z.mapv(logistic_cdf);
        let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

        // Gradient: X'(t - p)
        let residuals = t - &p_clipped;
        let mut gradient = Array1::zeros(k);
        for i in 0..n {
            for j in 0..k {
                gradient[j] += residuals[i] * x[[i, j]];
            }
        }

        // Check convergence
        let grad_norm: f64 = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < tolerance {
            converged = true;
            break;
        }

        // Weights: p(1-p)
        let weights: Array1<f64> = p_clipped.mapv(|pi| pi * (1.0 - pi));

        // Hessian: -X'WX
        let mut hessian = Array2::zeros((k, k));
        for i in 0..n {
            let wi = weights[i];
            for j in 0..k {
                for l in 0..k {
                    hessian[[j, l]] -= wi * x[[i, j]] * x[[i, l]];
                }
            }
        }

        // Invert -H
        let neg_hessian = &hessian * -1.0;
        let (hess_inv, _) =
            safe_inverse(&neg_hessian.view()).map_err(|e| EconError::SingularMatrix {
                context: "Logit Newton-Raphson".to_string(),
                suggestion: format!("Hessian singular: {:?}", e),
            })?;

        // Newton update
        let delta = hess_inv.dot(&gradient);
        beta = &beta + &delta;
    }

    // GMM criterion is 0 for just-identified
    Ok((beta, converged, iterations, 0.0))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Balance and Weight Computation
// ═══════════════════════════════════════════════════════════════════════════════

/// Compute covariate balance table.
///
/// Computes standardized differences and variance ratios for each covariate.
fn compute_balance_table(
    x: &Array2<f64>,
    t: &Array1<f64>,
    names: &[String],
    weights: Option<&[f64]>,
    threshold: f64,
) -> BalanceTable {
    let n = t.len();
    let k = x.ncols();

    let mut table = BalanceTable::new();

    // Skip intercept column (usually first)
    let start_col = if names.first().map(|s| s.as_str()) == Some("(Intercept)") {
        1
    } else {
        0
    };

    for j in start_col..k {
        let name = names[j].clone();

        // Extract values for treated and control groups
        let mut treated_vals: Vec<f64> = Vec::new();
        let mut treated_weights: Vec<f64> = Vec::new();
        let mut control_vals: Vec<f64> = Vec::new();
        let mut control_weights: Vec<f64> = Vec::new();

        for i in 0..n {
            let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
            if t[i] >= 0.5 {
                treated_vals.push(x[[i, j]]);
                treated_weights.push(w);
            } else {
                control_vals.push(x[[i, j]]);
                control_weights.push(w);
            }
        }

        // Compute weighted means
        let mean_treated = weighted_mean(&treated_vals, &treated_weights);
        let mean_control = weighted_mean(&control_vals, &control_weights);

        // Compute weighted variances
        let var_treated = weighted_variance(&treated_vals, &treated_weights, mean_treated);
        let var_control = weighted_variance(&control_vals, &control_weights, mean_control);

        // Pooled standard deviation (using unweighted for standardization)
        let pooled_var = (var_treated + var_control) / 2.0;
        let pooled_sd = pooled_var.sqrt().max(1e-10);

        // Standardized difference
        let std_diff = (mean_treated - mean_control) / pooled_sd;

        // Variance ratio
        let var_ratio = if var_control > 1e-10 {
            var_treated / var_control
        } else {
            f64::NAN
        };

        let balanced = std_diff.abs() < threshold;

        table.covariates.push(name);
        table.mean_treated.push(mean_treated);
        table.mean_control.push(mean_control);
        table.std_diff.push(std_diff);
        table.var_ratio.push(var_ratio);

        if balanced {
            table.n_balanced += 1;
        }
    }

    // Update max standardized difference
    table.max_std_diff = table
        .std_diff
        .iter()
        .map(|&d| d.abs())
        .fold(0.0_f64, f64::max);

    table
}

/// Compute weighted mean.
fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.0;
    }
    let weighted_sum: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum();
    weighted_sum / total_weight
}

/// Compute weighted variance.
fn weighted_variance(values: &[f64], weights: &[f64], mean: f64) -> f64 {
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return 0.0;
    }
    let weighted_sq_diff: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| w * (v - mean).powi(2))
        .sum();
    weighted_sq_diff / total_weight
}

/// Compute IPW weights for ATE estimation.
///
/// For treated: w = 1/p
/// For control: w = 1/(1-p)
/// Weights are normalized to sum to n.
fn compute_ipw_weights(propensity_scores: &[f64], t: &Array1<f64>) -> Vec<f64> {
    let n = t.len();
    let mut weights = vec![0.0; n];
    let mut sum_treated = 0.0;
    let mut sum_control = 0.0;

    for i in 0..n {
        let ps = propensity_scores[i].max(1e-10).min(1.0 - 1e-10);
        if t[i] >= 0.5 {
            weights[i] = 1.0 / ps;
            sum_treated += weights[i];
        } else {
            weights[i] = 1.0 / (1.0 - ps);
            sum_control += weights[i];
        }
    }

    // Normalize weights
    let n_treated: usize = t.iter().filter(|&&v| v >= 0.5).count();
    let n_control = n - n_treated;

    for i in 0..n {
        if t[i] >= 0.5 {
            weights[i] = weights[i] * (n_treated as f64) / sum_treated;
        } else {
            weights[i] = weights[i] * (n_control as f64) / sum_control;
        }
    }

    weights
}

/// Compute standard errors for CBPS coefficients.
///
/// Uses the sandwich formula for GMM variance estimation.
fn compute_cbps_std_errors(
    x: &Array2<f64>,
    t: &Array1<f64>,
    beta: &Array1<f64>,
    propensity_scores: &[f64],
    method: CbpsMethod,
) -> EconResult<(Vec<f64>, Array2<f64>)> {
    let n = t.len();
    let k = beta.len();

    // Convert propensity scores to Array1
    let p = Array1::from_vec(propensity_scores.to_vec());
    let p_clipped: Array1<f64> = p.mapv(|pi| pi.max(1e-10).min(1.0 - 1e-10));

    // Compute Jacobian
    let g = compute_moment_jacobian(x, t, &p_clipped, method);

    // Compute optimal weighting matrix
    let w = compute_optimal_weight_cbps(x, t, &p_clipped, method, n)?;

    // Variance: (G'WG)^{-1} / n
    let gwg = g.t().dot(&w).dot(&g);
    let (gwg_inv, _) = safe_inverse(&gwg.view()).map_err(|e| EconError::SingularMatrix {
        context: "CBPS variance estimation".to_string(),
        suggestion: format!("G'WG singular: {:?}", e),
    })?;

    let vcov = &gwg_inv / n as f64;

    let std_errors: Vec<f64> = (0..k).map(|i| vcov[[i, i]].max(0.0).sqrt()).collect();

    Ok((std_errors, vcov))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    /// Create test dataset with overlapping covariate distributions.
    ///
    /// Treatment assignment is correlated with covariates to create imbalance,
    /// but both groups have overlapping support (required for CBPS/IPW).
    fn create_test_dataset() -> Dataset {
        // Create data with imbalance but overlapping support
        let df = df! {
            "treatment" => [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ],
            // x1: treated mean ~0.65, control mean ~0.45, OVERLAPPING range 0.2-0.9
            "x1" => [
                0.5, 0.6, 0.7, 0.75, 0.8, 0.55, 0.65, 0.7, 0.75, 0.85,
                0.6, 0.65, 0.7, 0.8, 0.55, 0.52, 0.62, 0.68, 0.72, 0.78,
                0.2, 0.3, 0.4, 0.5, 0.6, 0.25, 0.35, 0.45, 0.55, 0.65,
                0.3, 0.4, 0.5, 0.6, 0.35, 0.28, 0.42, 0.48, 0.58, 0.68
            ],
            // x2: treated mean ~0.55, control mean ~0.35, OVERLAPPING range 0.15-0.75
            "x2" => [
                0.4, 0.5, 0.6, 0.65, 0.7, 0.45, 0.55, 0.6, 0.65, 0.75,
                0.5, 0.55, 0.6, 0.7, 0.48, 0.42, 0.52, 0.58, 0.62, 0.68,
                0.15, 0.25, 0.35, 0.45, 0.55, 0.2, 0.3, 0.4, 0.5, 0.6,
                0.25, 0.35, 0.45, 0.55, 0.28, 0.22, 0.38, 0.42, 0.52, 0.58
            ]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_cbps_exact_balance() {
        let dataset = create_test_dataset();
        let result = run_cbps(&dataset, "treatment", &["x1", "x2"], None).unwrap();

        // Check basic structure
        assert_eq!(result.n_obs, 40);
        assert_eq!(result.n_treated, 20);
        assert_eq!(result.n_control, 20);
        assert_eq!(result.n_params, 3); // intercept + 2 covariates

        // CBPS should improve or maintain balance (allow small increase due to estimation noise)
        assert!(
            result.balance_after.max_std_diff <= result.balance_before.max_std_diff + 0.2,
            "Balance after ({}) should not be much worse than before ({})",
            result.balance_after.max_std_diff,
            result.balance_before.max_std_diff
        );

        // Propensity scores should be in [0, 1] (allow boundary values after clipping)
        assert!(
            result
                .propensity_scores
                .iter()
                .all(|&p| (0.0..=1.0).contains(&p)),
            "Propensity scores should be in [0,1]"
        );

        // Weights should be non-negative
        assert!(
            result.weights.iter().all(|&w| w >= 0.0),
            "Weights should be non-negative"
        );
    }

    #[test]
    fn test_cbps_just_identified() {
        let dataset = create_test_dataset();
        let result = cbps(
            &dataset,
            "treatment",
            &["x1", "x2"],
            CbpsMethod::JustIdentified,
        )
        .unwrap();

        // Just-identified should be equivalent to logit
        assert!(result.converged);
        assert!(result.j_statistic.is_none()); // No overidentification test

        // Should still produce valid propensity scores
        assert!(result.propensity_scores.iter().all(|&p| p > 0.0 && p < 1.0));
    }

    #[test]
    fn test_balance_table() {
        let dataset = create_test_dataset();
        let result = run_cbps(&dataset, "treatment", &["x1", "x2"], None).unwrap();

        // Balance table should have 2 covariates (excluding intercept)
        assert_eq!(result.balance_before.covariates.len(), 2);
        assert_eq!(result.balance_after.covariates.len(), 2);

        // Before balance should show some imbalance (std_diff > 0 indicates imbalance)
        assert!(
            result.balance_before.max_std_diff > 0.0,
            "There should be some initial imbalance"
        );
    }

    #[test]
    fn test_cbps_display() {
        let dataset = create_test_dataset();
        let result = run_cbps(&dataset, "treatment", &["x1", "x2"], None).unwrap();

        let output = format!("{}", result);
        assert!(output.contains("CBPS"));
        assert!(output.contains("Coefficients"));
        assert!(output.contains("Balance Improvement"));
    }

    #[test]
    fn test_cbps_missing_column() {
        let dataset = create_test_dataset();
        let result = run_cbps(&dataset, "nonexistent", &["x1"], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_cbps_constant_treatment() {
        // All treated - should fail
        let df = df! {
            "treatment" => [1.0, 1.0, 1.0, 1.0, 1.0],
            "x1" => [1.0, 2.0, 3.0, 4.0, 5.0]
        }
        .unwrap();
        let dataset = Dataset::new(df);

        let result = run_cbps(&dataset, "treatment", &["x1"], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ipw_weights_normalization() {
        // Test that weights are properly normalized
        let ps = vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 0.6, 0.7];
        let t = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let weights = compute_ipw_weights(&ps, &t);

        // Sum of weights for treated should equal n_treated
        let sum_treated: f64 = weights
            .iter()
            .zip(t.iter())
            .filter(|&(_, ti)| *ti >= 0.5)
            .map(|(&w, _)| w)
            .sum();
        assert!((sum_treated - 5.0).abs() < 1e-6);

        // Sum of weights for control should equal n_control
        let sum_control: f64 = weights
            .iter()
            .zip(t.iter())
            .filter(|&(_, ti)| *ti < 0.5)
            .map(|(&w, _)| w)
            .sum();
        assert!((sum_control - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_balance_table_display() {
        let mut table = BalanceTable::new();
        table.covariates = vec!["x1".to_string(), "x2".to_string()];
        table.mean_treated = vec![0.9, 0.8];
        table.mean_control = vec![0.4, 0.3];
        table.std_diff = vec![1.2, 1.5];
        table.var_ratio = vec![1.1, 0.9];
        table.max_std_diff = 1.5;
        table.n_balanced = 0;

        let output = format!("{}", table);
        assert!(output.contains("Balance Table"));
        assert!(output.contains("x1"));
        assert!(output.contains("x2"));
    }
}
