//! Hausman specification test for panel data.
//!
//! # Mathematical Background
//!
//! Tests H₀: RE is consistent (Cov(αᵢ, Xᵢₜ) = 0) vs H₁: FE is required.
//!
//! H = (β̂ᶠᴱ - β̂ᴿᴱ)'(V̂ᶠᴱ - V̂ᴿᴱ)⁻¹(β̂ᶠᴱ - β̂ᴿᴱ) ~ χ²(k)
//!
//! # References
//!
//! - Hausman, J.A. (1978). Specification tests in econometrics. *Econometrica*,
//!   46(6), 1251-1271. https://doi.org/10.2307/1913827
//!
//! R equivalent: `plm::phtest()`

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::data::Dataset;
use crate::errors::EconResult;
use crate::traits::estimator::chi_squared_p_value;

use super::linear_models::{run_fixed_effects, run_random_effects};
use super::types::PanelResult;

/// Result from a Hausman specification test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HausmanResult {
    /// Chi-squared test statistic
    pub chi2_statistic: f64,
    /// P-value for the test
    pub p_value: f64,
    /// Degrees of freedom
    pub df: usize,
    /// Recommendation based on p-value
    pub recommendation: String,
    /// Fixed Effects estimation results
    pub fe_result: PanelResult,
    /// Random Effects estimation results
    pub re_result: PanelResult,
}

impl fmt::Display for HausmanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Hausman Specification Test")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "H0: Random Effects is consistent and efficient")?;
        writeln!(f, "H1: Random Effects is inconsistent (use Fixed Effects)")?;
        writeln!(f)?;
        writeln!(f, "Chi2 Statistic: {:.4}", self.chi2_statistic)?;
        writeln!(f, "Degrees of Freedom: {}", self.df)?;
        writeln!(f, "P-Value: {:.4}", self.p_value)?;
        writeln!(f)?;

        // Show coefficient comparison for intuition
        writeln!(f, "Coefficient Comparison:")?;
        let n_vars = self
            .fe_result
            .variables
            .len()
            .min(self.re_result.variables.len().saturating_sub(1));
        writeln!(
            f,
            "{:<20} {:>12} {:>12} {:>12}",
            "Variable", "FE Coef", "RE Coef", "Difference"
        )?;
        writeln!(f, "{:-<60}", "")?;
        for i in 0..n_vars {
            let fe_coef = self.fe_result.coefficients[i];
            let re_coef = self
                .re_result
                .coefficients
                .get(i + 1)
                .copied()
                .unwrap_or(0.0); // Skip RE intercept
            let diff = fe_coef - re_coef;
            writeln!(
                f,
                "{:<20} {:>12.4} {:>12.4} {:>12.4}",
                &self.fe_result.variables[i], fe_coef, re_coef, diff
            )?;
        }
        writeln!(f)?;

        writeln!(f, "Result: {}", self.recommendation)?;

        // Add interpretation note
        writeln!(f)?;
        writeln!(
            f,
            "Note: The Hausman test compares FE and RE coefficient estimates."
        )?;
        writeln!(
            f,
            "A significant result (p < 0.05) suggests entity effects are correlated"
        )?;
        writeln!(
            f,
            "with regressors, making RE inconsistent. A non-significant result"
        )?;
        writeln!(
            f,
            "suggests RE is more efficient, but may have low power in small samples."
        )?;
        Ok(())
    }
}

/// Run Hausman specification test comparing Fixed Effects vs Random Effects.
///
/// The Hausman test statistic is:
/// H = (β̂_FE - β̂_RE)' [Var(β̂_FE) - Var(β̂_RE)]⁻¹ (β̂_FE - β̂_RE)
///
/// Under H0 (RE is consistent and efficient), H ~ χ²(k).
///
/// # References
/// - Hausman, J.A. (1978). Specification tests in econometrics. Econometrica, 46(6), 1251-1271.
/// - Cameron, A.C. & Trivedi, P.K. (2005). Microeconometrics: Methods and Applications.
///   Cambridge University Press, Section 21.4.
pub fn run_hausman_test(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    entity_col: &str,
) -> EconResult<HausmanResult> {
    let fe_result = run_fixed_effects(dataset, y_col, x_cols, entity_col)?;
    let re_result = run_random_effects(dataset, y_col, x_cols, entity_col)?;

    // FE has k coefficients, RE has k+1 (with intercept)
    let k = fe_result.coefficients.len();

    // Extract RE coefficients excluding intercept
    let beta_fe = Array1::from_vec(fe_result.coefficients.clone());
    let beta_re_no_intercept = Array1::from_vec(re_result.coefficients[1..].to_vec());

    // Difference in coefficients
    let beta_diff = &beta_fe - &beta_re_no_intercept;

    // Construct diagonal variance-covariance matrices from standard errors
    // Under H0, Var(β̂_FE - β̂_RE) = Var(β̂_FE) - Var(β̂_RE)
    // Note: This is a simplified formula using diagonal Var matrices
    let mut var_diff_diag = Array1::zeros(k);
    for i in 0..k {
        let var_fe = fe_result.std_errors[i] * fe_result.std_errors[i];
        let var_re = if i + 1 < re_result.std_errors.len() {
            re_result.std_errors[i + 1] * re_result.std_errors[i + 1]
        } else {
            0.0
        };
        // Under H0, Var(FE) - Var(RE) should be positive semi-definite
        // Use max(0, diff) to handle numerical issues
        var_diff_diag[i] = (var_fe - var_re).max(1e-10);
    }

    // Hausman statistic: sum of (beta_diff[i]^2 / var_diff[i])
    // This is the diagonal approximation to the full quadratic form
    let chi2_statistic: f64 = beta_diff
        .iter()
        .zip(var_diff_diag.iter())
        .map(|(&b, &v)| if v > 1e-10 { b * b / v } else { 0.0 })
        .sum();

    let chi2_statistic = chi2_statistic.max(0.0);

    let p_value = chi_squared_p_value(chi2_statistic, k as f64);

    // Construct nuanced recommendation based on p-value and sample size
    let n_entities = fe_result.n_groups;
    let n_obs = fe_result.n_obs;

    let recommendation = if p_value < 0.01 {
        "Reject H0: Use Fixed Effects (strong evidence of systematic difference in coefficients)"
            .to_string()
    } else if p_value < 0.05 {
        "Reject H0: Use Fixed Effects (moderate evidence of systematic difference in coefficients)"
            .to_string()
    } else if p_value < 0.10 {
        format!(
            "Marginally fail to reject H0 (p={:.3}). Consider Fixed Effects if correlation between \
             entity effects and regressors is theoretically plausible.",
            p_value
        )
    } else if p_value > 0.90 {
        // Very high p-value suggests FE and RE are nearly identical
        format!(
            "FE and RE produce nearly identical estimates (p={:.3}). Either model is valid; \
             RE is more efficient. Note: Both may be biased if unobserved heterogeneity is time-varying.",
            p_value
        )
    } else if n_entities < 20 || n_obs < 100 {
        format!(
            "Fail to reject H0 (p={:.3}), but note: Hausman test may have low power with \
             {} entities and {} observations. Consider theoretical arguments for model choice.",
            p_value, n_entities, n_obs
        )
    } else {
        "Fail to reject H0: Random Effects is consistent and more efficient".to_string()
    };

    Ok(HausmanResult {
        chi2_statistic,
        p_value,
        df: k,
        recommendation,
        fe_result,
        re_result,
    })
}
