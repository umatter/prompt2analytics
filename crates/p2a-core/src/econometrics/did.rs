//! Difference-in-Differences (DiD) estimation.

use anyhow::{anyhow, Result};
use greeners::{CovarianceType, DiffInDiff, Formula};
use std::fmt;

use crate::data::Dataset;
use super::convert::polars_to_greeners;

/// Result from a Difference-in-Differences estimation.
#[derive(Debug, Clone)]
pub struct DiDResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Treatment group variable
    pub treatment_var: String,
    /// Post-treatment period variable
    pub post_var: String,
    /// The DiD estimate (ATT - Average Treatment Effect on Treated)
    pub att: f64,
    /// Standard error of ATT estimate
    pub std_error: f64,
    /// t-statistic
    pub t_stat: f64,
    /// p-value for ATT estimate
    pub p_value: f64,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Control group pre-treatment mean
    pub control_pre_mean: f64,
    /// Control group post-treatment mean
    pub control_post_mean: f64,
    /// Treated group pre-treatment mean
    pub treated_pre_mean: f64,
    /// Treated group post-treatment mean
    pub treated_post_mean: f64,
}

impl fmt::Display for DiDResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Difference-in-Differences Estimation")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Treatment: {}, Post: {}", self.treatment_var, self.post_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f)?;

        // Significance indicator
        let sig = if self.p_value < 0.001 {
            "***"
        } else if self.p_value < 0.01 {
            "**"
        } else if self.p_value < 0.05 {
            "*"
        } else if self.p_value < 0.1 {
            "."
        } else {
            ""
        };

        writeln!(f, "DiD ESTIMATE (Average Treatment Effect on Treated):")?;
        writeln!(f, "  ATT = {:.4} (SE: {:.4}, t = {:.2}, p = {:.3}){}",
                 self.att, self.std_error, self.t_stat, self.p_value, sig)?;
        writeln!(f)?;

        writeln!(f, "Group Means:")?;
        writeln!(f, "  Control (Pre):  {:.4}    Control (Post): {:.4}",
                 self.control_pre_mean, self.control_post_mean)?;
        writeln!(f, "  Treated (Pre):  {:.4}    Treated (Post): {:.4}",
                 self.treated_pre_mean, self.treated_post_mean)?;
        writeln!(f)?;

        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

/// Run Difference-in-Differences estimation.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `dep_var` - Dependent variable name
/// * `treatment_var` - Binary variable indicating treatment group (1 = treated, 0 = control)
/// * `post_var` - Binary variable indicating post-treatment period (1 = post, 0 = pre)
///
/// # Model
/// The model estimated is:
/// y = β₀ + β₁·treatment + β₂·post + β₃·(treatment × post) + ε
///
/// The DiD estimate (ATT) is β₃.
pub fn run_did(
    dataset: &Dataset,
    dep_var: &str,
    treatment_var: &str,
    post_var: &str,
) -> Result<DiDResult> {
    // Convert Polars DataFrame to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Build a simple formula with just the outcome variable
    let formula = format!("{} ~ 1", dep_var);

    // Parse the formula
    let parsed_formula = Formula::parse(&formula)
        .map_err(|e| anyhow!("Failed to parse DiD formula '{}': {}", formula, e))?;

    // Use robust standard errors (HC1)
    let cov_type = CovarianceType::HC1;

    // Run DiD estimation
    let result = DiffInDiff::from_formula(&parsed_formula, &gdf, treatment_var, post_var, cov_type)
        .map_err(|e| anyhow!("DiD estimation failed: {}", e))?;

    Ok(DiDResult {
        dep_var: dep_var.to_string(),
        treatment_var: treatment_var.to_string(),
        post_var: post_var.to_string(),
        att: result.att,
        std_error: result.std_error,
        t_stat: result.t_stat,
        p_value: result.p_value,
        r_squared: result.r_squared,
        n_obs: result.n_obs,
        control_pre_mean: result.control_pre_mean,
        control_post_mean: result.control_post_mean,
        treated_pre_mean: result.treated_pre_mean,
        treated_post_mean: result.treated_post_mean,
    })
}
