//! Discrete choice models: Logit and Probit.
//!
//! Binary outcome models for classification and probability estimation.

use anyhow::{anyhow, Result};
use greeners::{Formula, Logit, Probit};
use std::fmt;

use crate::data::Dataset;
use super::convert::polars_to_greeners;

/// Result from a discrete choice model (Logit/Probit).
#[derive(Debug, Clone)]
pub struct DiscreteResult {
    /// Model type ("Logit" or "Probit")
    pub model_type: String,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// z-statistics
    pub z_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// McFadden's Pseudo R-squared
    pub pseudo_r_squared: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Number of observations
    pub n_obs: usize,
}

impl fmt::Display for DiscreteResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Regression Results (MLE)", self.model_type)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "Log-Likelihood: {:.4}", self.log_likelihood)?;
        writeln!(f, "Pseudo R-squared: {:.4}", self.pseudo_r_squared)?;
        writeln!(f, "Iterations: {}", self.iterations)?;
        writeln!(f)?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10} {:>10}",
                 "Variable", "Coef", "Std Err", "z", "P>|z|")?;
        writeln!(f, "{}", "-".repeat(70))?;

        for i in 0..self.variables.len() {
            let sig = if self.p_values[i] < 0.001 {
                "***"
            } else if self.p_values[i] < 0.01 {
                "**"
            } else if self.p_values[i] < 0.05 {
                "*"
            } else if self.p_values[i] < 0.1 {
                "."
            } else {
                ""
            };

            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2} {:>10.3}{}",
                     self.variables[i],
                     self.coefficients[i],
                     self.std_errors[i],
                     self.z_stats[i],
                     self.p_values[i],
                     sig)?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

/// Run Logit (logistic) regression.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `formula` - R-style formula (e.g., "y ~ x1 + x2")
///
/// # Note
/// The dependent variable should be binary (0/1).
pub fn run_logit(dataset: &Dataset, formula: &str) -> Result<DiscreteResult> {
    // Parse the formula
    let parsed_formula = Formula::parse(formula)
        .map_err(|e| anyhow!("Failed to parse formula '{}': {}", formula, e))?;

    // Convert to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Fit Logit model
    let result = Logit::from_formula(&parsed_formula, &gdf)
        .map_err(|e| anyhow!("Logit estimation failed: {}", e))?;

    // Extract dependent variable from formula
    let dep_var = formula.split('~').next().unwrap_or("y").trim().to_string();

    // Build variable names
    let variables = result.variable_names.clone().unwrap_or_else(|| {
        (0..result.params.len()).map(|i| format!("x{}", i)).collect()
    });

    Ok(DiscreteResult {
        model_type: "Logit".to_string(),
        dep_var,
        variables,
        coefficients: result.params.to_vec(),
        std_errors: result.std_errors.to_vec(),
        z_stats: result.z_values.to_vec(),
        p_values: result.p_values.to_vec(),
        log_likelihood: result.log_likelihood,
        pseudo_r_squared: result.pseudo_r2,
        iterations: result.iterations,
        n_obs: result.params.len(), // Approximate - greeners doesn't expose n_obs directly
    })
}

/// Run Probit regression.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `formula` - R-style formula (e.g., "y ~ x1 + x2")
///
/// # Note
/// The dependent variable should be binary (0/1).
pub fn run_probit(dataset: &Dataset, formula: &str) -> Result<DiscreteResult> {
    // Parse the formula
    let parsed_formula = Formula::parse(formula)
        .map_err(|e| anyhow!("Failed to parse formula '{}': {}", formula, e))?;

    // Convert to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Fit Probit model
    let result = Probit::from_formula(&parsed_formula, &gdf)
        .map_err(|e| anyhow!("Probit estimation failed: {}", e))?;

    // Extract dependent variable from formula
    let dep_var = formula.split('~').next().unwrap_or("y").trim().to_string();

    // Build variable names
    let variables = result.variable_names.clone().unwrap_or_else(|| {
        (0..result.params.len()).map(|i| format!("x{}", i)).collect()
    });

    Ok(DiscreteResult {
        model_type: "Probit".to_string(),
        dep_var,
        variables,
        coefficients: result.params.to_vec(),
        std_errors: result.std_errors.to_vec(),
        z_stats: result.z_values.to_vec(),
        p_values: result.p_values.to_vec(),
        log_likelihood: result.log_likelihood,
        pseudo_r_squared: result.pseudo_r2,
        iterations: result.iterations,
        n_obs: result.params.len(), // Approximate
    })
}
