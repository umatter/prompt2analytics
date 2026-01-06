//! Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) estimation.

use anyhow::{anyhow, Result};
use greeners::{CovarianceType, Formula, IV};
use std::fmt;

use crate::data::Dataset;
use super::convert::polars_to_greeners;

/// Result from an IV/2SLS estimation.
#[derive(Debug, Clone)]
pub struct IVResult {
    /// Dependent variable name
    pub dep_var: String,
    /// Endogenous variable(s) description
    pub endogenous_desc: String,
    /// Instrument(s) description
    pub instruments_desc: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics (or z-statistics)
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// R-squared
    pub r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
}

impl fmt::Display for IVResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "2SLS / IV Regression Results")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Endogenous: {}", self.endogenous_desc)?;
        writeln!(f, "Instruments: {}", self.instruments_desc)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
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
                     self.t_stats[i],
                     self.p_values[i],
                     sig)?;
        }

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")?;

        Ok(())
    }
}

/// Run Instrumental Variables / Two-Stage Least Squares estimation.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `endog_formula` - Formula for endogenous model: "y ~ x1 + x2 + endog_var"
/// * `instrument_formula` - Formula for instruments: "endog_var ~ z1 + z2"
/// * `robust` - Whether to use heteroskedasticity-robust standard errors
///
/// # Example
/// ```ignore
/// // Second stage: wage ~ experience + education
/// // First stage: education ~ parents_education + distance_to_college
/// run_iv2sls(dataset, "wage ~ experience + education", "education ~ parents_edu + distance", true)
/// ```
pub fn run_iv2sls(
    dataset: &Dataset,
    endog_formula: &str,
    instrument_formula: &str,
    robust: bool,
) -> Result<IVResult> {
    // Convert Polars DataFrame to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Parse formulas
    let parsed_endog = Formula::parse(endog_formula)
        .map_err(|e| anyhow!("Failed to parse endogenous formula '{}': {}", endog_formula, e))?;
    let parsed_instr = Formula::parse(instrument_formula)
        .map_err(|e| anyhow!("Failed to parse instrument formula '{}': {}", instrument_formula, e))?;

    // Determine covariance type
    let cov_type = if robust {
        CovarianceType::HC1
    } else {
        CovarianceType::NonRobust
    };

    // Run IV/2SLS estimation
    let result = IV::from_formula(&parsed_endog, &parsed_instr, &gdf, cov_type)
        .map_err(|e| anyhow!("IV/2SLS estimation failed: {}", e))?;

    // Extract results from struct fields
    let coefficients = result.params.to_vec();
    let std_errors = result.std_errors.to_vec();
    let t_stats = result.t_values.to_vec();
    let p_values = result.p_values.to_vec();
    let variables = result.variable_names
        .unwrap_or_else(|| (0..coefficients.len()).map(|i| format!("x{}", i)).collect());

    // Extract dependent variable from formula
    let dep_var = endog_formula.split('~').next().unwrap_or("y").trim().to_string();

    // Extract endogenous and instrument descriptions
    let endogenous_desc = instrument_formula.split('~').next().unwrap_or("(endog)").trim().to_string();
    let instruments_desc = instrument_formula.split('~').nth(1).unwrap_or("(instruments)").trim().to_string();

    Ok(IVResult {
        dep_var,
        endogenous_desc,
        instruments_desc,
        variables,
        coefficients,
        std_errors,
        t_stats,
        p_values,
        r_squared: result.r_squared,
        n_obs: result.n_obs,
    })
}
