//! Instrumental Variables (IV) and Two-Stage Least Squares (2SLS) estimation.

use anyhow::{anyhow, Result};
use greeners::{CovarianceType, Formula, IV, OLS};
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

/// Result from first-stage diagnostics for IV/2SLS.
#[derive(Debug, Clone)]
pub struct FirstStageDiagnostics {
    /// Endogenous variable name
    pub endogenous_var: String,
    /// Instruments used
    pub instruments: Vec<String>,
    /// First-stage F-statistic (instrument strength test)
    pub f_statistic: f64,
    /// p-value for F-statistic
    pub f_pvalue: f64,
    /// First-stage R-squared
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Whether instruments pass weak instrument test (F > 10)
    pub strong_instruments: bool,
    /// Coefficients on instruments in first stage
    pub instrument_coeffs: Vec<(String, f64, f64, f64)>, // (name, coef, se, t-stat)
}

impl fmt::Display for FirstStageDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "First-Stage Diagnostics for IV/2SLS")?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Endogenous Variable: {}", self.endogenous_var)?;
        writeln!(f, "Instruments: {}", self.instruments.join(", "))?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f)?;

        writeln!(f, "Instrument Strength Test:")?;
        writeln!(f, "  F-statistic: {:.4}", self.f_statistic)?;
        writeln!(f, "  Prob (F): {:.4}", self.f_pvalue)?;
        writeln!(f, "  R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "  Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(f)?;

        // Stock-Yogo critical values interpretation
        let strength = if self.f_statistic > 10.0 {
            "STRONG (F > 10)"
        } else if self.f_statistic > 5.0 {
            "MODERATE (5 < F < 10)"
        } else {
            "WEAK (F < 5) - Caution!"
        };
        writeln!(f, "  Instrument Strength: {}", strength)?;
        writeln!(f)?;

        writeln!(f, "First-Stage Coefficients:")?;
        writeln!(f, "{:<20} {:>12} {:>12} {:>10}", "Instrument", "Coef", "Std Err", "t-stat")?;
        writeln!(f, "{}", "-".repeat(60))?;

        for (name, coef, se, t) in &self.instrument_coeffs {
            let sig = if t.abs() > 2.576 {
                "***"
            } else if t.abs() > 1.96 {
                "**"
            } else if t.abs() > 1.645 {
                "*"
            } else {
                ""
            };
            writeln!(f, "{:<20} {:>12.4} {:>12.4} {:>10.2}{}", name, coef, se, t, sig)?;
        }

        writeln!(f, "{}", "-".repeat(60))?;
        writeln!(f, "Note: Stock-Yogo (2005) critical value for 10% max bias: F > 16.38 (single instrument)")?;
        writeln!(f, "      Rule of thumb: F > 10 suggests instruments are not weak")?;

        Ok(())
    }
}

/// Run first-stage diagnostics for IV/2SLS.
///
/// Tests instrument strength by regressing the endogenous variable on instruments.
/// Key output is the F-statistic: F > 10 suggests instruments are not weak.
///
/// # Arguments
/// * `dataset` - The dataset
/// * `endog_var` - Name of the endogenous variable
/// * `instruments` - Names of the instrumental variables
/// * `controls` - Optional control variables to include in first stage
pub fn run_first_stage_diagnostics(
    dataset: &Dataset,
    endog_var: &str,
    instruments: &[&str],
    controls: Option<&[&str]>,
) -> Result<FirstStageDiagnostics> {
    // Convert to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Build first-stage formula: endogenous ~ instruments + controls
    let mut rhs_vars: Vec<String> = instruments.iter().map(|s| s.to_string()).collect();
    if let Some(ctrl) = controls {
        for c in ctrl {
            rhs_vars.push(c.to_string());
        }
    }

    let formula_str = format!("{} ~ {}", endog_var, rhs_vars.join(" + "));
    let formula = Formula::parse(&formula_str)
        .map_err(|e| anyhow!("Failed to parse formula '{}': {}", formula_str, e))?;

    // Run OLS for first stage
    let result = OLS::from_formula(&formula, &gdf, CovarianceType::HC1)
        .map_err(|e| anyhow!("First-stage regression failed: {}", e))?;

    // Extract instrument coefficients (skip intercept)
    let var_names = result.variable_names.clone()
        .unwrap_or_else(|| (0..result.params.len()).map(|i| format!("x{}", i)).collect());

    let mut instrument_coeffs = Vec::new();
    for (i, instr) in instruments.iter().enumerate() {
        // Find the instrument in variable names (skip const at index 0)
        let idx = var_names.iter().position(|v| v == *instr);
        if let Some(idx) = idx {
            instrument_coeffs.push((
                instr.to_string(),
                result.params[idx],
                result.std_errors[idx],
                result.t_values[idx],
            ));
        } else if i + 1 < result.params.len() {
            // Fallback: use position
            instrument_coeffs.push((
                instr.to_string(),
                result.params[i + 1],
                result.std_errors[i + 1],
                result.t_values[i + 1],
            ));
        }
    }

    Ok(FirstStageDiagnostics {
        endogenous_var: endog_var.to_string(),
        instruments: instruments.iter().map(|s| s.to_string()).collect(),
        f_statistic: result.f_statistic,
        f_pvalue: result.prob_f,
        r_squared: result.r_squared,
        adj_r_squared: result.adj_r_squared,
        n_obs: result.n_obs,
        strong_instruments: result.f_statistic > 10.0,
        instrument_coeffs,
    })
}
