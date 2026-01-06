//! Regression diagnostics for model validation.
//!
//! Provides tests for normality, heteroskedasticity, autocorrelation,
//! multicollinearity, and influential observations.

use anyhow::{anyhow, Result};
use greeners::{CovarianceType, Diagnostics, Formula, OLS};
use std::fmt;

use crate::data::Dataset;
use crate::econometrics::polars_to_greeners;

/// Result from regression diagnostics.
#[derive(Debug, Clone)]
pub struct DiagnosticsResult {
    /// Jarque-Bera test for normality
    pub jarque_bera: Option<TestResult>,
    /// Breusch-Pagan test for heteroskedasticity
    pub breusch_pagan: Option<TestResult>,
    /// Durbin-Watson statistic for autocorrelation
    pub durbin_watson: Option<f64>,
    /// Variance Inflation Factors for multicollinearity
    pub vif: Option<Vec<VifResult>>,
    /// Condition number of the design matrix
    pub condition_number: Option<f64>,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters
    pub n_params: usize,
}

/// A statistical test result with statistic and p-value.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub statistic: f64,
    pub p_value: f64,
    pub interpretation: String,
}

/// VIF result for a single variable.
#[derive(Debug, Clone)]
pub struct VifResult {
    pub variable: String,
    pub vif: f64,
    pub interpretation: String,
}

impl fmt::Display for DiagnosticsResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Regression Diagnostics")?;
        writeln!(f, "======================")?;
        writeln!(f, "Observations: {}, Parameters: {}", self.n_obs, self.n_params)?;
        writeln!(f)?;

        // Jarque-Bera test
        if let Some(ref jb) = self.jarque_bera {
            writeln!(f, "Jarque-Bera Test (Normality of Residuals)")?;
            writeln!(f, "  H0: Residuals are normally distributed")?;
            writeln!(f, "  Statistic: {:.4}", jb.statistic)?;
            writeln!(f, "  P-Value: {:.4}", jb.p_value)?;
            writeln!(f, "  {}", jb.interpretation)?;
            writeln!(f)?;
        }

        // Breusch-Pagan test
        if let Some(ref bp) = self.breusch_pagan {
            writeln!(f, "Breusch-Pagan Test (Heteroskedasticity)")?;
            writeln!(f, "  H0: Homoskedasticity (constant variance)")?;
            writeln!(f, "  LM Statistic: {:.4}", bp.statistic)?;
            writeln!(f, "  P-Value: {:.4}", bp.p_value)?;
            writeln!(f, "  {}", bp.interpretation)?;
            writeln!(f)?;
        }

        // Durbin-Watson
        if let Some(dw) = self.durbin_watson {
            writeln!(f, "Durbin-Watson Test (Autocorrelation)")?;
            writeln!(f, "  Statistic: {:.4} (range: 0-4, 2 = no autocorrelation)", dw)?;
            let interpretation = if dw < 1.5 {
                "Positive autocorrelation detected"
            } else if dw > 2.5 {
                "Negative autocorrelation detected"
            } else {
                "No significant autocorrelation"
            };
            writeln!(f, "  {}", interpretation)?;
            writeln!(f)?;
        }

        // Condition Number
        if let Some(cn) = self.condition_number {
            writeln!(f, "Condition Number (Multicollinearity)")?;
            writeln!(f, "  Value: {:.2}", cn)?;
            let interpretation = if cn < 10.0 {
                "No multicollinearity concern"
            } else if cn < 30.0 {
                "Moderate multicollinearity"
            } else if cn < 100.0 {
                "Strong multicollinearity - caution advised"
            } else {
                "Severe multicollinearity - results may be unreliable"
            };
            writeln!(f, "  {}", interpretation)?;
            writeln!(f)?;
        }

        // VIF
        if let Some(ref vif_results) = self.vif {
            writeln!(f, "Variance Inflation Factors (VIF)")?;
            writeln!(f, "  (VIF > 10 indicates problematic multicollinearity)")?;
            writeln!(f, "  {:<20} {:>10} {}", "Variable", "VIF", "Status")?;
            writeln!(f, "  {}", "-".repeat(50))?;
            for v in vif_results {
                if v.vif.is_nan() {
                    writeln!(f, "  {:<20} {:>10} {}", v.variable, "N/A", "(constant)")?;
                } else if v.vif.is_infinite() {
                    writeln!(f, "  {:<20} {:>10} {}", v.variable, "∞", "Perfect collinearity!")?;
                } else {
                    writeln!(f, "  {:<20} {:>10.2} {}", v.variable, v.vif, v.interpretation)?;
                }
            }
        }

        Ok(())
    }
}

/// Run regression diagnostics on a fitted model.
///
/// # Arguments
/// * `dataset` - The dataset containing the data
/// * `formula` - R-style formula (e.g., "y ~ x1 + x2")
pub fn run_diagnostics(dataset: &Dataset, formula: &str) -> Result<DiagnosticsResult> {
    // Parse the formula
    let parsed_formula = Formula::parse(formula)
        .map_err(|e| anyhow!("Failed to parse formula '{}': {}", formula, e))?;

    // Convert to greeners DataFrame
    let gdf = polars_to_greeners(dataset.df())?;

    // Get design matrix (y, X) from formula
    let (y, x) = gdf.to_design_matrix(&parsed_formula)
        .map_err(|e| anyhow!("Failed to build design matrix: {}", e))?;

    // Fit OLS model
    let ols_result = OLS::from_formula(&parsed_formula, &gdf, CovarianceType::NonRobust)
        .map_err(|e| anyhow!("OLS fitting failed: {}", e))?;

    let n_obs = ols_result.n_obs;
    let n_params = ols_result.params.len();

    // Compute residuals: y - X * beta
    let residuals = ols_result.residuals(&y, &x);

    // Build variable names
    let var_names = ols_result.variable_names.clone().unwrap_or_else(|| {
        let mut names = vec![];
        if parsed_formula.intercept {
            names.push("const".to_string());
        }
        names.extend(parsed_formula.independents.iter().cloned());
        names
    });

    // Jarque-Bera test for normality
    let jarque_bera = match Diagnostics::jarque_bera(&residuals) {
        Ok((stat, p)) => Some(TestResult {
            name: "Jarque-Bera".to_string(),
            statistic: stat,
            p_value: p,
            interpretation: if p < 0.05 {
                "Reject H0: Residuals are NOT normally distributed".to_string()
            } else {
                "Fail to reject H0: Residuals appear normally distributed".to_string()
            },
        }),
        Err(_) => None,
    };

    // Breusch-Pagan test for heteroskedasticity
    let breusch_pagan = match Diagnostics::breusch_pagan(&residuals, &x) {
        Ok((stat, p)) => Some(TestResult {
            name: "Breusch-Pagan".to_string(),
            statistic: stat,
            p_value: p,
            interpretation: if p < 0.05 {
                "Reject H0: Heteroskedasticity detected. Consider robust SEs.".to_string()
            } else {
                "Fail to reject H0: No significant heteroskedasticity".to_string()
            },
        }),
        Err(_) => None,
    };

    // Durbin-Watson test for autocorrelation
    let durbin_watson = Some(Diagnostics::durbin_watson(&residuals));

    // Condition number
    let condition_number = Diagnostics::condition_number(&x).ok();

    // VIF for each variable
    let vif = match Diagnostics::vif(&x) {
        Ok(vif_arr) => {
            let vif_results: Vec<VifResult> = var_names
                .iter()
                .zip(vif_arr.iter())
                .map(|(name, &vif_val)| VifResult {
                    variable: name.clone(),
                    vif: vif_val,
                    interpretation: if vif_val.is_nan() {
                        String::new()
                    } else if vif_val < 5.0 {
                        "OK".to_string()
                    } else if vif_val < 10.0 {
                        "Moderate".to_string()
                    } else {
                        "HIGH".to_string()
                    },
                })
                .collect();

            Some(vif_results)
        }
        Err(_) => None,
    };

    Ok(DiagnosticsResult {
        jarque_bera,
        breusch_pagan,
        durbin_watson,
        vif,
        condition_number,
        n_obs,
        n_params,
    })
}
