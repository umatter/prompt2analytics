//! Core types for panel data estimation.
//!
//! Contains `PanelResult` and `PanelMethod` used across panel estimators.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::traits::estimator::{LinearEstimator, SignificanceLevel};

/// Result from a panel data estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelResult {
    /// Estimation method used
    pub method: PanelMethod,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names (including intercept if present)
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// t-statistics
    pub t_stats: Vec<f64>,
    /// p-values
    pub p_values: Vec<f64>,
    /// R-squared (within for FE, overall for RE)
    pub r_squared: f64,
    /// Adjusted R-squared
    pub adj_r_squared: f64,
    /// F-statistic
    pub f_stat: f64,
    /// F-statistic p-value
    pub f_p_value: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of groups (entities)
    pub n_groups: usize,
    /// Degrees of freedom
    pub df: usize,
    /// Entity variable name
    pub entity_var: String,
    /// Significance levels
    pub significance: Vec<SignificanceLevel>,
    /// Variance components (for RE)
    pub sigma_u: Option<f64>,
    /// Idiosyncratic variance
    pub sigma_e: Option<f64>,
    /// Theta (quasi-demeaning factor for RE)
    pub theta: Option<f64>,

    // ── Trait-backing storage (LinearEstimator). Kept out of the public
    //    JSON surface to avoid duplicating the Vec-typed fields above; round-
    //    trip deserialization uses `default` to fall back to empty arrays.
    /// Coefficients as `Array1`, used by `LinearEstimator::coefficients`.
    #[serde(skip, default)]
    pub coef_arr: Array1<f64>,
    /// Standard errors as `Array1`, used by `LinearEstimator::std_errors`.
    #[serde(skip, default)]
    pub se_arr: Array1<f64>,
    /// Residuals (`Array1`), used by `LinearEstimator::residuals`.
    #[serde(skip, default)]
    pub residuals: Array1<f64>,
    /// Coefficient variance-covariance matrix (entity-clustered for FE,
    /// quasi-GLS for RE), used by `LinearEstimator::vcov_matrix`.
    #[serde(skip, default)]
    pub vcov: Array2<f64>,
}

impl LinearEstimator for PanelResult {
    fn coefficients(&self) -> &Array1<f64> {
        &self.coef_arr
    }
    fn std_errors(&self) -> &Array1<f64> {
        &self.se_arr
    }
    fn residuals(&self) -> &Array1<f64> {
        &self.residuals
    }
    fn vcov_matrix(&self) -> &Array2<f64> {
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

/// Panel estimation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelMethod {
    /// Fixed Effects (within) estimator
    FixedEffects,
    /// Random Effects (GLS) estimator
    RandomEffects,
}

impl fmt::Display for PanelMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PanelMethod::FixedEffects => write!(f, "Fixed Effects"),
            PanelMethod::RandomEffects => write!(f, "Random Effects"),
        }
    }
}

impl fmt::Display for PanelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Panel Regression Results", self.method)?;
        writeln!(f, "===========================================")?;
        writeln!(f, "Dep. Variable: {}", self.dep_var)?;
        writeln!(f, "Entity: {}", self.entity_var)?;
        writeln!(f, "No. Observations: {}", self.n_obs)?;
        writeln!(f, "No. Groups: {}", self.n_groups)?;
        writeln!(f, "R-squared: {:.4}", self.r_squared)?;
        writeln!(f, "Adj. R-squared: {:.4}", self.adj_r_squared)?;
        writeln!(
            f,
            "F-statistic: {:.4} (p-value: {:.4})",
            self.f_stat, self.f_p_value
        )?;

        if let Some(sigma_u) = self.sigma_u {
            writeln!(f, "sigma_u: {:.4}", sigma_u)?;
        }
        if let Some(sigma_e) = self.sigma_e {
            writeln!(f, "sigma_e: {:.4}", sigma_e)?;
        }
        if let Some(theta) = self.theta {
            writeln!(f, "theta: {:.4}", theta)?;
        }

        writeln!(f)?;
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

        writeln!(f, "{}", "-".repeat(70))?;
        writeln!(f, "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '†' 0.1")?;

        Ok(())
    }
}
