//! Shared types for discrete choice models.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::traits::estimator::SignificanceLevel;

/// Discrete choice model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscreteModelType {
    Logit,
    Probit,
}

impl fmt::Display for DiscreteModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscreteModelType::Logit => write!(f, "Logit"),
            DiscreteModelType::Probit => write!(f, "Probit"),
        }
    }
}

/// Result from a discrete choice model (Logit/Probit).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteResult {
    /// Model type
    pub model_type: DiscreteModelType,
    /// Dependent variable name
    pub dep_var: String,
    /// Variable names
    pub variables: Vec<String>,
    /// Estimated coefficients
    pub coefficients: Vec<f64>,
    /// Standard errors
    pub std_errors: Vec<f64>,
    /// Z-statistics
    pub z_stats: Vec<f64>,
    /// P-values
    pub p_values: Vec<f64>,
    /// Significance levels for each coefficient
    pub significance: Vec<SignificanceLevel>,
    /// Marginal effects at the mean
    pub marginal_effects: Vec<f64>,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// Null log-likelihood
    pub log_likelihood_null: f64,
    /// McFadden's pseudo R-squared
    pub pseudo_r_squared: f64,
    /// AIC
    pub aic: f64,
    /// BIC
    pub bic: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Number of positive outcomes (y=1)
    pub n_positive: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Any warnings (e.g., separation issues)
    pub warnings: Vec<String>,
}

impl fmt::Display for DiscreteResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{} Regression Results", self.model_type)?;
        writeln!(f, "===========================================")?;
        writeln!(
            f,
            "Dep. Variable: {:<20}  No. Observations: {}",
            self.dep_var, self.n_obs
        )?;
        writeln!(
            f,
            "Model: {:<23}  Log-Likelihood: {:.4}",
            self.model_type, self.log_likelihood
        )?;
        writeln!(
            f,
            "Pseudo R²: {:<19.4}  LL-Null: {:.4}",
            self.pseudo_r_squared, self.log_likelihood_null
        )?;
        writeln!(f, "AIC: {:<25.4}  BIC: {:.4}", self.aic, self.bic)?;
        writeln!(
            f,
            "Converged: {:<17}  Iterations: {}",
            self.converged, self.iterations
        )?;
        writeln!(f)?;

        // Warnings
        if !self.warnings.is_empty() {
            writeln!(f, "Warnings:")?;
            for warning in &self.warnings {
                writeln!(f, "  {}", warning)?;
            }
            writeln!(f)?;
        }

        // Coefficient table
        writeln!(f, "{:-<75}", "")?;
        writeln!(
            f,
            "{:<20} {:>12} {:>10} {:>10} {:>10} {:>6}",
            "Variable", "Coef", "Std.Err", "z", "P>|z|", "Sig"
        )?;
        writeln!(f, "{:-<75}", "")?;

        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.6} {:>10.6} {:>10.4} {:>10.4} {:>6}",
                &self.variables[i],
                self.coefficients[i],
                self.std_errors[i],
                self.z_stats[i],
                self.p_values[i],
                self.significance[i]
            )?;
        }

        writeln!(f, "{:-<75}", "")?;

        // Marginal effects
        writeln!(f)?;
        writeln!(f, "Marginal Effects at the Mean:")?;
        writeln!(f, "{:-<45}", "")?;
        writeln!(f, "{:<20} {:>12}", "Variable", "dy/dx")?;
        writeln!(f, "{:-<45}", "")?;
        for i in 0..self.variables.len() {
            writeln!(
                f,
                "{:<20} {:>12.6}",
                &self.variables[i], self.marginal_effects[i]
            )?;
        }
        writeln!(f, "{:-<45}", "")?;

        Ok(())
    }
}

/// Settings for Maximum Likelihood Estimation.
#[derive(Debug, Clone)]
pub struct MleSettings {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Step size for Newton-Raphson
    pub step_size: f64,
    /// Use backtracking line search
    pub use_line_search: bool,
    /// Armijo condition parameter (sufficient decrease)
    pub armijo_c: f64,
    /// Step reduction factor for line search
    pub step_reduction: f64,
    /// Maximum line search iterations
    pub max_line_search: usize,
}

impl Default for MleSettings {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-8,
            step_size: 1.0,
            use_line_search: true,
            armijo_c: 1e-4,
            step_reduction: 0.5,
            max_line_search: 20,
        }
    }
}

// Note: DiscreteResult doesn't implement LinearEstimator because:
// 1. Discrete models use Vec<f64> instead of Array1<f64> for coefficients
// 2. Discrete models don't have traditional residuals
// 3. Z-statistics are used instead of t-statistics
// Instead, DiscreteResult provides its own methods via Display and the export traits.
