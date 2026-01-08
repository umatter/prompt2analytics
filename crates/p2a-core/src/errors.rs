//! Unified error types for econometric operations.
//!
//! This module provides actionable error messages that help users
//! understand and resolve issues in their analyses.

use serde::Serialize;
use thiserror::Error;

/// Main error type for econometric operations.
#[derive(Debug, Error)]
pub enum EconError {
    // ═══════════════════════════════════════════════════════════════════
    // Matrix/Numerical Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Matrix is singular ({context}). Suggestion: {suggestion}")]
    SingularMatrix { context: String, suggestion: String },

    #[error("Matrix is ill-conditioned (condition number: {condition:.2e}). Results may be numerically unstable. Suggestion: {suggestion}")]
    IllConditioned {
        condition: f64,
        suggestion: String,
    },

    #[error("Perfect multicollinearity detected between columns: {columns:?}. Suggestion: Remove one of the collinear variables.")]
    PerfectMulticollinearity { columns: Vec<String> },

    // ═══════════════════════════════════════════════════════════════════
    // Data Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Insufficient data: need at least {required} observations, got {provided}. Context: {context}")]
    InsufficientData {
        required: usize,
        provided: usize,
        context: String,
    },

    #[error("Column '{column}' not found in dataset. Available columns: {available:?}")]
    ColumnNotFound { column: String, available: Vec<String> },

    #[error("Column '{column}' contains non-numeric values. Ensure all values are numeric.")]
    NonNumericColumn { column: String },

    #[error("Column '{column}' contains {count} null values. Consider imputation or filtering.")]
    NullValues { column: String, count: usize },

    #[error("Empty dataset: no observations to process")]
    EmptyDataset,

    // ═══════════════════════════════════════════════════════════════════
    // Estimation Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Convergence failed after {iterations} iterations (last change: {last_change:.2e}). Suggestion: {suggestion}")]
    ConvergenceFailure {
        iterations: usize,
        last_change: f64,
        suggestion: String,
    },

    #[error("Optimization diverged at iteration {iteration}. Suggestion: {suggestion}")]
    DivergenceError { iteration: usize, suggestion: String },

    #[error("Invalid model specification: {message}")]
    InvalidSpecification { message: String },

    // ═══════════════════════════════════════════════════════════════════
    // Panel Data Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Only {n_entities} entity groups found. Fixed effects estimation requires more variation. Suggestion: Use pooled OLS or check your entity identifier.")]
    InsufficientEntities { n_entities: usize },

    #[error("Unbalanced panel data detected. Some entities have only {min_obs} observations while others have {max_obs}.")]
    UnbalancedPanel { min_obs: usize, max_obs: usize },

    // ═══════════════════════════════════════════════════════════════════
    // Clustering Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Only {n_clusters} clusters found. Clustered standard errors are unreliable with fewer than 10 clusters. Consider using robust (HC) standard errors instead.")]
    FewClusters { n_clusters: usize },

    // ═══════════════════════════════════════════════════════════════════
    // IV/2SLS Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Under-identification: {n_endogenous} endogenous variable(s) but only {n_instruments} excluded instrument(s). Need at least as many instruments as endogenous variables.")]
    UnderIdentified {
        n_endogenous: usize,
        n_instruments: usize,
    },

    #[error("Weak instruments detected (first-stage F = {f_stat:.2}). Rule of thumb: F should be > 10. Consider finding stronger instruments.")]
    WeakInstruments { f_stat: f64 },

    // ═══════════════════════════════════════════════════════════════════
    // Time Series Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Insufficient time periods for VAR({lags}): need at least {required} observations, got {provided}")]
    InsufficientTimePeriods {
        lags: usize,
        required: usize,
        provided: usize,
    },

    #[error("Non-stationary series detected. Consider differencing the data or using VECM for cointegrated series.")]
    NonStationarySeries,

    // ═══════════════════════════════════════════════════════════════════
    // Generic Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Linear algebra error: {0}")]
    LinalgError(#[from] crate::linalg::LinalgError),

    #[error("Design matrix error: {0}")]
    DesignError(#[from] crate::linalg::DesignError),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Warnings that don't prevent estimation but should be noted.
#[derive(Debug, Clone, Serialize)]
pub enum EstimationWarning {
    /// Condition number is elevated but not critical
    HighConditionNumber { value: f64, threshold: f64 },

    /// Variance inflation factors indicate potential multicollinearity
    HighVIF { variable: String, vif: f64 },

    /// Few clusters for cluster-robust standard errors
    FewClusters { n_clusters: usize, recommended: usize },

    /// Potential heteroskedasticity detected
    Heteroskedasticity { test_name: String, p_value: f64 },

    /// Potential autocorrelation in residuals
    Autocorrelation { test_name: String, statistic: f64 },

    /// Non-normal residuals
    NonNormalResiduals { test_name: String, p_value: f64 },

    /// R-squared is negative (possible with IV estimation)
    NegativeRSquared { value: f64 },

    /// Slow convergence (converged but took many iterations)
    SlowConvergence { iterations: usize, tolerance: f64 },

    /// Some observations were dropped due to missing values
    DroppedObservations { count: usize, reason: String },
}

impl EstimationWarning {
    /// Convert warning to a human-readable message.
    pub fn message(&self) -> String {
        match self {
            Self::HighConditionNumber { value, threshold } => {
                format!(
                    "High condition number ({:.2e} > {:.2e}). Results may be sensitive to small data changes.",
                    value, threshold
                )
            }
            Self::HighVIF { variable, vif } => {
                format!(
                    "High VIF ({:.2}) on '{}' suggests multicollinearity. Consider removing correlated predictors.",
                    vif, variable
                )
            }
            Self::FewClusters { n_clusters, recommended } => {
                format!(
                    "Only {} clusters (recommended: {}+). Clustered standard errors may be biased downward.",
                    n_clusters, recommended
                )
            }
            Self::Heteroskedasticity { test_name, p_value } => {
                format!(
                    "{} test p-value = {:.4}. Consider using robust standard errors (HC0-HC3).",
                    test_name, p_value
                )
            }
            Self::Autocorrelation { test_name, statistic } => {
                format!(
                    "{} statistic = {:.4}. Consider using Newey-West standard errors or checking model specification.",
                    test_name, statistic
                )
            }
            Self::NonNormalResiduals { test_name, p_value } => {
                format!(
                    "{} test p-value = {:.4}. Residuals may not be normally distributed.",
                    test_name, p_value
                )
            }
            Self::NegativeRSquared { value } => {
                format!(
                    "R² = {:.4} is negative. This is possible with IV estimation when the model fits poorly.",
                    value
                )
            }
            Self::SlowConvergence { iterations, tolerance } => {
                format!(
                    "Convergence was slow ({} iterations to reach tolerance {:.2e}). Consider checking model specification.",
                    iterations, tolerance
                )
            }
            Self::DroppedObservations { count, reason } => {
                format!("{} observations dropped: {}", count, reason)
            }
        }
    }
}

impl std::fmt::Display for EstimationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

/// Result type alias for econometric operations.
pub type EconResult<T> = Result<T, EconError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = EconError::InsufficientData {
            required: 10,
            provided: 5,
            context: "OLS regression".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
        assert!(msg.contains("OLS"));
    }

    #[test]
    fn test_warning_message() {
        let warning = EstimationWarning::HighVIF {
            variable: "education".to_string(),
            vif: 15.3,
        };
        let msg = warning.message();
        assert!(msg.contains("education"));
        assert!(msg.contains("15.3"));
    }
}
