//! Unified error types for econometric operations.
//!
//! This module provides actionable error messages that help users
//! understand and resolve issues in their analyses.

use serde::Serialize;
use thiserror::Error;

// ═══════════════════════════════════════════════════════════════════════════
// Typo Suggestion Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Compute Levenshtein edit distance between two strings.
///
/// Returns the minimum number of single-character edits (insertions,
/// deletions, or substitutions) required to transform `a` into `b`.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a = a.to_lowercase();
    let b = b.to_lowercase();
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use two rows instead of full matrix for space efficiency
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Find column names similar to the target, sorted by similarity.
///
/// Returns columns within `max_distance` edits of the target, up to `max_suggestions`.
/// Useful for providing "Did you mean...?" suggestions on typos.
///
/// # Arguments
/// * `target` - The column name that wasn't found
/// * `available` - List of available column names
/// * `max_distance` - Maximum edit distance to consider (default: 3)
/// * `max_suggestions` - Maximum number of suggestions to return (default: 3)
///
/// # Example
/// ```
/// use p2a_core::errors::suggest_similar_columns;
/// let available = vec!["sqft".to_string(), "price".to_string(), "bedrooms".to_string()];
/// let suggestions = suggest_similar_columns("sqfeet", &available, 3, 3);
/// assert_eq!(suggestions, vec!["sqft"]);
/// ```
pub fn suggest_similar_columns(
    target: &str,
    available: &[String],
    max_distance: usize,
    max_suggestions: usize,
) -> Vec<String> {
    let mut scored: Vec<(usize, &String)> = available
        .iter()
        .map(|col| (levenshtein_distance(target, col), col))
        .filter(|(dist, _)| *dist > 0 && *dist <= max_distance)
        .collect();

    // Sort by distance (closest first)
    scored.sort_by_key(|(dist, _)| *dist);

    scored
        .into_iter()
        .take(max_suggestions)
        .map(|(_, col)| col.clone())
        .collect()
}

/// Format a "Did you mean...?" suggestion string.
///
/// Returns `None` if no similar columns found.
pub fn format_suggestions(target: &str, available: &[String]) -> Option<String> {
    let suggestions = suggest_similar_columns(target, available, 3, 3);
    if suggestions.is_empty() {
        None
    } else if suggestions.len() == 1 {
        Some(format!("Did you mean '{}'?", suggestions[0]))
    } else {
        Some(format!("Did you mean one of: {:?}?", suggestions))
    }
}

/// Format the ColumnNotFound error message with typo suggestions.
fn format_column_not_found_error(column: &str, available: &[String]) -> String {
    let base = format!("Column '{}' not found in dataset.", column);

    if let Some(suggestion) = format_suggestions(column, available) {
        format!("{} {} Available columns: {:?}", base, suggestion, available)
    } else {
        format!("{} Available columns: {:?}", base, available)
    }
}

/// Main error type for econometric operations.
#[derive(Debug, Error)]
pub enum EconError {
    // ═══════════════════════════════════════════════════════════════════
    // Matrix/Numerical Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error("Matrix is singular ({context}). Suggestion: {suggestion}")]
    SingularMatrix { context: String, suggestion: String },

    #[error(
        "Matrix is ill-conditioned (condition number: {condition:.2e}). Results may be numerically unstable. Suggestion: {suggestion}"
    )]
    IllConditioned { condition: f64, suggestion: String },

    #[error(
        "Perfect multicollinearity detected between columns: {columns:?}. Suggestion: Remove one of the collinear variables."
    )]
    PerfectMulticollinearity { columns: Vec<String> },

    // ═══════════════════════════════════════════════════════════════════
    // Data Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error(
        "Insufficient data: need at least {required} observations, got {provided}. Context: {context}"
    )]
    InsufficientData {
        required: usize,
        provided: usize,
        context: String,
    },

    #[error("{}", format_column_not_found_error(column, available))]
    ColumnNotFound {
        column: String,
        available: Vec<String>,
    },

    #[error("Column '{column}' contains non-numeric values. Ensure all values are numeric.")]
    NonNumericColumn { column: String },

    #[error("Column '{column}' contains {count} null values. Consider imputation or filtering.")]
    NullValues { column: String, count: usize },

    #[error("Empty dataset: no observations to process")]
    EmptyDataset,

    // ═══════════════════════════════════════════════════════════════════
    // Estimation Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error(
        "Convergence failed after {iterations} iterations (last change: {last_change:.2e}). Suggestion: {suggestion}"
    )]
    ConvergenceFailure {
        iterations: usize,
        last_change: f64,
        suggestion: String,
    },

    #[error("Optimization diverged at iteration {iteration}. Suggestion: {suggestion}")]
    DivergenceError {
        iteration: usize,
        suggestion: String,
    },

    #[error("Invalid model specification: {message}")]
    InvalidSpecification { message: String },

    #[error(
        "Perfect separation detected in logit/probit model. Variable(s) {variables:?} perfectly predict the outcome. MLE cannot converge. Suggestion: Remove or combine these predictors, or use Firth's penalized likelihood."
    )]
    PerfectSeparation { variables: Vec<String> },

    #[error(
        "Quasi-complete separation detected in logit/probit model. Variable(s) {variables:?} almost perfectly predict the outcome. Estimates may be unstable."
    )]
    QuasiSeparation { variables: Vec<String> },

    // ═══════════════════════════════════════════════════════════════════
    // Panel Data Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error(
        "Only {n_entities} entity groups found. Fixed effects estimation requires more variation. Suggestion: Use pooled OLS or check your entity identifier."
    )]
    InsufficientEntities { n_entities: usize },

    #[error(
        "Unbalanced panel data detected. Some entities have only {min_obs} observations while others have {max_obs}."
    )]
    UnbalancedPanel { min_obs: usize, max_obs: usize },

    // ═══════════════════════════════════════════════════════════════════
    // Clustering Errors (Cameron, Gelbach & Miller 2008; Cameron & Miller 2015)
    // ═══════════════════════════════════════════════════════════════════
    #[error(
        "Only {n_clusters} clusters. Cameron-Miller (2015) guidance: G < 20 has severe bias concerns, 20-50 is moderate, G >= 50 is generally adequate. Consider cluster-robust wild bootstrap for small G."
    )]
    FewClusters { n_clusters: usize },

    // ═══════════════════════════════════════════════════════════════════
    // IV/2SLS Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error(
        "Under-identification: {n_endogenous} endogenous variable(s) but only {n_instruments} excluded instrument(s). Need at least as many instruments as endogenous variables."
    )]
    UnderIdentified {
        n_endogenous: usize,
        n_instruments: usize,
    },

    #[error(
        "Weak instruments detected (first-stage F = {f_stat:.2}). Rule of thumb: F should be > 10. Consider finding stronger instruments."
    )]
    WeakInstruments { f_stat: f64 },

    // ═══════════════════════════════════════════════════════════════════
    // Time Series Errors
    // ═══════════════════════════════════════════════════════════════════
    #[error(
        "Insufficient time periods for VAR({lags}): need at least {required} observations, got {provided}"
    )]
    InsufficientTimePeriods {
        lags: usize,
        required: usize,
        provided: usize,
    },

    #[error(
        "Non-stationary series detected. Consider differencing the data or using VECM for cointegrated series."
    )]
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

    #[error("Computation error: {0}")]
    Computation(String),
}

/// Warnings that don't prevent estimation but should be noted.
#[derive(Debug, Clone, Serialize)]
pub enum EstimationWarning {
    /// Condition number is elevated but not critical
    HighConditionNumber { value: f64, threshold: f64 },

    /// Variance inflation factors indicate potential multicollinearity
    HighVIF { variable: String, vif: f64 },

    /// Few clusters for cluster-robust standard errors (Cameron-Miller 2015 guidance)
    FewClusters {
        n_clusters: usize,
        /// Severity level: "severe" (< 20), "moderate" (20-50), "adequate" (>= 50)
        severity: String,
    },

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
            Self::FewClusters {
                n_clusters,
                severity,
            } => {
                // Cameron, Gelbach & Miller (2008) and Cameron & Miller (2015) guidance
                match severity.as_str() {
                    "severe" => format!(
                        "Only {} clusters (< 20). Cameron-Miller (2015) warns of severe finite-sample \
                         bias. Consider cluster-robust wild bootstrap (Cameron et al. 2008) or \
                         bias-corrected estimates. Standard asymptotic inference is unreliable.",
                        n_clusters
                    ),
                    "moderate" => format!(
                        "Only {} clusters (20-50). Cameron-Miller (2015) notes moderate concerns. \
                         Consider cluster-robust wild bootstrap for more reliable inference. \
                         Asymptotic SEs may be downward biased.",
                        n_clusters
                    ),
                    _ => format!(
                        "{} clusters detected. Generally adequate for cluster-robust inference.",
                        n_clusters
                    ),
                }
            }
            Self::Heteroskedasticity { test_name, p_value } => {
                format!(
                    "{} test p-value = {:.4}. Consider using robust standard errors (HC0-HC3).",
                    test_name, p_value
                )
            }
            Self::Autocorrelation {
                test_name,
                statistic,
            } => {
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
            Self::SlowConvergence {
                iterations,
                tolerance,
            } => {
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

    #[test]
    fn test_levenshtein_distance() {
        // Same string
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        // One substitution
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        // One insertion
        assert_eq!(levenshtein_distance("hello", "helloo"), 1);
        // One deletion
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        // Multiple edits
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        // Case insensitive
        assert_eq!(levenshtein_distance("Hello", "HELLO"), 0);
        // Empty strings
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
    }

    #[test]
    fn test_suggest_similar_columns() {
        let available = vec![
            "sqft".to_string(),
            "price".to_string(),
            "bedrooms".to_string(),
            "bathrooms".to_string(),
        ];

        // Typo: sqfeet -> sqft (distance 2)
        let suggestions = suggest_similar_columns("sqfeet", &available, 3, 3);
        assert_eq!(suggestions, vec!["sqft"]);

        // Typo: bedroom -> bedrooms (distance 1)
        let suggestions = suggest_similar_columns("bedroom", &available, 3, 3);
        assert_eq!(suggestions, vec!["bedrooms"]);

        // No close match
        let suggestions = suggest_similar_columns("xyz", &available, 2, 3);
        assert!(suggestions.is_empty());

        // Exact match excluded (distance 0)
        let suggestions = suggest_similar_columns("price", &available, 3, 3);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_column_not_found_with_suggestion() {
        let err = EconError::ColumnNotFound {
            column: "sqfeet".to_string(),
            available: vec!["sqft".to_string(), "price".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("sqfeet"));
        assert!(msg.contains("Did you mean 'sqft'?"));
        assert!(msg.contains("Available columns"));
    }

    #[test]
    fn test_column_not_found_no_suggestion() {
        let err = EconError::ColumnNotFound {
            column: "xyz".to_string(),
            available: vec!["sqft".to_string(), "price".to_string()],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("xyz"));
        assert!(!msg.contains("Did you mean"));
        assert!(msg.contains("Available columns"));
    }
}
