//! Request types for survival analysis tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for Kaplan-Meier survival curve estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KaplanMeierRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Event indicator column name
    #[schemars(
        description = "Name of the column indicating event occurrence (1=event, 0=censored)."
    )]
    pub event: String,

    /// Optional group column for stratified analysis
    #[schemars(
        description = "Optional column name for stratified analysis (e.g., 'treatment_group')."
    )]
    pub group: Option<String>,

    /// Confidence level
    #[schemars(description = "Confidence level for survival estimates. Default is 0.95.")]
    pub confidence_level: Option<f64>,
}

/// Request for Log-Rank test comparing survival curves.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogRankRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Event indicator column name
    #[schemars(
        description = "Name of the column indicating event occurrence (1=event, 0=censored)."
    )]
    pub event: String,

    /// Group column name for comparison
    #[schemars(
        description = "Name of the column defining groups to compare (e.g., 'treatment_group')."
    )]
    pub group: String,
}

/// Request for Cox Proportional Hazards model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CoxPhRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Event indicator column name
    #[schemars(
        description = "Name of the column indicating event occurrence (1=event, 0=censored)."
    )]
    pub event: String,

    /// Covariate column names
    #[schemars(description = "Names of covariate columns to include in the model.")]
    pub covariates: Vec<String>,

    /// Method for handling ties
    #[schemars(
        description = "Method for handling tied event times: 'efron' (default) or 'breslow'."
    )]
    pub ties_method: Option<String>,

    /// Convergence tolerance
    #[schemars(
        description = "Convergence tolerance for Newton-Raphson optimization. Default is 1e-9."
    )]
    pub tolerance: Option<f64>,

    /// Maximum iterations
    #[schemars(
        description = "Maximum iterations for Newton-Raphson optimization. Default is 100."
    )]
    pub max_iter: Option<usize>,
}

/// Request for Accelerated Failure Time (AFT) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct AftRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Event indicator column name
    #[schemars(
        description = "Name of the column indicating event occurrence (1=event, 0=censored)."
    )]
    pub event: String,

    /// Covariate column names
    #[schemars(description = "Names of covariate columns to include in the model.")]
    pub covariates: Vec<String>,

    /// Distribution for the AFT model
    #[schemars(
        description = "Distribution assumption: 'weibull' (default), 'exponential', 'lognormal', or 'loglogistic'."
    )]
    pub distribution: Option<String>,

    /// Convergence tolerance
    #[schemars(description = "Convergence tolerance for optimization. Default is 1e-9.")]
    pub tolerance: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum iterations for optimization. Default is 100.")]
    pub max_iter: Option<usize>,
}

/// Request for Competing Risks analysis (Aalen-Johansen estimator).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompetingRisksRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Event type column name
    #[schemars(
        description = "Name of the column indicating event type (0=censored, 1=event of interest, 2=competing event, etc.)."
    )]
    pub event: String,

    /// Confidence level
    #[schemars(
        description = "Confidence level for cumulative incidence estimates. Default is 0.95."
    )]
    pub confidence_level: Option<f64>,
}
