//! Request types for survival analysis tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for Kaplan-Meier survival curve estimation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct KaplanMeierRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Status column, coded as R's Surv()
    #[schemars(
        description = "Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event, e.g. R's lung) or TRUE/FALSE. Other codes are an error."
    )]
    pub event: String,

    /// Optional group column for stratified analysis
    #[schemars(
        description = "Optional column name for stratified analysis (e.g., 'treatment_group')."
    )]
    pub group: Option<String>,

    /// Confidence level
    #[schemars(
        description = "Confidence level for survival estimates. Default is 0.95. Must be between 0 and 1."
    )]
    pub confidence_level: Option<f64>,

    /// Confidence interval type for the survival curve.
    #[serde(default)]
    #[schemars(
        description = "Confidence interval type, as R survfit's conf.type: 'log' (default, R's default), 'log-log' or 'plain'."
    )]
    pub conf_type: Option<String>,
}

/// Request for Log-Rank test comparing survival curves.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LogRankRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Status column, coded as R's Surv()
    #[schemars(
        description = "Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event, e.g. R's lung) or TRUE/FALSE. Other codes are an error."
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
#[serde(deny_unknown_fields)]
pub struct CoxPhRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Status column, coded as R's Surv()
    #[schemars(
        description = "Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event, e.g. R's lung) or TRUE/FALSE. Other codes are an error."
    )]
    pub event: String,

    /// Covariate column names
    #[schemars(description = "Names of covariate columns to include in the model.")]
    pub covariates: Vec<String>,

    /// Method for handling ties
    #[schemars(
        description = "Method for handling tied event times: 'efron' (default, as coxph) or 'breslow'. Other values are an error."
    )]
    pub ties_method: Option<String>,

    /// Convergence tolerance
    #[schemars(
        description = "Relative change in the log partial likelihood (coxph's eps). Default 1e-9."
    )]
    pub tolerance: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum Newton-Raphson iterations. Default 25.")]
    pub max_iter: Option<usize>,

    /// Robust (Lin-Wei sandwich) standard errors
    #[schemars(
        description = "Robust (Lin-Wei sandwich) standard errors, as coxph(robust = TRUE). Default false; implied by cluster."
    )]
    #[serde(alias = "robust")]
    pub robust_se: Option<bool>,

    /// Cluster column
    #[schemars(
        description = "Column whose values define clusters for clustered robust standard errors, as coxph(... + cluster(column)). Rows with a missing cluster label are dropped and reported."
    )]
    pub cluster: Option<String>,
}

/// Request for Accelerated Failure Time (AFT) model.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AftRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Time-to-event column name
    #[schemars(
        description = "Name of the column containing time-to-event or time-to-censoring values."
    )]
    pub time: String,

    /// Status column, coded as R's Surv()
    #[schemars(
        description = "Status column, coded as R's Surv(): 0/1 (1 = event), 1/2 (2 = event, e.g. R's lung) or TRUE/FALSE. Other codes are an error."
    )]
    pub event: String,

    /// Covariate column names
    #[schemars(description = "Names of covariate columns to include in the model.")]
    pub covariates: Vec<String>,

    /// Distribution for the AFT model
    #[schemars(
        description = "'weibull' (default), 'exponential' (scale fixed at 1), 'lognormal' or 'loglogistic', as survreg(dist =). Other values are an error."
    )]
    pub distribution: Option<String>,

    /// Convergence tolerance
    #[schemars(
        description = "Relative change in the log-likelihood (default 1e-9), as survreg's rel.tolerance."
    )]
    pub tolerance: Option<f64>,

    /// Maximum iterations
    #[schemars(description = "Maximum iterations for optimization. Default is 100.")]
    pub max_iter: Option<usize>,
}

/// Request for Competing Risks analysis (Aalen-Johansen estimator).
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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
        description = "Event type column: 0 = censored, 1, 2, ... = event types (whole numbers up to 255)."
    )]
    pub event: String,

    /// Confidence level
    #[schemars(
        description = "Confidence level for cumulative incidence estimates. Default is 0.95. Must be between 0 and 1."
    )]
    pub confidence_level: Option<f64>,
}
