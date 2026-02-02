//! Request types for data quality and cleaning tools.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for data quality profile.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DataQualityProfileRequest {
    /// Name/ID of the dataset to profile
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request for previewing a cleaning operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PreviewCleaningRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Type of cleaning operation to preview
    #[schemars(
        description = "The type of cleaning operation: 'trim', 'lowercase', 'uppercase', 'fill_na', 'drop_na', 'deduplicate', 'replace', or 'filter'."
    )]
    pub operation: String,

    /// Target column(s) - behavior depends on operation type
    #[schemars(
        description = "Column name(s) to apply the operation to. For some operations this can be omitted to apply to all columns."
    )]
    pub columns: Option<Vec<String>>,

    /// Strategy for fill_na operations
    #[schemars(
        description = "For fill_na: strategy to use ('mean', 'median', 'mode', 'forward', 'backward', 'constant')."
    )]
    pub strategy: Option<String>,

    /// Value for fill_na with constant, or replacement value
    #[schemars(
        description = "For fill_na with constant: the fill value. For replace: the new value."
    )]
    pub value: Option<String>,

    /// Old value for replace operation
    #[schemars(description = "For replace: the value to search for and replace.")]
    pub old_value: Option<String>,

    /// How to handle drop_na: 'any' or 'all'
    #[schemars(
        description = "For drop_na: 'any' (drop if any null) or 'all' (drop only if all null)."
    )]
    pub how: Option<String>,

    /// Keep strategy for deduplicate: 'first', 'last', or 'none'
    #[schemars(
        description = "For deduplicate: which duplicate to keep ('first', 'last', 'none')."
    )]
    pub keep: Option<String>,

    /// Operator for filter: '>', '<', '>=', '<=', '==', '!=', 'contains'
    #[schemars(
        description = "For filter: comparison operator ('>', '<', '>=', '<=', '==', '!=', 'contains')."
    )]
    pub operator: Option<String>,

    /// Value for filter comparison
    #[schemars(description = "For filter: value to compare against.")]
    pub filter_value: Option<String>,

    /// Number of sample changes to show
    #[schemars(description = "Number of example changes to include in the preview. Default is 5.")]
    pub sample_size: Option<usize>,
}

/// Request for verifying a cleaning operation after applying.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VerifyCleaningRequest {
    /// Name/ID of the original dataset (before cleaning)
    #[schemars(description = "Name or ID of the original dataset before cleaning.")]
    pub before_dataset: String,

    /// Name/ID of the cleaned dataset (after cleaning)
    #[schemars(description = "Name or ID of the cleaned dataset after applying the operation.")]
    pub after_dataset: String,

    /// Description of the operation that was performed
    #[schemars(description = "Description of the cleaning operation that was performed.")]
    pub operation_description: String,
}

/// Request to start a new cleaning session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CleaningSessionStartRequest {
    /// Name/ID of the dataset to start cleaning
    #[schemars(description = "Name or ID of the dataset to create a cleaning session for.")]
    pub dataset: String,

    /// Optional name for the session
    #[schemars(description = "Optional descriptive name for the cleaning session.")]
    pub session_name: Option<String>,
}

/// Request for cleaning session status.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CleaningSessionStatusRequest {
    /// Session ID
    #[schemars(description = "The session ID returned by cleaning_session_start.")]
    pub session_id: String,
}

/// Request to rollback a cleaning session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CleaningRollbackRequest {
    /// Session ID
    #[schemars(description = "The session ID to rollback.")]
    pub session_id: String,

    /// Optional checkpoint index to rollback to (defaults to previous checkpoint)
    #[schemars(
        description = "Checkpoint index to rollback to. If not provided, rolls back to the previous checkpoint."
    )]
    pub checkpoint_index: Option<usize>,
}

/// Request to apply a cleaning operation within a session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CleaningSessionApplyRequest {
    /// Session ID
    #[schemars(description = "The session ID to apply the operation to.")]
    pub session_id: String,

    /// Type of cleaning operation to apply
    #[schemars(
        description = "The type of cleaning operation: 'trim', 'lowercase', 'uppercase', 'fill_na', 'drop_na', 'deduplicate', 'replace', or 'filter'."
    )]
    pub operation: String,

    /// Target column(s) - behavior depends on operation type
    #[schemars(description = "Column name(s) to apply the operation to.")]
    pub columns: Option<Vec<String>>,

    /// Strategy for fill_na operations
    #[schemars(description = "For fill_na: strategy to use.")]
    pub strategy: Option<String>,

    /// Value for fill_na or replace
    #[schemars(description = "For fill_na with constant or replace: the value.")]
    pub value: Option<String>,

    /// Old value for replace operation
    #[schemars(description = "For replace: the value to search for.")]
    pub old_value: Option<String>,

    /// How to handle drop_na
    #[schemars(description = "For drop_na: 'any' or 'all'.")]
    pub how: Option<String>,

    /// Keep strategy for deduplicate
    #[schemars(description = "For deduplicate: 'first', 'last', or 'none'.")]
    pub keep: Option<String>,

    /// Operator for filter
    #[schemars(description = "For filter: comparison operator.")]
    pub operator: Option<String>,

    /// Value for filter comparison
    #[schemars(description = "For filter: value to compare against.")]
    pub filter_value: Option<String>,
}

/// Request to list all checkpoints in a session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CleaningSessionCheckpointsRequest {
    /// Session ID
    #[schemars(description = "The session ID to list checkpoints for.")]
    pub session_id: String,
}

/// Request to generate smart cleaning suggestions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SuggestCleaningRequest {
    /// The name/ID of the loaded dataset to analyze.
    #[schemars(description = "The name/ID of the loaded dataset.")]
    pub dataset: String,
    /// Minimum priority level to include (optional, default: all).
    #[schemars(
        description = "Minimum priority: 'low', 'medium', 'high', or 'critical'. Default: include all."
    )]
    pub min_priority: Option<String>,
    /// Maximum number of suggestions to return (optional).
    #[schemars(description = "Maximum number of suggestions to return. Default: all.")]
    pub limit: Option<usize>,
}

/// Request to list all active cleaning sessions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListCleaningSessionsRequest {}
