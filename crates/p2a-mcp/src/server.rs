//! Analytics MCP Server implementation.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

use rmcp::{
    handler::server::router::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::*,
    tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};
use schemars::JsonSchema;
use serde::Deserialize;

use p2a_core::{
    data::{
        DataLoader, Dataset, DatasetInfo,
        // Database connectivity
        query_sqlite, list_sqlite_tables, sqlite_table_schema,
        query_duckdb, list_duckdb_tables, duckdb_table_schema,
        query_file_with_duckdb,
        // Data quality profiling
        generate_quality_profile,
        // Verification and preview
        preview_cleaning, verify_cleaning, CleaningOperation,
        // Session management
        CleaningSession,
        // Smart suggestions
        generate_suggestions, SuggestionPriority,
        // Data munging
        munging::{
            // Transform operations
            filter, select, drop_columns, rename, mutate, sort, sample,
            MutateExpr, ArithOp,
            // Clean operations
            drop_na, fill_na, deduplicate, FillStrategy,
            trim, to_lowercase, to_uppercase, replace,
            // Regex and string operations
            regex_replace, regex_extract, regex_count,
            str_split, str_concat, str_substring, str_length,
            // Join operations
            left_join, right_join, inner_join, full_join, concat,
            // Aggregate operations
            group_by, value_counts, AggFn, AggSpec,
            // Reshape operations
            pivot, melt,
            // Feature engineering
            lag, lead, diff, pct_change, standardize, normalize, bin, one_hot_encode,
            BinStrategy,
        },
    },
    regression::{run_ols, run_ols_clustered, run_diagnostics, CovarianceType},
    stats::{correlation_matrix, DescriptiveStats},
    // Econometrics
    run_fixed_effects, run_random_effects, run_hausman_test, run_iv2sls, run_did,
    run_logit, run_probit, run_first_stage_diagnostics,
    run_hdfe, HdfeConfig,
    // Treatment effects
    run_ipw_treatment, run_doubly_robust, IpwConfig, DoublyRobustConfig, Estimand, DRMethod,
    // Mediation analysis
    run_mediation_analysis, MediationConfig,
    // Time series
    run_var, run_varma, run_vecm, run_var_irf,
    // Forecasting
    run_arima, forecast_arima, run_mstl,
    run_changepoint, run_binary_segmentation, CostFunction,
    // Machine Learning
    kmeans, dbscan, pca, hierarchical, tsne, random_forest, linear_svm,
    Linkage,
    // Visualization
    histogram, scatter_plot, box_plot, line_chart, correlation_heatmap,
    event_study_plot, coefficient_plot, irf_plot, residual_diagnostics,
    ChartConfig,
    // Reports
    HtmlReport, ReportSection, ReportTable,
};

/// The main analytics server that handles MCP requests.
#[derive(Clone)]
pub struct AnalyticsServer {
    /// Currently loaded datasets, keyed by a unique ID
    datasets: Arc<RwLock<HashMap<String, Dataset>>>,
    /// Active cleaning sessions, keyed by session ID
    cleaning_sessions: Arc<RwLock<HashMap<String, CleaningSession>>>,
    /// Global random seed for ML reproducibility
    global_seed: Arc<RwLock<Option<u64>>>,
    /// Tool router for handling tool calls
    tool_router: ToolRouter<Self>,
}

// ============================================================================
// Tool Input/Output Types
// ============================================================================

/// Request to load a dataset from a file.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadDatasetRequest {
    /// Path to the data file
    #[schemars(description = "Absolute or relative path to the data file. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats.")]
    pub path: String,

    /// Optional name/identifier for the dataset
    #[schemars(description = "Optional name to identify this dataset. If not provided, the filename will be used.")]
    pub name: Option<String>,
}

/// Request to describe a loaded dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DescribeDatasetRequest {
    /// Name/ID of the dataset to describe
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request to preview rows from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HeadDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Number of rows to return (default: 5)
    #[schemars(description = "Number of rows to return. Default is 5.")]
    pub n: Option<usize>,
}

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
    #[schemars(description = "The type of cleaning operation: 'trim', 'lowercase', 'uppercase', 'fill_na', 'drop_na', 'deduplicate', 'replace', or 'filter'.")]
    pub operation: String,

    /// Target column(s) - behavior depends on operation type
    #[schemars(description = "Column name(s) to apply the operation to. For some operations this can be omitted to apply to all columns.")]
    pub columns: Option<Vec<String>>,

    /// Strategy for fill_na operations
    #[schemars(description = "For fill_na: strategy to use ('mean', 'median', 'mode', 'forward', 'backward', 'constant').")]
    pub strategy: Option<String>,

    /// Value for fill_na with constant, or replacement value
    #[schemars(description = "For fill_na with constant: the fill value. For replace: the new value.")]
    pub value: Option<String>,

    /// Old value for replace operation
    #[schemars(description = "For replace: the value to search for and replace.")]
    pub old_value: Option<String>,

    /// How to handle drop_na: 'any' or 'all'
    #[schemars(description = "For drop_na: 'any' (drop if any null) or 'all' (drop only if all null).")]
    pub how: Option<String>,

    /// Keep strategy for deduplicate: 'first', 'last', or 'none'
    #[schemars(description = "For deduplicate: which duplicate to keep ('first', 'last', 'none').")]
    pub keep: Option<String>,

    /// Operator for filter: '>', '<', '>=', '<=', '==', '!=', 'contains'
    #[schemars(description = "For filter: comparison operator ('>', '<', '>=', '<=', '==', '!=', 'contains').")]
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
    #[schemars(description = "Checkpoint index to rollback to. If not provided, rolls back to the previous checkpoint.")]
    pub checkpoint_index: Option<usize>,
}

/// Request to apply a cleaning operation within a session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CleaningSessionApplyRequest {
    /// Session ID
    #[schemars(description = "The session ID to apply the operation to.")]
    pub session_id: String,

    /// Type of cleaning operation to apply
    #[schemars(description = "The type of cleaning operation: 'trim', 'lowercase', 'uppercase', 'fill_na', 'drop_na', 'deduplicate', 'replace', or 'filter'.")]
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
    #[schemars(description = "Minimum priority: 'low', 'medium', 'high', or 'critical'. Default: include all.")]
    pub min_priority: Option<String>,
    /// Maximum number of suggestions to return (optional).
    #[schemars(description = "Maximum number of suggestions to return. Default: all.")]
    pub limit: Option<usize>,
}

/// Request to list all active cleaning sessions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListCleaningSessionsRequest {}

/// Request for correlation matrix.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CorrelationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
}

/// Request for OLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for regression diagnostics.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiagnosticsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for OLS with clustered standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsClusteredRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// First cluster dimension column (e.g., "firm_id")
    #[schemars(description = "Column name for first clustering dimension (e.g., 'firm_id').")]
    pub cluster1: String,

    /// Second cluster dimension column (optional, for two-way clustering)
    #[schemars(description = "Optional column for second clustering dimension (e.g., 'year'). If provided, two-way clustering is used.")]
    pub cluster2: Option<String>,
}

// ============================================================================
// Econometrics Tool Input Types
// ============================================================================

/// Request for Panel Fixed Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelFERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id').")]
    pub entity_var: String,
}

/// Request for Panel Random Effects regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelRERequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id').")]
    pub entity_var: String,
}

/// Request for Hausman specification test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HausmanRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Entity/individual identifier column
    #[schemars(description = "Column name for entity/individual identifier (e.g., 'firm_id', 'person_id').")]
    pub entity_var: String,
}

/// Request for IV/2SLS regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IV2SLSRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Exogenous independent variables
    #[schemars(description = "Names of exogenous independent variable columns (not instrumented).")]
    pub x_exog: Vec<String>,

    /// Endogenous variable to be instrumented
    #[schemars(description = "Names of endogenous variables that need instruments.")]
    pub x_endog: Vec<String>,

    /// Instrumental variables
    #[schemars(description = "Names of instrument columns (excluded from structural equation).")]
    pub instruments: Vec<String>,

    /// Use robust standard errors
    #[schemars(description = "Whether to use heteroskedasticity-robust standard errors. Default is true.")]
    pub robust: Option<bool>,
}

/// Request for first-stage diagnostics.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FirstStageRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Endogenous variable name
    #[schemars(description = "Name of the endogenous variable to test instrument strength for.")]
    pub endogenous_var: String,

    /// Instrument variable names
    #[schemars(description = "Names of the instrumental variables (e.g., ['parents_edu', 'distance_to_college']).")]
    pub instruments: Vec<String>,

    /// Control variable names (optional)
    #[schemars(description = "Optional control variables to include in first-stage regression.")]
    pub controls: Option<Vec<String>>,
}

/// Request for Difference-in-Differences estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiDRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable column name
    #[schemars(description = "Name of the outcome/dependent variable column.")]
    pub dep_var: String,

    /// Treatment group indicator column (0/1)
    #[schemars(description = "Column indicating treatment group (1 = treated, 0 = control).")]
    pub treatment_var: String,

    /// Post-treatment period indicator column (0/1)
    #[schemars(description = "Column indicating post-treatment period (1 = post, 0 = pre).")]
    pub post_var: String,
}

/// Request for Inverse Probability Weighting (IPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IpwRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary.")]
    pub treatment: String,

    /// Covariate columns for propensity score model
    #[schemars(description = "Names of covariate columns to include in propensity score model.")]
    pub covariates: Vec<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(description = "Treatment effect estimand: 'ate' for Average Treatment Effect (default), 'att' for Average Treatment Effect on Treated.")]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05.")]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(description = "Number of bootstrap replications for standard error estimation. Default is 999.")]
    pub bootstrap: Option<usize>,
}

/// Request for Doubly Robust (AIPW) treatment effect estimation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DoublyRobustRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary.")]
    pub treatment: String,

    /// Covariate columns for propensity score and outcome models
    #[schemars(description = "Names of covariate columns to include in both propensity score and outcome models.")]
    pub covariates: Vec<String>,

    /// Estimation method: 'aipw' (default), 'ipw', or 'regression'
    #[schemars(description = "Estimation method: 'aipw' for Augmented IPW (default, doubly robust), 'ipw' for IPW only, 'regression' for outcome regression only.")]
    pub method: Option<String>,

    /// Estimand: 'ate' (Average Treatment Effect) or 'att' (Average Treatment Effect on Treated)
    #[schemars(description = "Treatment effect estimand: 'ate' for Average Treatment Effect (default), 'att' for Average Treatment Effect on Treated.")]
    pub estimand: Option<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05.")]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(description = "Number of bootstrap replications for standard error estimation. Default is 999.")]
    pub bootstrap: Option<usize>,
}

/// Request for Causal Mediation Analysis.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MediationRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Outcome variable column name
    #[schemars(description = "Name of the outcome variable column.")]
    pub outcome: String,

    /// Treatment indicator column (0/1)
    #[schemars(description = "Column indicating treatment status (1 = treated, 0 = control). Must be binary.")]
    pub treatment: String,

    /// Mediator variable column name
    #[schemars(description = "Name of the mediator variable column - the intermediate variable through which treatment may affect the outcome.")]
    pub mediator: String,

    /// Covariate columns for propensity score models
    #[schemars(description = "Names of covariate columns for adjustment in propensity score models.")]
    pub covariates: Vec<String>,

    /// Trimming threshold for extreme propensity scores
    #[schemars(description = "Trim observations with propensity scores below trim or above 1-trim. Default is 0.05.")]
    pub trim: Option<f64>,

    /// Number of bootstrap replications for standard errors
    #[schemars(description = "Number of bootstrap replications for standard error estimation. Default is 999.")]
    pub bootstrap: Option<usize>,
}

/// Request for Logit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (must be binary 0/1)
    #[schemars(description = "Name of the dependent variable (Y) column. Must be binary (0/1).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for Probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name (must be binary 0/1)
    #[schemars(description = "Name of the dependent variable (Y) column. Must be binary (0/1).")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,
}

/// Request for High-Dimensional Fixed Effects (HDFE) regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PanelHdfeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Dependent variable (Y) column name
    #[schemars(description = "Name of the dependent variable (Y) column.")]
    pub y: String,

    /// Independent variables (X) column names
    #[schemars(description = "Names of the independent variable (X) columns.")]
    pub x: Vec<String>,

    /// Fixed effect columns to absorb
    #[schemars(description = "Column names for fixed effects to absorb (e.g., ['firm_id', 'year']). Supports multiple dimensions.")]
    pub fe: Vec<String>,

    /// Convergence tolerance for MAP algorithm
    #[schemars(description = "Convergence tolerance for the Method of Alternating Projections. Default is 1e-8.")]
    pub tolerance: Option<f64>,

    /// Maximum iterations for MAP algorithm
    #[schemars(description = "Maximum iterations for the demeaning algorithm. Default is 10000.")]
    pub max_iterations: Option<usize>,

    /// Standard error type
    #[schemars(description = "Standard error type: 'standard', 'hc0', 'hc1' (default), 'hc2', or 'hc3'.")]
    pub se_type: Option<String>,
}

// ============================================================================
// Time Series Tool Input Types
// ============================================================================

/// Request for VAR (Vector Autoregression) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VAR model
    #[schemars(description = "Names of the columns to include in the VAR model (e.g., ['gdp', 'inflation', 'interest_rate']).")]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags to include in the VAR model.")]
    pub lags: usize,
}

/// Request for VARMA (Vector ARMA) model.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarmaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VARMA model
    #[schemars(description = "Names of the columns to include in the VARMA model.")]
    pub columns: Vec<String>,

    /// AR lags (p)
    #[schemars(description = "Number of autoregressive (AR) lags.")]
    pub p: usize,

    /// MA lags (q)
    #[schemars(description = "Number of moving average (MA) lags.")]
    pub q: usize,
}

/// Request for VECM (Vector Error Correction Model).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VecmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VECM model
    #[schemars(description = "Names of the columns to include in the VECM model. Should be I(1) cointegrated series.")]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags for the VECM (must be at least 2).")]
    pub lags: usize,

    /// Cointegration rank
    #[schemars(description = "Cointegration rank (number of cointegrating relationships). Must be between 1 and k-1 where k is the number of variables.")]
    pub rank: usize,
}

/// Request for VAR Impulse Response Functions.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarIrfRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to include in the VAR model
    #[schemars(description = "Names of the columns to include in the VAR model.")]
    pub columns: Vec<String>,

    /// Number of lags
    #[schemars(description = "Number of lags for the VAR model.")]
    pub lags: usize,

    /// Number of IRF steps/periods
    #[schemars(description = "Number of periods to compute impulse responses for.")]
    pub steps: usize,
}

// ============================================================================
// Forecasting Tool Input Types
// ============================================================================

/// Request for ARIMA model fitting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// AR order (p)
    #[schemars(description = "Number of autoregressive (AR) terms.")]
    pub p: usize,

    /// Differencing order (d)
    #[schemars(description = "Number of differences to make the series stationary.")]
    pub d: usize,

    /// MA order (q)
    #[schemars(description = "Number of moving average (MA) terms.")]
    pub q: usize,
}

/// Request for ARIMA forecasting.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ArimaForecastRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// AR order (p)
    #[schemars(description = "Number of autoregressive (AR) terms.")]
    pub p: usize,

    /// Differencing order (d)
    #[schemars(description = "Number of differences to make the series stationary.")]
    pub d: usize,

    /// MA order (q)
    #[schemars(description = "Number of moving average (MA) terms.")]
    pub q: usize,

    /// Forecast horizon
    #[schemars(description = "Number of periods to forecast ahead.")]
    pub horizon: usize,
}

/// Request for MSTL decomposition.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MstlRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Seasonal periods
    #[schemars(description = "Seasonal periods to extract (e.g., [7, 365] for daily data with weekly and yearly seasonality).")]
    pub periods: Vec<usize>,
}

/// Request for changepoint detection.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChangepointRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name with time series values
    #[schemars(description = "Name of the column containing the time series values.")]
    pub column: String,

    /// Penalty for adding a changepoint (optional, uses BIC if not specified)
    #[schemars(description = "Penalty for adding a changepoint. Higher values = fewer changepoints. Default uses BIC (log(n)).")]
    pub penalty: Option<f64>,

    /// Minimum segment length between changepoints
    #[schemars(description = "Minimum number of observations between changepoints. Default is 2.")]
    pub min_segment_length: Option<usize>,

    /// Detection method: 'pelt' or 'binary'
    #[schemars(description = "Algorithm to use: 'pelt' (Pruned Exact Linear Time, default) or 'binary' (Binary Segmentation).")]
    pub method: Option<String>,

    /// Type of change to detect: 'mean', 'variance', or 'both'
    #[schemars(description = "Type of change to detect: 'mean' (default), 'variance', or 'both'.")]
    pub change_type: Option<String>,
}

// ============================================================================
// Report Generation Tool Input Types
// ============================================================================

/// A section in the HTML report.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReportSectionInput {
    /// Section title
    #[schemars(description = "Title for this section of the report.")]
    pub title: String,

    /// Content items for the section
    #[schemars(description = "Content items to include in this section.")]
    pub content: Vec<ReportContentInput>,
}

/// Content item for a report section.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReportContentInput {
    /// Type of content: 'text', 'code', 'table', 'chart', or 'stats'
    #[schemars(description = "Type of content: 'text' (paragraph), 'code' (code block), 'table' (data table), 'chart' (base64 image), or 'stats' (key-value pairs).")]
    pub content_type: String,

    /// Text content (for text and code types)
    #[schemars(description = "Text content for 'text' or 'code' types.")]
    pub text: Option<String>,

    /// Programming language (for code blocks)
    #[schemars(description = "Programming language for code block syntax highlighting.")]
    pub language: Option<String>,

    /// Table headers (for table type)
    #[schemars(description = "Column headers for table content.")]
    pub headers: Option<Vec<String>>,

    /// Table rows (for table type) - each row is a list of cell values
    #[schemars(description = "Table rows, where each row is a list of string values.")]
    pub rows: Option<Vec<Vec<String>>>,

    /// Table caption
    #[schemars(description = "Caption for the table.")]
    pub caption: Option<String>,

    /// Base64-encoded chart image (for chart type)
    #[schemars(description = "Base64-encoded PNG image data for chart content.")]
    pub image_base64: Option<String>,

    /// Chart title
    #[schemars(description = "Title for the chart.")]
    pub chart_title: Option<String>,

    /// Chart caption
    #[schemars(description = "Caption for the chart.")]
    pub chart_caption: Option<String>,

    /// Key-value statistics (for stats type)
    #[schemars(description = "Key-value pairs for statistics display. Format: [[key, value], ...]")]
    pub stats: Option<Vec<Vec<String>>>,
}

/// Request to generate an HTML report.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GenerateReportRequest {
    /// Report title
    #[schemars(description = "Title for the report.")]
    pub title: String,

    /// Report subtitle (optional)
    #[schemars(description = "Optional subtitle or description for the report.")]
    pub subtitle: Option<String>,

    /// Author name (optional)
    #[schemars(description = "Optional author name.")]
    pub author: Option<String>,

    /// Report sections
    #[schemars(description = "Sections to include in the report.")]
    pub sections: Vec<ReportSectionInput>,
}

// ============================================================================
// Machine Learning Tool Input Types
// ============================================================================

/// Request for K-means clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KMeansRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Number of clusters (k)
    #[schemars(description = "Number of clusters to create.")]
    pub k: usize,

    /// Maximum iterations (optional, default: 300)
    #[schemars(description = "Maximum number of iterations. Default is 300.")]
    pub max_iterations: Option<usize>,

    /// Number of initializations (optional, default: 10)
    #[schemars(description = "Number of random initializations to try. Default is 10.")]
    pub n_init: Option<usize>,

    /// Random seed for reproducibility (optional)
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for DBSCAN clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DBSCANRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Epsilon (neighborhood radius)
    #[schemars(description = "Maximum distance between two samples for them to be considered in the same neighborhood.")]
    pub eps: f64,

    /// Minimum samples for core point
    #[schemars(description = "Minimum number of samples in a neighborhood for a point to be considered a core point.")]
    pub min_samples: usize,
}

/// Request for PCA (Principal Component Analysis).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PCARequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features
    #[schemars(description = "Names of the numeric columns to include in PCA.")]
    pub columns: Vec<String>,

    /// Number of principal components to keep (optional)
    #[schemars(description = "Number of principal components to keep. If not specified, keeps all components.")]
    pub n_components: Option<usize>,
}

/// Request for Hierarchical clustering.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HierarchicalRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features for clustering
    #[schemars(description = "Names of the numeric columns to use as features for clustering.")]
    pub columns: Vec<String>,

    /// Number of clusters to cut the dendrogram into (optional)
    #[schemars(description = "Number of clusters to create. If not specified, uses distance_threshold.")]
    pub n_clusters: Option<usize>,

    /// Distance threshold for cutting the dendrogram (optional)
    #[schemars(description = "Distance threshold for cutting. Used if n_clusters is not specified.")]
    pub distance_threshold: Option<f64>,

    /// Linkage method
    #[schemars(description = "Linkage method: 'single', 'complete', 'average', or 'ward'. Default is 'ward'.")]
    pub linkage: Option<String>,
}

/// Request for t-SNE dimensionality reduction.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TsneRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Columns to use as features
    #[schemars(description = "Names of the numeric columns to include in t-SNE.")]
    pub columns: Vec<String>,

    /// Number of output dimensions (default: 2)
    #[schemars(description = "Number of output dimensions. Default is 2.")]
    pub n_components: Option<usize>,

    /// Perplexity parameter (default: 30.0)
    #[schemars(description = "Perplexity parameter, related to number of nearest neighbors. Default is 30.")]
    pub perplexity: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Learning rate (default: 200.0)
    #[schemars(description = "Learning rate for optimization. Default is 200.")]
    pub learning_rate: Option<f64>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Random Forest regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RandomForestRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the target column (Y variable).")]
    pub target: String,

    /// Number of trees (default: 100)
    #[schemars(description = "Number of trees in the forest. Default is 100.")]
    pub n_trees: Option<usize>,

    /// Maximum tree depth (default: 10)
    #[schemars(description = "Maximum depth of each tree. Default is 10.")]
    pub max_depth: Option<usize>,

    /// Minimum samples to split (default: 2)
    #[schemars(description = "Minimum samples required to split a node. Default is 2.")]
    pub min_samples_split: Option<usize>,

    /// Max features per split
    #[schemars(description = "Max features to consider per split: 'sqrt', 'log2', 'all', or a number. Default is 'sqrt'.")]
    pub max_features: Option<String>,

    /// Random seed for reproducibility
    #[schemars(description = "Optional random seed for reproducible results.")]
    pub seed: Option<u64>,
}

/// Request for Linear SVM classification.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SvmRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Feature column names
    #[schemars(description = "Names of the feature columns (X variables).")]
    pub features: Vec<String>,

    /// Target column name
    #[schemars(description = "Name of the binary target column (Y variable). Must have exactly 2 unique values.")]
    pub target: String,

    /// Regularization parameter C (default: 1.0)
    #[schemars(description = "Regularization parameter C. Larger values = less regularization. Default is 1.0.")]
    pub c: Option<f64>,

    /// Maximum iterations (default: 1000)
    #[schemars(description = "Maximum number of iterations. Default is 1000.")]
    pub max_iterations: Option<usize>,

    /// Convergence tolerance (default: 1e-3)
    #[schemars(description = "Convergence tolerance. Default is 0.001.")]
    pub tolerance: Option<f64>,
}

// ============================================================================
// Database Tool Input Types
// ============================================================================

/// Request to query a SQLite database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteQueryRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file (.db, .sqlite, .sqlite3).")]
    pub db_path: String,

    /// SQL query to execute
    #[schemars(description = "SQL query to execute (SELECT statements only recommended).")]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the resulting dataset. If not provided, a default name will be generated.")]
    pub name: Option<String>,
}

/// Request to list tables in a SQLite database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteListTablesRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file.")]
    pub db_path: String,
}

/// Request to get schema for a SQLite table.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SqliteSchemaRequest {
    /// Path to the SQLite database file
    #[schemars(description = "Path to the SQLite database file.")]
    pub db_path: String,

    /// Table name
    #[schemars(description = "Name of the table to get schema for.")]
    pub table_name: String,
}

/// Request to query a DuckDB database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBQueryRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file (.duckdb, .db). Use ':memory:' for in-memory database.")]
    pub db_path: String,

    /// SQL query to execute
    #[schemars(description = "SQL query to execute. DuckDB supports advanced analytics SQL.")]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the resulting dataset.")]
    pub name: Option<String>,
}

/// Request to list tables in a DuckDB database.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBListTablesRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file.")]
    pub db_path: String,
}

/// Request to get schema for a DuckDB table.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBSchemaRequest {
    /// Path to the DuckDB database file
    #[schemars(description = "Path to the DuckDB database file.")]
    pub db_path: String,

    /// Table name
    #[schemars(description = "Name of the table to get schema for.")]
    pub table_name: String,
}

/// Request to query a file (Parquet, CSV) using DuckDB SQL.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DuckDBFileQueryRequest {
    /// Path to the data file (Parquet or CSV)
    #[schemars(description = "Path to the data file (.parquet, .csv). DuckDB can query these files directly with SQL.")]
    pub file_path: String,

    /// SQL query to execute
    #[schemars(description = "SQL query to execute. Use {file} as placeholder for the file path. Example: 'SELECT * FROM {file} WHERE amount > 100'")]
    pub query: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the resulting dataset. If not provided, one will be generated.")]
    pub name: Option<String>,
}

// ============================================================================
// Visualization Request Types
// ============================================================================

/// Request to generate a histogram.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HistogramRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name to plot
    #[schemars(description = "Name of the numeric column to create histogram from.")]
    pub column: String,

    /// Number of bins (optional, auto-calculated if not specified)
    #[schemars(description = "Number of bins for the histogram. If not specified, uses Sturges' rule.")]
    pub bins: Option<usize>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a scatter plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ScatterPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values.")]
    pub x_column: String,

    /// Y-axis column name
    #[schemars(description = "Name of the column for Y-axis values.")]
    pub y_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a line chart.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LineChartRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// X-axis column name
    #[schemars(description = "Name of the column for X-axis values (e.g., time index).")]
    pub x_column: String,

    /// Y-axis column names (one or more series)
    #[schemars(description = "Names of the columns to plot as lines (can be multiple for multi-series).")]
    pub y_columns: Vec<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a box plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BoxPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to include in box plot
    #[schemars(description = "Names of numeric columns to create box plots for.")]
    pub columns: Vec<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a correlation heatmap.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct HeatmapRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column names to include (optional, uses all numeric if not specified)
    #[schemars(description = "Names of numeric columns to include. If not specified, uses all numeric columns.")]
    pub columns: Option<Vec<String>>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the heatmap.")]
    pub title: Option<String>,
}

/// Request to generate an event study plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct EventStudyRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for time/period relative to treatment
    #[schemars(description = "Column with time periods relative to treatment (e.g., -3, -2, -1, 0, 1, 2, 3).")]
    pub time_column: String,

    /// Column name for point estimates
    #[schemars(description = "Column with coefficient estimates at each time period.")]
    pub estimate_column: String,

    /// Column name for lower confidence interval bound
    #[schemars(description = "Column with lower bound of confidence interval.")]
    pub ci_lower_column: String,

    /// Column name for upper confidence interval bound
    #[schemars(description = "Column with upper bound of confidence interval.")]
    pub ci_upper_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate a coefficient plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CoefficientPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for variable/coefficient names
    #[schemars(description = "Column with variable names or coefficient labels.")]
    pub name_column: String,

    /// Column name for coefficient estimates
    #[schemars(description = "Column with coefficient point estimates.")]
    pub estimate_column: String,

    /// Column name for lower confidence interval bound
    #[schemars(description = "Column with lower bound of confidence interval.")]
    pub ci_lower_column: String,

    /// Column name for upper confidence interval bound
    #[schemars(description = "Column with upper bound of confidence interval.")]
    pub ci_upper_column: String,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,

    /// Horizontal orientation (optional, default: true)
    #[schemars(description = "If true, draw horizontal error bars (default). If false, draw vertical.")]
    pub horizontal: Option<bool>,
}

/// Request to generate an IRF (Impulse Response Function) plot.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IrfPlotRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// Column name for time horizon
    #[schemars(description = "Column with time horizon (0, 1, 2, ...).")]
    pub horizon_column: String,

    /// Column name for response values
    #[schemars(description = "Column with impulse response values.")]
    pub response_column: String,

    /// Column name for lower confidence interval bound (optional)
    #[schemars(description = "Optional column with lower bound of confidence interval.")]
    pub ci_lower_column: Option<String>,

    /// Column name for upper confidence interval bound (optional)
    #[schemars(description = "Optional column with upper bound of confidence interval.")]
    pub ci_upper_column: Option<String>,

    /// Label for the shock (optional)
    #[schemars(description = "Optional label for the shock variable.")]
    pub shock_label: Option<String>,

    /// Label for the response (optional)
    #[schemars(description = "Optional label for the response variable.")]
    pub response_label: Option<String>,

    /// Chart title (optional)
    #[schemars(description = "Optional title for the chart.")]
    pub title: Option<String>,
}

/// Request to generate residual diagnostic plots.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResidualDiagnosticsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset containing regression results.")]
    pub dataset: String,

    /// Column name for fitted/predicted values
    #[schemars(description = "Column with fitted (predicted) values from regression.")]
    pub fitted_column: String,

    /// Column name for residual values
    #[schemars(description = "Column with residual values (observed - fitted).")]
    pub residuals_column: String,

    /// Column name for leverage (hat) values (optional)
    #[schemars(description = "Optional column with leverage (hat) values. If not provided, will be estimated.")]
    pub leverage_column: Option<String>,
}

/// Request to batch process multiple datasets with the same operation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BatchProcessRequest {
    /// Names/IDs of datasets to process
    #[schemars(description = "List of dataset names to process. Each must be previously loaded.")]
    pub datasets: Vec<String>,

    /// Operation to perform on each dataset
    #[schemars(description = "Operation to perform: 'describe' (summary stats), 'correlation' (correlation matrix), or 'ols' (regression).")]
    pub operation: String,

    /// Columns to analyze (optional, defaults to all numeric for describe/correlation)
    #[schemars(description = "List of column names to analyze. For 'ols', first column is dependent variable.")]
    pub columns: Option<Vec<String>>,

    /// Whether to return combined summary across all datasets
    #[schemars(description = "If true, also returns an aggregated summary across all datasets.")]
    pub combine_results: Option<bool>,
}

/// Request to compare the same columns across multiple datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompareDatasetRequest {
    /// Names/IDs of datasets to compare
    #[schemars(description = "List of dataset names to compare. Each must be previously loaded.")]
    pub datasets: Vec<String>,

    /// Columns to compare
    #[schemars(description = "List of column names to compare across datasets.")]
    pub columns: Vec<String>,

    /// Type of comparison
    #[schemars(description = "Comparison type: 'summary' (side-by-side stats), 'correlation' (correlation differences), or 'distribution' (distribution comparison).")]
    pub comparison_type: Option<String>,
}

/// Request to export the current analysis session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExportSessionRequest {
    /// Path to save the session file
    #[schemars(description = "File path to save the session (JSON format). If not provided, returns session data as string.")]
    pub file_path: Option<String>,

    /// Whether to include dataset data (default: true)
    #[schemars(description = "Include full dataset data. If false, only metadata and file paths are saved.")]
    pub include_data: Option<bool>,
}

/// Request to import a previously exported session.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportSessionRequest {
    /// Path to the session file to import
    #[schemars(description = "File path to the session JSON file to import.")]
    pub file_path: String,

    /// Whether to merge with existing session (default: false, replaces)
    #[schemars(description = "If true, merges with existing datasets instead of replacing.")]
    pub merge: Option<bool>,
}

/// Request to set the global random seed for ML reproducibility.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetSeedRequest {
    /// The random seed value
    #[schemars(description = "The random seed value. Set to null/omit to clear the global seed.")]
    pub seed: Option<u64>,
}

/// Request to get the current global seed.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetSeedRequest {}

/// Request to visualize hierarchical clustering results as a dendrogram.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DendrogramRequest {
    /// Linkage matrix from hierarchical clustering (JSON array of arrays)
    #[schemars(description = "Linkage matrix from hierarchical clustering. Array of [cluster1, cluster2, distance, size] tuples.")]
    pub linkage_matrix: Vec<Vec<f64>>,

    /// Optional labels for leaf nodes
    #[schemars(description = "Optional labels for leaf nodes (original samples). If not provided, uses indices.")]
    pub labels: Option<Vec<String>>,

    /// Chart width
    #[schemars(description = "Width of the chart in pixels (default: 800).")]
    pub width: Option<u32>,

    /// Chart height
    #[schemars(description = "Height of the chart in pixels (default: 600).")]
    pub height: Option<u32>,

    /// Chart title
    #[schemars(description = "Title for the dendrogram (default: 'Dendrogram').")]
    pub title: Option<String>,
}

// ============================================================================
// Data Munging Tool Input Types
// ============================================================================

/// Request to filter rows in a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset to filter.")]
    pub dataset: String,

    /// Column to filter on
    #[schemars(description = "Name of the column to filter on.")]
    pub column: String,

    /// Comparison operator
    #[schemars(description = "Comparison operator: 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'contains', 'starts_with', 'ends_with'.")]
    pub op: String,

    /// Value to compare against
    #[schemars(description = "Value to compare against (as string, will be parsed based on column type).")]
    pub value: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the filtered result. If not provided, overwrites the source dataset.")]
    pub result_name: Option<String>,
}

/// Request to select columns from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SelectColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to select
    #[schemars(description = "List of column names to keep.")]
    pub columns: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result. If not provided, overwrites the source dataset.")]
    pub result_name: Option<String>,
}

/// Request to drop columns from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DropColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to drop
    #[schemars(description = "List of column names to drop.")]
    pub columns: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result. If not provided, overwrites the source dataset.")]
    pub result_name: Option<String>,
}

/// Request to rename columns in a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RenameColumnsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Mapping of old names to new names
    #[schemars(description = "Mapping of old column names to new names as pairs: [[\"old1\", \"new1\"], [\"old2\", \"new2\"]].")]
    pub renames: Vec<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result. If not provided, overwrites the source dataset.")]
    pub result_name: Option<String>,
}

/// Request to sort a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SortDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to sort by
    #[schemars(description = "List of column names to sort by.")]
    pub by: Vec<String>,

    /// Sort in descending order
    #[schemars(description = "If true, sort in descending order. Default is ascending.")]
    pub descending: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result. If not provided, overwrites the source dataset.")]
    pub result_name: Option<String>,
}

/// Request to join two datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct JoinDatasetsRequest {
    /// Name/ID of the left dataset
    #[schemars(description = "Name or ID of the left dataset.")]
    pub left: String,

    /// Name/ID of the right dataset
    #[schemars(description = "Name or ID of the right dataset.")]
    pub right: String,

    /// Columns to join on (from left dataset)
    #[schemars(description = "Column names from the left dataset to join on.")]
    pub left_on: Vec<String>,

    /// Columns to join on (from right dataset)
    #[schemars(description = "Column names from the right dataset to join on. If not provided, uses left_on.")]
    pub right_on: Option<Vec<String>>,

    /// Type of join
    #[schemars(description = "Join type: 'left', 'right', 'inner', or 'full'. Default is 'left'.")]
    pub join_type: Option<String>,

    /// Suffix for duplicate column names
    #[schemars(description = "Suffix to add to duplicate column names from the right dataset. Default is '_right'.")]
    pub suffix: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the joined result.")]
    pub result_name: Option<String>,
}

/// Request to concatenate datasets.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ConcatDatasetsRequest {
    /// Names/IDs of datasets to concatenate
    #[schemars(description = "List of dataset names to concatenate vertically (row-bind).")]
    pub datasets: Vec<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the concatenated result.")]
    pub result_name: Option<String>,
}

/// Request to group and aggregate a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct GroupByRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to group by
    #[schemars(description = "Column names to group by.")]
    pub by: Vec<String>,

    /// Aggregation specifications
    #[schemars(description = "Aggregation specs as [[\"column\", \"function\"], ...]. Functions: 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'first', 'last'.")]
    pub aggs: Vec<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the grouped result.")]
    pub result_name: Option<String>,
}

/// Request to compute value counts for a column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ValueCountsRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to count values in
    #[schemars(description = "Column name to compute value counts for.")]
    pub column: String,

    /// Whether to normalize to percentages
    #[schemars(description = "If true, return percentages instead of counts.")]
    pub normalize: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to pivot a dataset from long to wide format.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PivotDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Index columns (will remain as rows)
    #[schemars(description = "Column names to use as index (will remain as rows).")]
    pub index: Vec<String>,

    /// Column whose values become new column names
    #[schemars(description = "Column whose values become new column names.")]
    pub on: String,

    /// Column containing values to fill the new columns
    #[schemars(description = "Column containing values to fill the new columns.")]
    pub values: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the pivoted result.")]
    pub result_name: Option<String>,
}

/// Request to melt a dataset from wide to long format.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MeltDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// ID columns to keep as-is
    #[schemars(description = "Column names to keep as identifier variables.")]
    pub id_vars: Vec<String>,

    /// Value columns to unpivot
    #[schemars(description = "Column names to unpivot into rows.")]
    pub value_vars: Vec<String>,

    /// Name for the variable column
    #[schemars(description = "Name for the new variable column. Default is 'variable'.")]
    pub variable_name: Option<String>,

    /// Name for the value column
    #[schemars(description = "Name for the new value column. Default is 'value'.")]
    pub value_name: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the melted result.")]
    pub result_name: Option<String>,
}

/// Request to drop rows with null values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DropNaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to check for nulls
    #[schemars(description = "Column names to check for nulls. If not provided, checks all columns.")]
    pub columns: Option<Vec<String>>,

    /// How to drop rows
    #[schemars(description = "How to drop: 'any' (drop if any null) or 'all' (drop only if all null). Default is 'any'.")]
    pub how: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to fill null values.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FillNaRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to fill nulls in
    #[schemars(description = "Column names to fill nulls in. If not provided, fills all columns.")]
    pub columns: Option<Vec<String>>,

    /// Fill strategy
    #[schemars(description = "Fill strategy: 'mean', 'median', 'mode', 'forward', 'backward', or a constant value.")]
    pub strategy: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to remove duplicate rows.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeduplicateRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to check for duplicates
    #[schemars(description = "Column names to check for duplicates. If not provided, checks all columns.")]
    pub columns: Option<Vec<String>>,

    /// Which duplicate to keep
    #[schemars(description = "Which duplicate to keep: 'first', 'last', or 'none'. Default is 'first'.")]
    pub keep: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

// =============================================================================
// STRING CLEANING REQUESTS
// =============================================================================

/// Request to trim whitespace from string columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TrimRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to trim
    #[schemars(description = "Column names to trim. If not provided, trims all string columns.")]
    pub columns: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to convert a string column to lowercase.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToLowercaseRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to convert
    #[schemars(description = "Name of the string column to convert to lowercase.")]
    pub column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to convert a string column to uppercase.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ToUppercaseRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to convert
    #[schemars(description = "Name of the string column to convert to uppercase.")]
    pub column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to replace exact values in a column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReplaceValueRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to modify
    #[schemars(description = "Name of the column to modify.")]
    pub column: String,

    /// Value to find
    #[schemars(description = "Exact value to search for and replace.")]
    pub old_value: String,

    /// Replacement value
    #[schemars(description = "Value to replace matches with.")]
    pub new_value: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to replace substrings matching a regex pattern.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexReplaceRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to modify
    #[schemars(description = "Name of the string column to modify.")]
    pub column: String,

    /// Regex pattern
    #[schemars(description = "Regular expression pattern to match.")]
    pub pattern: String,

    /// Replacement string
    #[schemars(description = "Replacement string. Use $1, $2, etc. for capture groups.")]
    pub replacement: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to extract substrings matching a regex pattern.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexExtractRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to extract from
    #[schemars(description = "Name of the string column to extract from.")]
    pub column: String,

    /// Regex pattern with capture groups
    #[schemars(description = "Regular expression pattern. Use capture groups () to specify what to extract.")]
    pub pattern: String,

    /// Name for the new column
    #[schemars(description = "Name for the new column containing extracted values.")]
    pub new_column: String,

    /// Which capture group to extract
    #[schemars(description = "Which capture group to extract: 0 = entire match, 1 = first group, etc. Default is 1.")]
    pub group: Option<usize>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to count regex matches in each row.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegexCountRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to search in
    #[schemars(description = "Name of the string column to search in.")]
    pub column: String,

    /// Regex pattern
    #[schemars(description = "Regular expression pattern to count matches for.")]
    pub pattern: String,

    /// Name for the new count column
    #[schemars(description = "Name for the new column containing match counts.")]
    pub new_column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to split a string column into multiple columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrSplitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to split
    #[schemars(description = "Name of the string column to split.")]
    pub column: String,

    /// Pattern to split on
    #[schemars(description = "Pattern to split on (regex supported). E.g., ',' or '\\s+'.")]
    pub pattern: String,

    /// Maximum number of splits
    #[schemars(description = "Maximum number of splits. If not provided, splits on all occurrences.")]
    pub max_splits: Option<usize>,

    /// Prefix for new column names
    #[schemars(description = "Prefix for new column names. Creates columns named prefix_0, prefix_1, etc.")]
    pub prefix: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to concatenate multiple string columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrConcatRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to concatenate
    #[schemars(description = "Names of the string columns to concatenate.")]
    pub columns: Vec<String>,

    /// Name for the new column
    #[schemars(description = "Name for the new concatenated column.")]
    pub new_column: String,

    /// Separator between values
    #[schemars(description = "Separator to insert between values. Default is empty string.")]
    pub separator: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to get string lengths.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrLengthRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to measure
    #[schemars(description = "Name of the string column to measure lengths for.")]
    pub column: String,

    /// Name for the new length column
    #[schemars(description = "Name for the new column containing string lengths.")]
    pub new_column: String,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to extract a substring.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StrSubstringRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to extract from
    #[schemars(description = "Name of the string column.")]
    pub column: String,

    /// Start index
    #[schemars(description = "Start index (0-based). Negative values count from end.")]
    pub start: i64,

    /// Length to extract
    #[schemars(description = "Number of characters to extract. If not provided, extracts to end.")]
    pub length: Option<usize>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to create lag or lead columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LagLeadRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to shift
    #[schemars(description = "Column name to create lag/lead for.")]
    pub column: String,

    /// Number of periods to shift
    #[schemars(description = "Number of periods to shift. Positive for lag, negative for lead (or use 'direction').")]
    pub periods: i64,

    /// Direction: 'lag' or 'lead'
    #[schemars(description = "Direction: 'lag' (shift forward) or 'lead' (shift backward). Default is 'lag'.")]
    pub direction: Option<String>,

    /// Columns to group by (for panel data)
    #[schemars(description = "Optional group-by columns for panel data (e.g., ['firm_id']).")]
    pub group_by: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to standardize or normalize columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct StandardizeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Columns to transform
    #[schemars(description = "Column names to standardize/normalize.")]
    pub columns: Vec<String>,

    /// Method: 'standardize' or 'normalize'
    #[schemars(description = "Method: 'standardize' (z-score) or 'normalize' (0-1 range). Default is 'standardize'.")]
    pub method: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to bin a continuous variable.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct BinColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to bin
    #[schemars(description = "Column name to bin.")]
    pub column: String,

    /// Binning strategy
    #[schemars(description = "Binning strategy: 'uniform' (equal width), 'quantile' (equal frequency), or 'custom'.")]
    pub strategy: String,

    /// Number of bins or custom breaks
    #[schemars(description = "Number of bins (for uniform/quantile) or list of break points (for custom).")]
    pub bins: Vec<f64>,

    /// Optional labels for bins
    #[schemars(description = "Optional labels for the bins.")]
    pub labels: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to one-hot encode a categorical column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OneHotEncodeRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to encode
    #[schemars(description = "Categorical column name to one-hot encode.")]
    pub column: String,

    /// Whether to drop the first category
    #[schemars(description = "If true, drop first category to avoid multicollinearity. Default is false.")]
    pub drop_first: Option<bool>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to compute differences or percent changes.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DiffRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Column to compute differences for
    #[schemars(description = "Column name to compute differences for.")]
    pub column: String,

    /// Number of periods
    #[schemars(description = "Number of periods for difference. Default is 1.")]
    pub periods: Option<i64>,

    /// Type of difference
    #[schemars(description = "Type: 'diff' (absolute difference) or 'pct_change' (percent change). Default is 'diff'.")]
    pub diff_type: Option<String>,

    /// Columns to group by (for panel data)
    #[schemars(description = "Optional group-by columns for panel data.")]
    pub group_by: Option<Vec<String>>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to sample rows from a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SampleDatasetRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Number of rows to sample
    #[schemars(description = "Number of rows to sample.")]
    pub n: usize,

    /// Whether to sample with replacement
    #[schemars(description = "If true, sample with replacement. Default is false.")]
    pub replace: Option<bool>,

    /// Random seed for reproducibility
    #[schemars(description = "Random seed for reproducible sampling.")]
    pub seed: Option<u64>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

/// Request to create a new column by computation.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MutateColumnRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of the dataset.")]
    pub dataset: String,

    /// Name for the new column
    #[schemars(description = "Name for the new column.")]
    pub new_column: String,

    /// Expression type
    #[schemars(description = "Expression type: 'arithmetic' (e.g., col1 + col2), 'function' (e.g., log(col)), or 'constant'.")]
    pub expr_type: String,

    /// Left operand (column name for arithmetic)
    #[schemars(description = "Left operand: column name for arithmetic, column for function, or constant value.")]
    pub left: String,

    /// Operator (for arithmetic: '+', '-', '*', '/')
    #[schemars(description = "Operator for arithmetic: '+', '-', '*', '/'. For function: function name ('log', 'exp', 'sqrt', 'abs', 'square').")]
    pub operator: Option<String>,

    /// Right operand (column name for arithmetic)
    #[schemars(description = "Right operand: column name for arithmetic expressions.")]
    pub right: Option<String>,

    /// Optional name for the resulting dataset
    #[schemars(description = "Optional name for the result.")]
    pub result_name: Option<String>,
}

// ============================================================================
// Tool Router Implementation
// ============================================================================

#[tool_router]
impl AnalyticsServer {
    /// Create a new AnalyticsServer instance.
    pub fn new() -> Self {
        Self {
            datasets: Arc::new(RwLock::new(HashMap::new())),
            cleaning_sessions: Arc::new(RwLock::new(HashMap::new())),
            global_seed: Arc::new(RwLock::new(None)),
            tool_router: Self::tool_router(),
        }
    }

    /// Create a new AnalyticsServer with existing dataset storage.
    /// Used for HTTP transport where each session has its own dataset store.
    #[cfg(feature = "http")]
    pub fn with_session(session: &crate::session::Session) -> Self {
        Self {
            datasets: session.datasets.clone(),
            cleaning_sessions: Arc::new(RwLock::new(HashMap::new())),
            global_seed: session.global_seed.clone(),
            tool_router: Self::tool_router(),
        }
    }

    /// List available tools for HTTP API discovery.
    #[cfg(feature = "http")]
    pub fn list_tools(&self) -> Vec<crate::transport::http::ToolDefinition> {
        use crate::transport::http::ToolDefinition;

        // Tool definitions with their descriptions and input schemas
        // This is a static list matching the #[tool] definitions
        vec![
            // Data management tools
            ToolDefinition {
                name: "list_datasets".to_string(),
                description: "List all currently loaded datasets with their basic information (name, dimensions, column types).".to_string(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            },
            ToolDefinition {
                name: "load_dataset".to_string(),
                description: "Load a dataset from a file. Supports CSV, Parquet, Excel, Stata, and SAS formats.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the data file"},
                        "name": {"type": "string", "description": "Optional name for the dataset"}
                    },
                    "required": ["path"]
                }),
            },
            ToolDefinition {
                name: "describe_dataset".to_string(),
                description: "Compute descriptive statistics for all columns in a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "head_dataset".to_string(),
                description: "Show the first N rows of a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "n": {"type": "integer", "description": "Number of rows (default: 5)"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "data_quality_profile".to_string(),
                description: "Generate a comprehensive data quality profile for LLM-assisted data cleaning. Returns column-level statistics (nulls, uniques, types), numeric outlier detection, string pattern analysis, and automated issue detection.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset to profile"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "compute_correlation".to_string(),
                description: "Compute correlation matrix for numeric columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            // Regression tools
            ToolDefinition {
                name: "regression_ols".to_string(),
                description: "Run OLS regression with robust standard errors (HC0-HC3).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string", "description": "Dependent variable"},
                        "x": {"type": "array", "items": {"type": "string"}, "description": "Independent variables"}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            ToolDefinition {
                name: "regression_diagnostics".to_string(),
                description: "Run regression diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            ToolDefinition {
                name: "regression_clustered".to_string(),
                description: "Run OLS with clustered standard errors.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "cluster1": {"type": "string"},
                        "cluster2": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "cluster1"]
                }),
            },
            // Panel econometrics
            ToolDefinition {
                name: "panel_fixed_effects".to_string(),
                description: "Run fixed effects panel regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_col": {"type": "string"},
                        "time_col": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "entity_col"]
                }),
            },
            ToolDefinition {
                name: "panel_random_effects".to_string(),
                description: "Run random effects panel regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_col": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "entity_col"]
                }),
            },
            ToolDefinition {
                name: "panel_hdfe".to_string(),
                description: "Run high-dimensional fixed effects regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "fe": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x", "fe"]
                }),
            },
            ToolDefinition {
                name: "hausman_test".to_string(),
                description: "Perform Hausman test for FE vs RE specification.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}},
                        "entity_col": {"type": "string"}
                    },
                    "required": ["dataset", "y", "x", "entity_col"]
                }),
            },
            // Causal inference
            ToolDefinition {
                name: "iv_2sls".to_string(),
                description: "Run two-stage least squares regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "endogenous": {"type": "array", "items": {"type": "string"}},
                        "instruments": {"type": "array", "items": {"type": "string"}},
                        "exogenous": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "endogenous", "instruments"]
                }),
            },
            ToolDefinition {
                name: "iv_first_stage".to_string(),
                description: "Run first-stage diagnostics for IV regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "endogenous": {"type": "string"},
                        "instruments": {"type": "array", "items": {"type": "string"}},
                        "exogenous": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "endogenous", "instruments"]
                }),
            },
            ToolDefinition {
                name: "diff_in_diff".to_string(),
                description: "Run difference-in-differences analysis.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "treatment_col": {"type": "string"},
                        "time_col": {"type": "string"},
                        "treatment_time": {"type": "number"}
                    },
                    "required": ["dataset", "y", "treatment_col", "time_col", "treatment_time"]
                }),
            },
            // Discrete choice
            ToolDefinition {
                name: "logit".to_string(),
                description: "Run logistic regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            ToolDefinition {
                name: "probit".to_string(),
                description: "Run probit regression.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "y": {"type": "string"},
                        "x": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset", "y", "x"]
                }),
            },
            // Time series
            ToolDefinition {
                name: "ts_var".to_string(),
                description: "Estimate Vector Autoregression model.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "lags": {"type": "integer"}
                    },
                    "required": ["dataset", "columns", "lags"]
                }),
            },
            ToolDefinition {
                name: "ts_arima_fit".to_string(),
                description: "Fit ARIMA model to time series.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "column": {"type": "string"},
                        "p": {"type": "integer"},
                        "d": {"type": "integer"},
                        "q": {"type": "integer"}
                    },
                    "required": ["dataset", "column", "p", "d", "q"]
                }),
            },
            // Machine learning
            ToolDefinition {
                name: "ml_kmeans".to_string(),
                description: "Run K-means clustering.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "k": {"type": "integer"}
                    },
                    "required": ["dataset", "columns", "k"]
                }),
            },
            ToolDefinition {
                name: "ml_pca".to_string(),
                description: "Run Principal Component Analysis.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "n_components": {"type": "integer"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            // Visualization
            ToolDefinition {
                name: "viz_histogram".to_string(),
                description: "Create a histogram plot.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "column": {"type": "string"},
                        "bins": {"type": "integer"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "viz_scatter".to_string(),
                description: "Create a scatter plot.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"}
                    },
                    "required": ["dataset", "x", "y"]
                }),
            },
            ToolDefinition {
                name: "viz_line".to_string(),
                description: "Create a line chart.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "x": {"type": "string"},
                        "y": {"type": "string"}
                    },
                    "required": ["dataset", "x", "y"]
                }),
            },
            ToolDefinition {
                name: "viz_heatmap".to_string(),
                description: "Create a correlation heatmap.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"}
                    },
                    "required": ["dataset"]
                }),
            },
            // Database tools
            ToolDefinition {
                name: "db_sqlite_query".to_string(),
                description: "Execute SQL query on SQLite database.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "query": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["database", "query"]
                }),
            },
            ToolDefinition {
                name: "db_duckdb_query".to_string(),
                description: "Execute SQL query on DuckDB database.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "query": {"type": "string"},
                        "name": {"type": "string"}
                    },
                    "required": ["database", "query"]
                }),
            },
            ToolDefinition {
                name: "db_query_file".to_string(),
                description: "Execute SQL query directly on a Parquet or CSV file using DuckDB. Use {file} as placeholder for the file path.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the Parquet or CSV file"},
                        "query": {"type": "string", "description": "SQL query with {file} placeholder for the file path"},
                        "name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["file_path", "query"]
                }),
            },
            // Data munging tools
            ToolDefinition {
                name: "munge_filter".to_string(),
                description: "Filter rows in a dataset based on a condition. Supports operators: eq, ne, gt, ge, lt, le, contains, startswith, endswith.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset to filter"},
                        "column": {"type": "string", "description": "Column to filter on"},
                        "operator": {"type": "string", "description": "Comparison operator (eq, ne, gt, ge, lt, le, contains, startswith, endswith)"},
                        "value": {"type": "string", "description": "Value to compare against"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column", "operator", "value"]
                }),
            },
            ToolDefinition {
                name: "munge_select".to_string(),
                description: "Select specific columns from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to select"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_drop_columns".to_string(),
                description: "Drop columns from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to drop"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_rename".to_string(),
                description: "Rename columns in a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "renames": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Map of old names to new names"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "renames"]
                }),
            },
            ToolDefinition {
                name: "munge_sort".to_string(),
                description: "Sort a dataset by one or more columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to sort by"},
                        "descending": {"type": "boolean", "description": "Sort in descending order (default: false)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_mutate".to_string(),
                description: "Create a new column or modify an existing one using an expression. Supports arithmetic operations on columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "new_column": {"type": "string", "description": "Name of the new column"},
                        "expression": {"type": "string", "description": "Expression type: 'copy', 'constant', 'add', 'subtract', 'multiply', 'divide'"},
                        "left": {"type": "string", "description": "Left operand (column name or constant value)"},
                        "right": {"type": "string", "description": "Right operand (column name, for arithmetic operations)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "new_column", "expression", "left"]
                }),
            },
            ToolDefinition {
                name: "munge_sample".to_string(),
                description: "Take a random sample of rows from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "n": {"type": "integer", "description": "Number of rows to sample"},
                        "with_replacement": {"type": "boolean", "description": "Sample with replacement (default: false)"},
                        "seed": {"type": "integer", "description": "Optional random seed for reproducibility"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "n"]
                }),
            },
            ToolDefinition {
                name: "munge_join".to_string(),
                description: "Join two datasets on key columns. Supports left, right, inner, and full outer joins.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "left_dataset": {"type": "string", "description": "Name of the left dataset"},
                        "right_dataset": {"type": "string", "description": "Name of the right dataset"},
                        "on": {"type": "array", "items": {"type": "string"}, "description": "Columns to join on"},
                        "right_on": {"type": "array", "items": {"type": "string"}, "description": "Right key columns if different from left"},
                        "join_type": {"type": "string", "description": "Join type: left, right, inner, full (default: left)"},
                        "suffix": {"type": "string", "description": "Suffix for duplicate column names (default: _right)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["left_dataset", "right_dataset", "on"]
                }),
            },
            ToolDefinition {
                name: "munge_concat".to_string(),
                description: "Concatenate multiple datasets vertically (row-bind).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "datasets": {"type": "array", "items": {"type": "string"}, "description": "Names of datasets to concatenate"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["datasets"]
                }),
            },
            ToolDefinition {
                name: "munge_group_by".to_string(),
                description: "Group dataset by columns and compute aggregations (sum, mean, count, min, max, std, var, first, last, median).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "by": {"type": "array", "items": {"type": "string"}, "description": "Columns to group by"},
                        "aggs": {"type": "array", "items": {"type": "object"}, "description": "Aggregation specs: [{column, function}]"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "by", "aggs"]
                }),
            },
            ToolDefinition {
                name: "munge_value_counts".to_string(),
                description: "Count occurrences of unique values in a column.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to count values in"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "munge_pivot".to_string(),
                description: "Pivot a dataset from long to wide format.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "index": {"type": "array", "items": {"type": "string"}, "description": "Columns to use as index (row identifiers)"},
                        "on": {"type": "string", "description": "Column whose values become new column names"},
                        "values": {"type": "string", "description": "Column containing values to fill the new columns"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "index", "on", "values"]
                }),
            },
            ToolDefinition {
                name: "munge_melt".to_string(),
                description: "Melt a dataset from wide to long format (unpivot).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "id_vars": {"type": "array", "items": {"type": "string"}, "description": "Columns to keep as identifiers"},
                        "value_vars": {"type": "array", "items": {"type": "string"}, "description": "Columns to unpivot into rows"},
                        "variable_name": {"type": "string", "description": "Name for the variable column (default: variable)"},
                        "value_name": {"type": "string", "description": "Name for the value column (default: value)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "id_vars", "value_vars"]
                }),
            },
            ToolDefinition {
                name: "munge_drop_na".to_string(),
                description: "Drop rows with missing values.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to check for NA (all if not specified)"},
                        "how": {"type": "string", "description": "How to drop: 'any' or 'all' (default: any)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "munge_fill_na".to_string(),
                description: "Fill missing values using a strategy (mean, median, constant, forward, backward, zero).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to fill (all if not specified)"},
                        "strategy": {"type": "string", "description": "Fill strategy: mean, median, constant, forward, backward, zero"},
                        "constant_value": {"type": "number", "description": "Value to use for constant strategy"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "strategy"]
                }),
            },
            ToolDefinition {
                name: "munge_deduplicate".to_string(),
                description: "Remove duplicate rows from a dataset.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "subset": {"type": "array", "items": {"type": "string"}, "description": "Columns to consider for duplicates (all if not specified)"},
                        "keep": {"type": "string", "description": "Which duplicate to keep: 'first', 'last', or 'none' (default: first)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset"]
                }),
            },
            ToolDefinition {
                name: "munge_lag_lead".to_string(),
                description: "Create lag or lead of a column (shift values forward or backward).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to lag or lead"},
                        "periods": {"type": "integer", "description": "Number of periods to shift"},
                        "operation": {"type": "string", "description": "Operation: 'lag' or 'lead'"},
                        "group_by": {"type": "array", "items": {"type": "string"}, "description": "Optional group by columns for panel data"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column", "periods", "operation"]
                }),
            },
            ToolDefinition {
                name: "munge_diff".to_string(),
                description: "Compute difference or percentage change of a column.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to difference"},
                        "periods": {"type": "integer", "description": "Number of periods for differencing (default: 1)"},
                        "pct_change": {"type": "boolean", "description": "Compute percentage change instead of difference (default: false)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
            ToolDefinition {
                name: "munge_standardize".to_string(),
                description: "Standardize (z-score) or normalize (0-1) columns.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "columns": {"type": "array", "items": {"type": "string"}, "description": "Columns to standardize"},
                        "method": {"type": "string", "description": "Method: 'standardize' (z-score) or 'normalize' (0-1)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "columns"]
                }),
            },
            ToolDefinition {
                name: "munge_bin".to_string(),
                description: "Bin a continuous column into discrete categories.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to bin"},
                        "strategy": {"type": "string", "description": "Binning strategy: 'equal_width', 'quantile', or 'custom'"},
                        "n_bins": {"type": "integer", "description": "Number of bins for equal_width or quantile strategies"},
                        "breaks": {"type": "array", "items": {"type": "number"}, "description": "Custom bin edges for custom strategy"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column", "strategy"]
                }),
            },
            ToolDefinition {
                name: "munge_one_hot_encode".to_string(),
                description: "One-hot encode a categorical column into dummy variables.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Name of the dataset"},
                        "column": {"type": "string", "description": "Column to one-hot encode"},
                        "drop_first": {"type": "boolean", "description": "Drop first category to avoid multicollinearity (default: false)"},
                        "output_name": {"type": "string", "description": "Optional name for the resulting dataset"}
                    },
                    "required": ["dataset", "column"]
                }),
            },
        ]
    }

    /// Call a tool by name with session context (for HTTP transport).
    /// This creates a session-scoped server and dispatches the tool call.
    #[cfg(feature = "http")]
    pub async fn call_tool_with_session(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        session: &crate::session::Session,
    ) -> Result<crate::transport::http::ToolResult, String> {
        use crate::transport::http::{ContentItem, ToolResult};

        // Create a session-scoped server instance that shares the session's datasets
        let session_server = Self::with_session(session);

        // Helper to convert CallToolResult to our ToolResult
        fn convert_result(call_result: CallToolResult) -> ToolResult {
            let is_error = call_result.is_error.unwrap_or(false);
            let content: Vec<ContentItem> = call_result
                .content
                .into_iter()
                .filter_map(|c| {
                    // Content in rmcp is an Annotated<RawContent>
                    // We need to access the inner raw content
                    match &c.raw {
                        RawContent::Text(text_content) => Some(ContentItem::Text {
                            text: text_content.text.clone(),
                        }),
                        RawContent::Image(img) => Some(ContentItem::Image {
                            data: img.data.clone(),
                            mime_type: img.mime_type.clone(),
                        }),
                        _ => None,
                    }
                })
                .collect();

            let error = if is_error {
                content.first().and_then(|c| {
                    if let ContentItem::Text { text } = c {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            } else {
                None
            };

            ToolResult {
                success: !is_error,
                content,
                error,
            }
        }

        // Parse arguments and dispatch to the appropriate tool method
        // For now, we support a subset of the most commonly used tools
        let result = match tool_name {
            "list_datasets" => session_server.list_datasets().await,
            "load_dataset" => {
                let req: LoadDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .load_dataset(Parameters(req))
                    .await
            }
            "describe_dataset" => {
                let req: DescribeDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .describe_dataset(Parameters(req))
                    .await
            }
            "head_dataset" => {
                let req: HeadDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .head_dataset(Parameters(req))
                    .await
            }
            "data_quality_profile" => {
                let req: DataQualityProfileRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .data_quality_profile(Parameters(req))
                    .await
            }
            "preview_cleaning" => {
                let req: PreviewCleaningRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .preview_cleaning(Parameters(req))
                    .await
            }
            "verify_cleaning" => {
                let req: VerifyCleaningRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .verify_cleaning(Parameters(req))
                    .await
            }
            "compute_correlation" => {
                let req: CorrelationRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .compute_correlation(Parameters(req))
                    .await
            }
            "regression_ols" => {
                let req: OlsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .regression_ols(Parameters(req))
                    .await
            }
            "regression_diagnostics" => {
                let req: DiagnosticsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .regression_diagnostics(Parameters(req))
                    .await
            }
            "panel_fixed_effects" => {
                let req: PanelFERequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .panel_fixed_effects(Parameters(req))
                    .await
            }
            "panel_random_effects" => {
                let req: PanelRERequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .panel_random_effects(Parameters(req))
                    .await
            }
            "iv_2sls" => {
                let req: IV2SLSRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .iv_2sls(Parameters(req))
                    .await
            }
            "diff_in_diff" => {
                let req: DiDRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .diff_in_diff(Parameters(req))
                    .await
            }
            "treatment_ipw" => {
                let req: IpwRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .treatment_ipw(Parameters(req))
                    .await
            }
            "treatment_doubly_robust" => {
                let req: DoublyRobustRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .treatment_doubly_robust(Parameters(req))
                    .await
            }
            "mediation_analysis" => {
                let req: MediationRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .mediation_analysis(Parameters(req))
                    .await
            }
            "logit" => {
                let req: LogitRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .logit(Parameters(req))
                    .await
            }
            "probit" => {
                let req: ProbitRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .probit(Parameters(req))
                    .await
            }
            "ml_kmeans" => {
                let req: KMeansRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .ml_kmeans(Parameters(req))
                    .await
            }
            "ml_pca" => {
                let req: PCARequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .ml_pca(Parameters(req))
                    .await
            }
            "viz_histogram" => {
                let req: HistogramRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .viz_histogram(Parameters(req))
                    .await
            }
            "viz_scatter" => {
                let req: ScatterPlotRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .viz_scatter(Parameters(req))
                    .await
            }
            "viz_line" => {
                let req: LineChartRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .viz_line(Parameters(req))
                    .await
            }
            "viz_heatmap" => {
                let req: HeatmapRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .viz_heatmap(Parameters(req))
                    .await
            }
            "db_sqlite_query" => {
                let req: SqliteQueryRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .db_sqlite_query(Parameters(req))
                    .await
            }
            "db_duckdb_query" => {
                let req: DuckDBQueryRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .db_duckdb_query(Parameters(req))
                    .await
            }
            "db_query_file" => {
                let req: DuckDBFileQueryRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .db_query_file(Parameters(req))
                    .await
            }
            // Data munging tools
            "munge_filter" => {
                let req: FilterDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_filter(Parameters(req))
                    .await
            }
            "munge_select" => {
                let req: SelectColumnsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_select(Parameters(req))
                    .await
            }
            "munge_drop_columns" => {
                let req: DropColumnsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_drop_columns(Parameters(req))
                    .await
            }
            "munge_rename" => {
                let req: RenameColumnsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_rename(Parameters(req))
                    .await
            }
            "munge_sort" => {
                let req: SortDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_sort(Parameters(req))
                    .await
            }
            "munge_mutate" => {
                let req: MutateColumnRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_mutate(Parameters(req))
                    .await
            }
            "munge_sample" => {
                let req: SampleDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_sample(Parameters(req))
                    .await
            }
            "munge_join" => {
                let req: JoinDatasetsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_join(Parameters(req))
                    .await
            }
            "munge_concat" => {
                let req: ConcatDatasetsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_concat(Parameters(req))
                    .await
            }
            "munge_group_by" => {
                let req: GroupByRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_group_by(Parameters(req))
                    .await
            }
            "munge_value_counts" => {
                let req: ValueCountsRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_value_counts(Parameters(req))
                    .await
            }
            "munge_pivot" => {
                let req: PivotDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_pivot(Parameters(req))
                    .await
            }
            "munge_melt" => {
                let req: MeltDatasetRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_melt(Parameters(req))
                    .await
            }
            "munge_drop_na" => {
                let req: DropNaRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_drop_na(Parameters(req))
                    .await
            }
            "munge_fill_na" => {
                let req: FillNaRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_fill_na(Parameters(req))
                    .await
            }
            "munge_deduplicate" => {
                let req: DeduplicateRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_deduplicate(Parameters(req))
                    .await
            }
            "munge_lag_lead" => {
                let req: LagLeadRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_lag_lead(Parameters(req))
                    .await
            }
            "munge_diff" => {
                let req: DiffRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_diff(Parameters(req))
                    .await
            }
            "munge_standardize" => {
                let req: StandardizeRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_standardize(Parameters(req))
                    .await
            }
            "munge_bin" => {
                let req: BinColumnRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_bin(Parameters(req))
                    .await
            }
            "munge_one_hot_encode" => {
                let req: OneHotEncodeRequest = serde_json::from_value(arguments)
                    .map_err(|e| format!("Invalid arguments: {}", e))?;
                session_server
                    .munge_one_hot_encode(Parameters(req))
                    .await
            }
            _ => {
                return Err(format!("Unknown tool: {}", tool_name));
            }
        };

        match result {
            Ok(call_result) => Ok(convert_result(call_result)),
            Err(e) => Err(format!("Tool execution failed: {:?}", e)),
        }
    }

    /// List all currently loaded datasets.
    #[tool(description = "List all currently loaded datasets with their basic information (name, dimensions, column types).")]
    async fn list_datasets(&self) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        if datasets.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No datasets currently loaded. Use the 'load_dataset' tool to load a data file.",
            )]));
        }

        let mut result = String::from("Loaded Datasets:\n\n");
        for (id, dataset) in datasets.iter() {
            let info: DatasetInfo = dataset.into();
            result.push_str(&format!(
                "- **{}**: {} rows x {} columns\n",
                id, info.nrows, info.ncols
            ));
            result.push_str("  Columns: ");
            let col_summary: Vec<String> = info
                .columns
                .iter()
                .map(|c| format!("{} ({})", c.name, c.dtype))
                .collect();
            result.push_str(&col_summary.join(", "));
            result.push_str("\n\n");
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Load a dataset from a file.
    #[tool(description = "Load a dataset from a file. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats. Returns dataset information including dimensions and column types.")]
    async fn load_dataset(
        &self,
        Parameters(request): Parameters<LoadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let path = PathBuf::from(&request.path);

        // Load the dataset
        let dataset = match DataLoader::load(&path) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to load dataset: {}",
                    e
                ))]));
            }
        };

        // Generate an ID for the dataset
        let id = request.name.unwrap_or_else(|| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset")
                .to_string()
        });

        // Get info before moving
        let info: DatasetInfo = (&dataset).into();

        // Store the dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(id.clone(), dataset);

        let result = format!(
            "Successfully loaded dataset '{}'\n\n\
             Dimensions: {} rows x {} columns\n\n\
             Columns:\n{}",
            id,
            info.nrows,
            info.ncols,
            info.columns
                .iter()
                .map(|c| format!(
                    "  - {} ({}): {} nulls ({:.1}%)",
                    c.name,
                    c.dtype,
                    c.null_count,
                    if info.nrows > 0 {
                        (c.null_count as f64 / info.nrows as f64) * 100.0
                    } else {
                        0.0
                    }
                ))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Get summary statistics for a dataset.
    #[tool(description = "Compute descriptive statistics for all columns in a dataset. Returns count, mean, std, min, quartiles, max for numeric columns; unique count for categorical columns.")]
    async fn describe_dataset(
        &self,
        Parameters(request): Parameters<DescribeDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let stats = match DescriptiveStats::compute(dataset) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to compute statistics: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(stats.to_string())]))
    }

    /// Preview the first rows of a dataset.
    #[tool(description = "Show the first N rows of a dataset. Default is 5 rows.")]
    async fn head_dataset(
        &self,
        Parameters(request): Parameters<HeadDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let n = request.n.unwrap_or(5);
        let head_df = dataset.head(Some(n));

        Ok(CallToolResult::success(vec![Content::text(format!(
            "First {} rows of '{}':\n\n{}",
            n, request.dataset, head_df
        ))]))
    }

    /// Generate a comprehensive data quality profile.
    #[tool(description = "Generate a comprehensive data quality profile for LLM-assisted data cleaning. Returns column-level statistics (nulls, uniques, types), numeric outlier detection, string pattern analysis, and automated issue detection with severity ratings.")]
    async fn data_quality_profile(
        &self,
        Parameters(request): Parameters<DataQualityProfileRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let profile = generate_quality_profile(dataset);

        // Format the profile for LLM consumption
        let mut result = profile.summary();

        // Add detailed column information
        result.push_str("\n\nColumn Details:\n");
        result.push_str("===============\n");

        for col in &profile.columns {
            result.push_str(&format!("\n## {} ({})\n", col.name, col.dtype));
            result.push_str(&format!("  - Null: {} ({:.1}%)\n", col.null_count, col.null_pct * 100.0));
            result.push_str(&format!("  - Unique: {} ({:.1}%)\n", col.unique_count, col.unique_pct * 100.0));

            if let Some(ref stats) = col.numeric_stats {
                result.push_str(&format!("  - Range: {:.2} to {:.2}\n", stats.min, stats.max));
                result.push_str(&format!("  - Mean: {:.2}, Median: {:.2}, Std: {:.2}\n",
                    stats.mean, stats.median, stats.std));
                if stats.outlier_count > 0 {
                    result.push_str(&format!("  - Outliers: {} (outside {:.2} to {:.2})\n",
                        stats.outlier_count, stats.outlier_lower_bound, stats.outlier_upper_bound));
                }
            }

            if let Some(ref stats) = col.string_stats {
                result.push_str(&format!("  - Length: {} to {} chars (avg {:.1})\n",
                    stats.min_length, stats.max_length, stats.mean_length));
                if !stats.detected_patterns.is_empty() {
                    result.push_str(&format!("  - Patterns: {}\n", stats.detected_patterns.join(", ")));
                }
                if !stats.top_values.is_empty() {
                    let top_3: Vec<String> = stats.top_values.iter()
                        .take(3)
                        .map(|(v, c)| format!("'{}' ({})", v, c))
                        .collect();
                    result.push_str(&format!("  - Top values: {}\n", top_3.join(", ")));
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Preview what a cleaning operation would do before applying it.
    #[tool(description = "Preview a data cleaning operation before applying it. Shows how many rows would be affected, sample changes, and warnings. Supports: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter.")]
    async fn preview_cleaning(
        &self,
        Parameters(request): Parameters<PreviewCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Build the CleaningOperation from request parameters
        let operation = match request.operation.to_lowercase().as_str() {
            "trim" => CleaningOperation::Trim { columns: request.columns },
            "lowercase" | "to_lowercase" => {
                let column = request.columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("lowercase operation requires a column", None))?
                    .clone();
                CleaningOperation::ToLowercase { column }
            }
            "uppercase" | "to_uppercase" => {
                let column = request.columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("uppercase operation requires a column", None))?
                    .clone();
                CleaningOperation::ToUppercase { column }
            }
            "fill_na" | "fillna" => {
                CleaningOperation::FillNa {
                    columns: request.columns,
                    strategy: request.strategy.unwrap_or_else(|| "constant".to_string()),
                    value: request.value,
                }
            }
            "drop_na" | "dropna" => {
                CleaningOperation::DropNa {
                    columns: request.columns,
                    how: request.how.unwrap_or_else(|| "any".to_string()),
                }
            }
            "deduplicate" | "dedup" => {
                CleaningOperation::Deduplicate {
                    columns: request.columns,
                    keep: request.keep.unwrap_or_else(|| "first".to_string()),
                }
            }
            "replace" => {
                let column = request.columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("replace operation requires a column", None))?
                    .clone();
                let old_value = request.old_value
                    .ok_or_else(|| McpError::invalid_request("replace operation requires old_value", None))?;
                let new_value = request.value
                    .ok_or_else(|| McpError::invalid_request("replace operation requires value (new value)", None))?;
                CleaningOperation::Replace { column, old_value, new_value }
            }
            "filter" => {
                let column = request.columns
                    .as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("filter operation requires a column", None))?
                    .clone();
                let operator = request.operator
                    .ok_or_else(|| McpError::invalid_request("filter operation requires operator", None))?;
                let value = request.filter_value
                    .ok_or_else(|| McpError::invalid_request("filter operation requires filter_value", None))?;
                CleaningOperation::Filter { column, operator, value }
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown operation '{}'. Supported: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter",
                    request.operation
                ))]));
            }
        };

        let sample_size = request.sample_size.unwrap_or(5);
        let preview = preview_cleaning(dataset, &operation, sample_size);

        Ok(CallToolResult::success(vec![Content::text(preview.summary())]))
    }

    /// Verify a cleaning operation by comparing before and after datasets.
    #[tool(description = "Verify a cleaning operation by comparing the original and cleaned datasets. Returns a detailed report with row counts, quality delta (completeness change, issues resolved/introduced), and sample changes.")]
    async fn verify_cleaning(
        &self,
        Parameters(request): Parameters<VerifyCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let before = match datasets.get(&request.before_dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.before_dataset
                ))]));
            }
        };

        let after = match datasets.get(&request.after_dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.after_dataset
                ))]));
            }
        };

        let report = verify_cleaning(before, after, &request.operation_description);

        Ok(CallToolResult::success(vec![Content::text(report.summary())]))
    }

    // ========================================================================
    // Cleaning Session Management Tools
    // ========================================================================

    /// Start a new cleaning session for a dataset.
    #[tool(description = "Start a new cleaning session for a dataset. Returns a session ID that can be used to track progress, apply operations, and rollback changes. Each session maintains checkpoints for undo capability.")]
    async fn cleaning_session_start(
        &self,
        Parameters(request): Parameters<CleaningSessionStartRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds.clone(),
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };
        drop(datasets);

        let session_name = request.session_name.unwrap_or_else(|| request.dataset.clone());
        let session = CleaningSession::new(dataset, &session_name);
        let session_id = session.id.clone();
        let status = session.status();

        let mut sessions = self.cleaning_sessions.write().await;
        sessions.insert(session_id.clone(), session);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Cleaning session started!\n\n\
             Session ID: {}\n\
             Dataset: {}\n\
             Rows: {}\n\
             Completeness: {:.1}%\n\n\
             Use 'cleaning_session_apply' to apply cleaning operations.\n\
             Use 'cleaning_session_status' to check progress.\n\
             Use 'cleaning_rollback' to undo operations.",
            session_id, session_name, status.current_row_count, status.current_completeness * 100.0
        ))]))
    }

    /// Get the status of a cleaning session.
    #[tool(description = "Get the current status of a cleaning session, including checkpoint count, operations performed, current row count, and completeness score.")]
    async fn cleaning_session_status(
        &self,
        Parameters(request): Parameters<CleaningSessionStatusRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sessions = self.cleaning_sessions.read().await;

        let session = match sessions.get(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(session.summary())]))
    }

    /// List all active cleaning sessions.
    #[tool(description = "List all active cleaning sessions with their current status.")]
    async fn list_cleaning_sessions(
        &self,
        Parameters(_request): Parameters<ListCleaningSessionsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sessions = self.cleaning_sessions.read().await;

        if sessions.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No active cleaning sessions.\n\nUse 'cleaning_session_start' to start a new session."
            )]));
        }

        let mut result = String::from("Active Cleaning Sessions\n");
        result.push_str("========================\n\n");

        for (id, session) in sessions.iter() {
            let status = session.status();
            result.push_str(&format!(
                "Session: {} ({})\n  - Checkpoints: {}\n  - Operations: {}\n  - Rows: {}\n  - Completeness: {:.1}%\n\n",
                id, status.dataset_name, status.total_checkpoints, status.total_operations,
                status.current_row_count, status.current_completeness * 100.0
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Apply a cleaning operation within a session.
    #[tool(description = "Apply a cleaning operation within a session. Creates a new checkpoint automatically. Returns a verification report showing what changed.")]
    async fn cleaning_session_apply(
        &self,
        Parameters(request): Parameters<CleaningSessionApplyRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut sessions = self.cleaning_sessions.write().await;

        let session = match sessions.get_mut(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        let operation_type = request.operation.to_lowercase();
        let description = format!("{} operation", operation_type);
        let params = std::collections::HashMap::new();

        // Helper to convert Vec<String> to Vec<&str>
        fn to_str_slice(v: &Option<Vec<String>>) -> Option<Vec<&str>> {
            v.as_ref().map(|cols| cols.iter().map(|s| s.as_str()).collect())
        }

        // Apply the operation based on type
        let result = match operation_type.as_str() {
            "trim" => {
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                session.apply_operation(&operation_type, &description, params, move |ds| {
                    trim(ds, cols_ref.as_deref()).map_err(|e| e.to_string())
                })
            }
            "lowercase" | "to_lowercase" => {
                let column = request.columns.as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("lowercase requires a column", None))?
                    .clone();
                session.apply_operation(&operation_type, &format!("lowercase {}", column), params, |ds| {
                    to_lowercase(ds, &column).map_err(|e| e.to_string())
                })
            }
            "uppercase" | "to_uppercase" => {
                let column = request.columns.as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("uppercase requires a column", None))?
                    .clone();
                session.apply_operation(&operation_type, &format!("uppercase {}", column), params, |ds| {
                    to_uppercase(ds, &column).map_err(|e| e.to_string())
                })
            }
            "fill_na" | "fillna" => {
                let strategy_str = request.strategy.as_deref().unwrap_or("constant");
                let value = request.value.clone();
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                let strategy = match strategy_str {
                    "mean" => FillStrategy::Mean,
                    "median" => FillStrategy::Median,
                    "forward" => FillStrategy::Forward,
                    "backward" => FillStrategy::Backward,
                    "constant" | _ => FillStrategy::Constant(value.unwrap_or_else(|| "0".to_string())),
                };
                session.apply_operation(&operation_type, &format!("fill_na with {:?}", strategy), params, move |ds| {
                    fill_na(ds, cols_ref.as_deref(), strategy.clone()).map_err(|e| e.to_string())
                })
            }
            "drop_na" | "dropna" => {
                let how = request.how.as_deref().unwrap_or("any").to_string();
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                session.apply_operation(&operation_type, &format!("drop_na ({})", how), params, move |ds| {
                    drop_na(ds, cols_ref.as_deref(), &how).map_err(|e| e.to_string())
                })
            }
            "deduplicate" | "dedup" => {
                let keep = request.keep.as_deref().unwrap_or("first").to_string();
                let cols = request.columns.clone();
                let cols_ref = to_str_slice(&cols);
                session.apply_operation(&operation_type, &format!("deduplicate (keep={})", keep), params, move |ds| {
                    deduplicate(ds, cols_ref.as_deref(), &keep).map_err(|e| e.to_string())
                })
            }
            "replace" => {
                let column = request.columns.as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("replace requires a column", None))?
                    .clone();
                let old_value = request.old_value.clone()
                    .ok_or_else(|| McpError::invalid_request("replace requires old_value", None))?;
                let new_value = request.value.clone()
                    .ok_or_else(|| McpError::invalid_request("replace requires value", None))?;
                session.apply_operation(&operation_type, &format!("replace '{}' with '{}' in {}", old_value, new_value, column), params, |ds| {
                    replace(ds, &column, &old_value, &new_value).map_err(|e| e.to_string())
                })
            }
            "filter" => {
                let column = request.columns.as_ref()
                    .and_then(|c| c.first())
                    .ok_or_else(|| McpError::invalid_request("filter requires a column", None))?
                    .clone();
                let operator = request.operator.clone()
                    .ok_or_else(|| McpError::invalid_request("filter requires operator", None))?;
                let value = request.filter_value.clone()
                    .ok_or_else(|| McpError::invalid_request("filter requires filter_value", None))?;
                session.apply_operation(&operation_type, &format!("filter {} {} {}", column, operator, value), params, |ds| {
                    filter(ds, &column, &operator, &value).map_err(|e| e.to_string())
                })
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown operation '{}'. Supported: trim, lowercase, uppercase, fill_na, drop_na, deduplicate, replace, filter",
                    request.operation
                ))]));
            }
        };

        match result {
            Ok(report) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Operation applied successfully!\n\n{}",
                report.summary()
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Operation failed: {}",
                e
            ))])),
        }
    }

    /// Rollback a cleaning session to a previous checkpoint.
    #[tool(description = "Rollback a cleaning session to a previous checkpoint. If no checkpoint index is provided, rolls back to the previous checkpoint (undo last operation).")]
    async fn cleaning_rollback(
        &self,
        Parameters(request): Parameters<CleaningRollbackRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut sessions = self.cleaning_sessions.write().await;

        let session = match sessions.get_mut(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        let result = match request.checkpoint_index {
            Some(index) => session.rollback_to(index),
            None => session.rollback(),
        };

        match result {
            Ok(()) => {
                let status = session.status();
                Ok(CallToolResult::success(vec![Content::text(format!(
                    "Rollback successful!\n\n\
                     Current checkpoint: {}\n\
                     Rows: {}\n\
                     Completeness: {:.1}%",
                    status.current_checkpoint, status.current_row_count, status.current_completeness * 100.0
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Rollback failed: {}",
                e
            ))])),
        }
    }

    /// List all checkpoints in a cleaning session.
    #[tool(description = "List all checkpoints in a cleaning session, showing the state at each point.")]
    async fn cleaning_session_checkpoints(
        &self,
        Parameters(request): Parameters<CleaningSessionCheckpointsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let sessions = self.cleaning_sessions.read().await;

        let session = match sessions.get(&request.session_id) {
            Some(s) => s,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Session '{}' not found. Use 'list_cleaning_sessions' to see active sessions.",
                    request.session_id
                ))]));
            }
        };

        let checkpoints = session.list_checkpoints();
        let mut result = String::from("Session Checkpoints\n");
        result.push_str("===================\n\n");

        for cp in checkpoints {
            let marker = if cp.is_current { " <-- current" } else { "" };
            result.push_str(&format!(
                "#{}: {}{}\n  - Rows: {}\n  - Completeness: {:.1}%\n  - Created: {}\n\n",
                cp.index, cp.description, marker, cp.row_count, cp.completeness * 100.0,
                cp.created_at.format("%H:%M:%S")
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Generate smart cleaning suggestions for a dataset.
    #[tool(description = "Analyze a dataset and generate prioritized cleaning suggestions. Returns specific operations with parameters, estimated impact, and reasoning. Use this to get intelligent recommendations before starting a cleaning workflow.")]
    async fn suggest_cleaning(
        &self,
        Parameters(request): Parameters<SuggestCleaningRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'load_dataset' first.",
                    request.dataset
                ))]));
            }
        };

        // Generate quality profile and suggestions
        let profile = generate_quality_profile(dataset);
        let report = generate_suggestions(&profile);

        // Parse minimum priority filter
        let min_priority = request.min_priority.as_ref().and_then(|p| {
            match p.to_lowercase().as_str() {
                "low" => Some(SuggestionPriority::Low),
                "medium" => Some(SuggestionPriority::Medium),
                "high" => Some(SuggestionPriority::High),
                "critical" => Some(SuggestionPriority::Critical),
                _ => None,
            }
        });

        // Filter suggestions
        let mut suggestions: Vec<_> = report.suggestions.iter()
            .filter(|s| min_priority.map_or(true, |min| s.priority >= min))
            .collect();

        // Apply limit if specified
        if let Some(limit) = request.limit {
            suggestions.truncate(limit);
        }

        // Build result
        let mut result = format!(
            "Cleaning Suggestions for '{}'\n\
             ================================\n\n\
             Dataset: {} rows x {} columns\n\
             Completeness: {:.1}%\n\
             Issues found: {}\n\
             Suggestions: {}\n\n",
            request.dataset,
            report.dataset_summary.row_count,
            report.dataset_summary.column_count,
            report.dataset_summary.completeness_score * 100.0,
            report.issues_analyzed,
            suggestions.len()
        );

        if suggestions.is_empty() {
            result.push_str("No cleaning suggestions - your data looks clean!\n");
        } else {
            for (i, s) in suggestions.iter().enumerate() {
                result.push_str(&format!(
                    "{}. [{}] {}\n\
                     ----------------------------------------\n\
                     Category: {:?}\n\
                     Issue: {}\n\
                     \n\
                     Description: {}\n\
                     \n\
                     Reasoning: {}\n\
                     \n\
                     Impact: {}\n\
                     \n\
                     Operation: '{}'\n\
                     Parameters:\n\
                     - column: {}\n\
                     - value: {}\n\
                     - strategy: {}\n\
                     \n\
                     Considerations:\n",
                    i + 1,
                    s.priority.label(),
                    s.title,
                    s.category,
                    s.addresses_issue,
                    s.description,
                    s.reasoning,
                    s.estimated_impact.impact_description,
                    s.operation,
                    s.parameters.column.as_deref().unwrap_or("-"),
                    s.parameters.value.as_deref().unwrap_or("-"),
                    s.parameters.strategy.as_deref().unwrap_or("-"),
                ));

                for consideration in &s.considerations {
                    result.push_str(&format!("  - {}\n", consideration));
                }
                result.push_str("\n");
            }

            // Add overall recommendation
            result.push_str(&format!("\nRecommendation\n--------------\n{}\n", report.overall_recommendation));
        }

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    /// Compute correlation matrix for numeric columns.
    #[tool(description = "Compute the Pearson correlation matrix for all numeric columns in a dataset.")]
    async fn compute_correlation(
        &self,
        Parameters(request): Parameters<CorrelationRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let corr = match correlation_matrix(dataset) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to compute correlation matrix: {}",
                    e
                ))]));
            }
        };

        if corr.columns.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No numeric columns found in dataset.",
            )]));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Correlation Matrix for '{}':\n\n{}",
            request.dataset,
            corr.to_string_table()
        ))]))
    }

    /// Run OLS regression.
    #[tool(description = "Run Ordinary Least Squares (OLS) regression. Returns coefficients, standard errors, t-values, p-values, R-squared, and F-statistic.")]
    async fn regression_ols(
        &self,
        Parameters(request): Parameters<OlsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_ols(dataset, &request.y, &x_refs, true, CovarianceType::HC1) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regression failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run regression diagnostics.
    #[tool(description = "Run comprehensive regression diagnostics. Tests include: Jarque-Bera (normality), Breusch-Pagan (heteroskedasticity), Durbin-Watson (autocorrelation), VIF (multicollinearity), and condition number.")]
    async fn regression_diagnostics(
        &self,
        Parameters(request): Parameters<DiagnosticsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_diagnostics(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Diagnostics failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run OLS with clustered standard errors.
    #[tool(description = "Run OLS regression with clustered standard errors. Supports one-way (firm, state) or two-way (firm + time) clustering. Essential for panel data with correlated errors.")]
    async fn regression_clustered(
        &self,
        Parameters(request): Parameters<OlsClusteredRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_ols_clustered(
            dataset,
            &request.y,
            &x_refs,
            &request.cluster1,
            request.cluster2.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Clustered regression failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Econometrics Tools
    // ========================================================================

    /// Run Fixed Effects panel regression.
    #[tool(description = "Run Fixed Effects (within) panel regression. Controls for time-invariant unobserved heterogeneity. Requires panel data with entity identifiers.")]
    async fn panel_fixed_effects(
        &self,
        Parameters(request): Parameters<PanelFERequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_fixed_effects(dataset, &request.y, &x_refs, &request.entity_var) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Fixed Effects estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Random Effects panel regression.
    #[tool(description = "Run Random Effects (GLS) panel regression. Assumes individual effects are uncorrelated with regressors. More efficient than FE if assumption holds.")]
    async fn panel_random_effects(
        &self,
        Parameters(request): Parameters<PanelRERequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_random_effects(dataset, &request.y, &x_refs, &request.entity_var) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Random Effects estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Hausman specification test.
    #[tool(description = "Run Hausman specification test to choose between Fixed Effects and Random Effects. Tests H0: RE is consistent. If p-value < 0.05, use Fixed Effects.")]
    async fn hausman_test(
        &self,
        Parameters(request): Parameters<HausmanRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_hausman_test(dataset, &request.y, &x_refs, &request.entity_var) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Hausman test failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run IV/2SLS regression.
    #[tool(description = "Run Instrumental Variables (2SLS) regression. Use when an explanatory variable is endogenous (correlated with the error term). Requires valid instruments.")]
    async fn iv_2sls(
        &self,
        Parameters(request): Parameters<IV2SLSRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let robust = request.robust.unwrap_or(true);
        let x_exog_refs: Vec<&str> = request.x_exog.iter().map(|s| s.as_str()).collect();
        let x_endog_refs: Vec<&str> = request.x_endog.iter().map(|s| s.as_str()).collect();
        let instruments_refs: Vec<&str> = request.instruments.iter().map(|s| s.as_str()).collect();

        let result = match run_iv2sls(dataset, &request.y, &x_exog_refs, &x_endog_refs, &instruments_refs, robust) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "IV/2SLS estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run first-stage diagnostics for IV/2SLS.
    #[tool(description = "Run first-stage diagnostics to test instrument strength. Reports F-statistic (F > 10 suggests strong instruments), R-squared, and coefficient estimates. Essential before running 2SLS.")]
    async fn iv_first_stage(
        &self,
        Parameters(request): Parameters<FirstStageRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let instruments: Vec<&str> = request.instruments.iter().map(|s| s.as_str()).collect();
        let controls: Option<Vec<&str>> = request.controls.as_ref()
            .map(|c| c.iter().map(|s| s.as_str()).collect());

        let result = match run_first_stage_diagnostics(
            dataset,
            &request.endogenous_var,
            &instruments,
            controls.as_deref(),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "First-stage diagnostics failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Difference-in-Differences estimation.
    #[tool(description = "Run Difference-in-Differences (DiD) estimation. Estimates causal treatment effects by comparing treated vs control groups before and after treatment.")]
    async fn diff_in_diff(
        &self,
        Parameters(request): Parameters<DiDRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match run_did(dataset, &request.dep_var, &request.treatment_var, &request.post_var, None) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DiD estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Treatment Effect Estimation
    // ========================================================================

    /// Run IPW treatment effect estimation.
    #[tool(description = "Estimate Average Treatment Effect (ATE) or Average Treatment Effect on Treated (ATT) using Inverse Probability Weighting. Uses propensity scores to create pseudo-populations that balance covariates between treatment groups. Returns effect estimate with bootstrap standard errors and confidence intervals.")]
    async fn treatment_ipw(
        &self,
        Parameters(request): Parameters<IpwRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("att") | Some("ATT") => Estimand::ATT,
            _ => Estimand::ATE,
        };

        let config = IpwConfig {
            trim: request.trim.unwrap_or(0.05),
            estimand,
            bootstrap: request.bootstrap.unwrap_or(999),
            normalized: true,
            seed: None,
        };

        let result = match run_ipw_treatment(dataset, &request.outcome, &request.treatment, &cov_refs, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "IPW estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Doubly Robust (AIPW) treatment effect estimation.
    #[tool(description = "Estimate treatment effects using Augmented IPW (doubly robust). Combines propensity score weighting with outcome regression. Consistent if either the propensity model OR the outcome model is correctly specified. Returns effect estimate with bootstrap standard errors and model fit diagnostics.")]
    async fn treatment_doubly_robust(
        &self,
        Parameters(request): Parameters<DoublyRobustRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        // Parse method
        let method = match request.method.as_deref() {
            Some("ipw") | Some("IPW") => DRMethod::IPW,
            Some("regression") | Some("reg") => DRMethod::Regression,
            _ => DRMethod::AIPW,
        };

        // Parse estimand
        let estimand = match request.estimand.as_deref() {
            Some("att") | Some("ATT") => Estimand::ATT,
            _ => Estimand::ATE,
        };

        let config = DoublyRobustConfig {
            method,
            trim: request.trim.unwrap_or(0.05),
            estimand,
            bootstrap: request.bootstrap.unwrap_or(999),
            seed: None,
        };

        let result = match run_doubly_robust(dataset, &request.outcome, &request.treatment, &cov_refs, config) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Doubly robust estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Causal Mediation Analysis
    // ========================================================================

    /// Run causal mediation analysis.
    #[tool(description = "Perform causal mediation analysis to decompose treatment effects into direct and indirect (mediated) effects. Uses IPW-based identification following Huber (2014). Returns Natural Direct Effect (NDE), Natural Indirect Effect (NIE), proportion mediated, and bootstrap inference.")]
    async fn mediation_analysis(
        &self,
        Parameters(request): Parameters<MediationRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let cov_refs: Vec<&str> = request.covariates.iter().map(|s| s.as_str()).collect();

        let config = MediationConfig {
            bootstrap: request.bootstrap.unwrap_or(999),
            trim: request.trim.unwrap_or(0.05),
            seed: None,
        };

        let result = match run_mediation_analysis(
            dataset,
            &request.outcome,
            &request.treatment,
            &request.mediator,
            &cov_refs,
            config,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mediation analysis failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Discrete Choice Models
    // ========================================================================

    /// Run Logit (logistic) regression.
    #[tool(description = "Run Logit (logistic) regression for binary outcomes. Uses MLE with Newton-Raphson. Dependent variable must be 0/1.")]
    async fn logit(
        &self,
        Parameters(request): Parameters<LogitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_logit(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Logit estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Probit regression.
    #[tool(description = "Run Probit regression for binary outcomes. Uses MLE with Newton-Raphson. Dependent variable must be 0/1.")]
    async fn probit(
        &self,
        Parameters(request): Parameters<ProbitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();

        let result = match run_probit(dataset, &request.y, &x_refs) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Probit estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run High-Dimensional Fixed Effects regression with multiple absorbed FE.
    #[tool(description = "Run High-Dimensional Fixed Effects (HDFE) regression with multiple absorbed fixed effects (e.g., firm + year + industry). Uses the Method of Alternating Projections (MAP) for efficient estimation. Equivalent to R's lfe::felm() or Stata's reghdfe.")]
    async fn panel_hdfe(
        &self,
        Parameters(request): Parameters<PanelHdfeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let x_refs: Vec<&str> = request.x.iter().map(|s| s.as_str()).collect();
        let fe_refs: Vec<&str> = request.fe.iter().map(|s| s.as_str()).collect();

        // Build config from optional parameters
        let config = HdfeConfig {
            tolerance: request.tolerance.unwrap_or(1e-8),
            max_iterations: request.max_iterations.unwrap_or(10000),
            accelerate: true,
        };

        // Parse SE type
        let cov_type = match request.se_type.as_deref() {
            Some("standard") => CovarianceType::Standard,
            Some("hc0") => CovarianceType::HC0,
            Some("hc1") | None => CovarianceType::HC1,
            Some("hc2") => CovarianceType::HC2,
            Some("hc3") => CovarianceType::HC3,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown SE type '{}'. Use 'standard', 'hc0', 'hc1', 'hc2', or 'hc3'.",
                    other
                ))]));
            }
        };

        let result = match run_hdfe(dataset, &request.y, &x_refs, &fe_refs, Some(config), cov_type) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "HDFE estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Time Series Models
    // ========================================================================

    /// Run VAR (Vector Autoregression) model.
    #[tool(description = "Run Vector Autoregression (VAR) model for multivariate time series. Returns coefficients, residual covariance, AIC, and BIC.")]
    async fn ts_var(
        &self,
        Parameters(request): Parameters<VarRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_var(dataset, &columns, request.lags) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VAR estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run VARMA (Vector ARMA) model.
    #[tool(description = "Run VARMA(p,q) model using Hannan-Rissanen estimation. Combines autoregressive and moving average components for multivariate time series.")]
    async fn ts_varma(
        &self,
        Parameters(request): Parameters<VarmaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_varma(dataset, &columns, request.p, request.q) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VARMA estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run VECM (Vector Error Correction Model).
    #[tool(description = "Run VECM using Johansen Maximum Likelihood. For cointegrated I(1) time series. Returns cointegration vectors (beta), adjustment speeds (alpha), and eigenvalues.")]
    async fn ts_vecm(
        &self,
        Parameters(request): Parameters<VecmRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_vecm(dataset, &columns, request.lags, request.rank) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VECM estimation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Compute VAR Impulse Response Functions.
    #[tool(description = "Compute Impulse Response Functions (IRF) from a VAR model. Shows how variables respond to shocks over time using Cholesky orthogonalization.")]
    async fn ts_var_irf(
        &self,
        Parameters(request): Parameters<VarIrfRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();

        let result = match run_var_irf(dataset, &columns, request.lags, request.steps) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "VAR IRF computation failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Forecasting Models
    // ========================================================================

    /// Fit an ARIMA model.
    #[tool(description = "Fit an ARIMA(p,d,q) model to a univariate time series. Returns AR/MA coefficients, residuals, AIC, and model diagnostics.")]
    async fn ts_arima_fit(
        &self,
        Parameters(request): Parameters<ArimaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match run_arima(dataset, &request.column, request.p, request.d, request.q) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ARIMA fitting failed: {}",
                    e
                ))]));
            }
        };

        // Format result
        let output = format!(
            "ARIMA({},{},{}) Model Results\n\
             ==============================\n\
             Column: {}\n\
             Observations: {}\n\n\
             AR Coefficients (phi): {:?}\n\
             MA Coefficients (theta): {:?}\n\
             Intercept: {:.6}\n\n\
             Sum of Squared Residuals: {:.4}\n\
             AIC: {:.4}",
            result.p, result.d, result.q,
            result.column,
            result.n_obs,
            result.ar_coeffs,
            result.ma_coeffs,
            result.intercept,
            result.ssr,
            result.aic
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Forecast using an ARIMA model.
    #[tool(description = "Forecast future values using an ARIMA(p,d,q) model. Fits the model and generates h-step ahead forecasts.")]
    async fn ts_arima_forecast(
        &self,
        Parameters(request): Parameters<ArimaForecastRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match forecast_arima(
            dataset,
            &request.column,
            request.p,
            request.d,
            request.q,
            request.horizon,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "ARIMA forecasting failed: {}",
                    e
                ))]));
            }
        };

        // Format result
        let mut output = format!(
            "ARIMA Forecast Results\n\
             ======================\n\
             Column: {}\n\
             Horizon: {} periods\n\n\
             Forecasted Values:\n",
            result.column, result.horizon
        );

        for (i, val) in result.forecast.iter().enumerate() {
            output.push_str(&format!("  t+{}: {:.4}\n", i + 1, val));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Run MSTL decomposition.
    #[tool(description = "Perform MSTL (Multiple Seasonal-Trend decomposition using LOESS) on a time series. Extracts trend, seasonal components, and residuals.")]
    async fn ts_mstl(
        &self,
        Parameters(request): Parameters<MstlRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match run_mstl(dataset, &request.column, &request.periods) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "MSTL decomposition failed: {}",
                    e
                ))]));
            }
        };

        // Format result with summary statistics
        let trend_mean: f64 = result.trend.iter().sum::<f64>() / result.trend.len() as f64;
        let resid_var: f64 = result.residuals.iter().map(|r| r * r).sum::<f64>() / result.residuals.len() as f64;

        let mut output = format!(
            "MSTL Decomposition Results\n\
             ==========================\n\
             Column: {}\n\
             Observations: {}\n\
             Seasonal Periods: {:?}\n\n\
             Component Statistics:\n\
             - Trend mean: {:.4}\n\
             - Residual variance: {:.4}\n",
            result.column,
            result.n_obs,
            result.periods,
            trend_mean,
            resid_var
        );

        // Show first few values of each component
        let show_n = 5.min(result.n_obs);
        output.push_str(&format!("\nFirst {} values:\n", show_n));
        output.push_str("  Trend: [");
        for (i, val) in result.trend.iter().take(show_n).enumerate() {
            if i > 0 { output.push_str(", "); }
            output.push_str(&format!("{:.2}", val));
        }
        output.push_str("]\n");

        for (idx, seasonal) in result.seasonal.iter().enumerate() {
            output.push_str(&format!("  Seasonal (period {}): [", result.periods[idx]));
            for (i, val) in seasonal.iter().take(show_n).enumerate() {
                if i > 0 { output.push_str(", "); }
                output.push_str(&format!("{:.2}", val));
            }
            output.push_str("]\n");
        }

        output.push_str("  Residuals: [");
        for (i, val) in result.residuals.iter().take(show_n).enumerate() {
            if i > 0 { output.push_str(", "); }
            output.push_str(&format!("{:.2}", val));
        }
        output.push_str("]\n");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Detect changepoints (structural breaks) in a time series.
    #[tool(description = "Detect changepoints (structural breaks) in a time series using PELT or Binary Segmentation. Identifies points where the statistical properties (mean, variance) change significantly. Useful for regime detection, anomaly detection, and segmenting time series into homogeneous periods.")]
    async fn ts_changepoint(
        &self,
        Parameters(request): Parameters<ChangepointRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Determine cost function
        let cost_fn = match request.change_type.as_deref() {
            Some("variance") => CostFunction::VarianceChange,
            Some("both") => CostFunction::MeanAndVariance,
            _ => CostFunction::MeanChange,
        };

        // Run detection based on method
        let result = match request.method.as_deref() {
            Some("binary") => {
                match run_binary_segmentation(
                    dataset,
                    &request.column,
                    Some(10), // max changepoints
                    request.min_segment_length,
                    request.penalty,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Changepoint detection failed: {}",
                            e
                        ))]));
                    }
                }
            }
            _ => {
                match run_changepoint(
                    dataset,
                    &request.column,
                    request.penalty,
                    request.min_segment_length,
                    cost_fn,
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Changepoint detection failed: {}",
                            e
                        ))]));
                    }
                }
            }
        };

        // Get observation count from result
        let n_obs: usize = result.segments.iter().map(|s| s.n_points).sum();

        // Format output
        let mut output = format!(
            "Changepoint Detection Results\n\
             ==============================\n\
             Column: {}\n\
             Observations: {}\n\
             Method: {}\n\
             Penalty: {:.4}\n\n\
             Changepoints Detected: {}\n",
            request.column,
            n_obs,
            result.method,
            result.penalty,
            result.n_changepoints,
        );

        if result.n_changepoints > 0 {
            output.push_str(&format!("Changepoint Positions: {:?}\n\n", result.changepoints));
        } else {
            output.push_str("\nNo changepoints detected (series appears stationary).\n\n");
        }

        output.push_str("Segment Statistics:\n");
        for (i, seg) in result.segments.iter().enumerate() {
            output.push_str(&format!(
                "  Segment {}: indices [{}, {}) | n={} | mean={:.4} | variance={:.4}\n",
                i + 1,
                seg.start,
                seg.end,
                seg.n_points,
                seg.mean,
                seg.variance
            ));
        }

        output.push_str(&format!("\nTotal Cost: {:.4}\n", result.total_cost));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Report Generation Tools
    // ========================================================================

    /// Generate an HTML report from structured analysis results.
    #[tool(description = "Generate a self-contained HTML report from analysis results. The report includes proper styling, tables, charts (as embedded images), and is suitable for sharing or printing. Returns the complete HTML document as a string.")]
    async fn generate_report(
        &self,
        Parameters(request): Parameters<GenerateReportRequest>,
    ) -> Result<CallToolResult, McpError> {
        // Build the report structure
        let mut report = HtmlReport::new(&request.title);

        if let Some(ref subtitle) = request.subtitle {
            report = report.with_subtitle(subtitle);
        }

        if let Some(ref author) = request.author {
            report = report.with_author(author);
        }

        // Process each section
        for section_input in &request.sections {
            let mut section = ReportSection::new(&section_input.title);

            for content_input in &section_input.content {
                match content_input.content_type.as_str() {
                    "text" => {
                        if let Some(ref text) = content_input.text {
                            section.add_text(text);
                        }
                    }
                    "code" => {
                        if let Some(ref code) = content_input.text {
                            section.add_code(code, content_input.language.as_deref());
                        }
                    }
                    "table" => {
                        if let (Some(headers), Some(rows)) = (&content_input.headers, &content_input.rows) {
                            let mut table = ReportTable::new(headers.clone());
                            if let Some(ref caption) = content_input.caption {
                                table = table.with_caption(caption);
                            }
                            for row in rows {
                                table.add_row(row.clone());
                            }
                            section.add_table(table);
                        }
                    }
                    "chart" => {
                        if let Some(ref image) = content_input.image_base64 {
                            section.add_chart(
                                image,
                                content_input.chart_title.as_deref(),
                                content_input.chart_caption.as_deref(),
                            );
                        }
                    }
                    "stats" => {
                        if let Some(ref stats) = content_input.stats {
                            let items: Vec<(String, String)> = stats
                                .iter()
                                .filter_map(|pair| {
                                    if pair.len() >= 2 {
                                        Some((pair[0].clone(), pair[1].clone()))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            section.add_statistics(items);
                        }
                    }
                    _ => {
                        // Unknown content type, skip
                    }
                }
            }

            report.add_section(section);
        }

        // Generate the HTML
        let html = report.to_html();

        // Return the HTML - it's quite long so we provide summary info
        let summary = format!(
            "HTML Report Generated\n\
             =====================\n\
             Title: {}\n\
             Sections: {}\n\
             HTML Length: {} characters\n\n\
             The complete HTML report follows:\n\n{}",
            request.title,
            request.sections.len(),
            html.len(),
            html
        );

        Ok(CallToolResult::success(vec![Content::text(summary)]))
    }

    // ========================================================================
    // Machine Learning Tools
    // ========================================================================

    /// Run K-means clustering.
    #[tool(description = "Run K-means clustering to partition data into k clusters. Uses k-means++ initialization for better convergence. Returns cluster assignments, centroids, and inertia (within-cluster sum of squares).")]
    async fn ml_kmeans(
        &self,
        Parameters(request): Parameters<KMeansRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract numeric columns into ndarray
        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match kmeans(
            data.view(),
            request.k,
            request.max_iterations,
            None, // tolerance
            request.n_init,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "K-means clustering failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run DBSCAN clustering.
    #[tool(description = "Run DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering. Finds clusters of arbitrary shape and identifies outliers as noise points. Does not require specifying number of clusters.")]
    async fn ml_dbscan(
        &self,
        Parameters(request): Parameters<DBSCANRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract numeric columns into ndarray
        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let result = match dbscan(data.view(), request.eps, request.min_samples) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DBSCAN clustering failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run PCA (Principal Component Analysis).
    #[tool(description = "Run Principal Component Analysis (PCA) for dimensionality reduction. Returns principal components, explained variance ratios, and loadings. Useful for understanding data structure and reducing feature dimensionality.")]
    async fn ml_pca(
        &self,
        Parameters(request): Parameters<PCARequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        // Extract numeric columns into ndarray
        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let result = match pca(data.view(), request.n_components, false) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "PCA failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Hierarchical (Agglomerative) clustering.
    #[tool(description = "Run Hierarchical clustering using agglomerative approach. Supports Ward, single, complete, and average linkage methods. Returns cluster assignments and dendrogram information.")]
    async fn ml_hierarchical(
        &self,
        Parameters(request): Parameters<HierarchicalRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let linkage_method = match request.linkage.as_deref() {
            Some("single") => Linkage::Single,
            Some("complete") => Linkage::Complete,
            Some("average") => Linkage::Average,
            Some("ward") | None => Linkage::Ward,
            Some(other) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown linkage method '{}'. Use 'single', 'complete', 'average', or 'ward'.",
                    other
                ))]));
            }
        };

        let result = match hierarchical(
            data.view(),
            request.n_clusters,
            linkage_method,
            request.distance_threshold,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Hierarchical clustering failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run t-SNE dimensionality reduction.
    #[tool(description = "Run t-SNE (t-distributed Stochastic Neighbor Embedding) for visualizing high-dimensional data in 2D or 3D. Preserves local structure while revealing clusters. Good for exploratory visualization.")]
    async fn ml_tsne(
        &self,
        Parameters(request): Parameters<TsneRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let data = match extract_numeric_matrix(dataset, &request.columns) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match tsne(
            data.view(),
            request.n_components,
            request.perplexity,
            request.max_iterations,
            request.learning_rate,
            seed,
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "t-SNE failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Random Forest regression.
    #[tool(description = "Run Random Forest regression. Ensemble of decision trees for robust predictions. Returns feature importances, out-of-bag score, and predictions.")]
    async fn ml_random_forest(
        &self,
        Parameters(request): Parameters<RandomForestRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        // Use per-tool seed if provided, otherwise fall back to global seed
        let global_seed = self.global_seed.read().await;
        let seed = request.seed.or(*global_seed);

        let result = match random_forest(
            data.view(),
            target.view(),
            request.n_trees,
            request.max_depth,
            request.min_samples_split,
            request.max_features.as_deref(),
            seed,
            Some(request.features.clone()),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Random Forest failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    /// Run Linear SVM classification.
    #[tool(description = "Run Linear Support Vector Machine (SVM) for binary classification. Uses SMO algorithm. Returns weights, bias, support vector count, and predictions.")]
    async fn ml_svm(
        &self,
        Parameters(request): Parameters<SvmRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let data = match extract_numeric_matrix(dataset, &request.features) {
            Ok(d) => d,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract target column
        let df = dataset.df();
        let target_col = match df.column(&request.target) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Target column '{}' not found: {}",
                    request.target, e
                ))]));
            }
        };

        let target_values: Vec<f64> = match target_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Target column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert target to numeric: {}",
                    e
                ))]));
            }
        };

        let target = ndarray::Array1::from_vec(target_values);

        let result = match linear_svm(
            data.view(),
            target.view(),
            request.c,
            request.max_iterations,
            request.tolerance,
            Some(request.features.clone()),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SVM failed: {}",
                    e
                ))]));
            }
        };

        Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
    }

    // ========================================================================
    // Database Tools
    // ========================================================================

    /// Query a SQLite database and load results as a dataset.
    #[tool(description = "Execute a SQL query against a SQLite database and load the results as a dataset. The resulting dataset can then be analyzed using other tools.")]
    async fn db_sqlite_query(
        &self,
        Parameters(request): Parameters<SqliteQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let result = match query_sqlite(&request.db_path, &request.query) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "SQLite query failed: {}",
                    e
                ))]));
            }
        };

        // Get preview before moving dataframe
        let preview = result.dataframe.head(Some(5));

        // Create dataset from result
        let dataset = Dataset::new(result.dataframe);

        // Generate name
        let name = request.name.unwrap_or_else(|| {
            format!("sqlite_query_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs())
        });

        // Store dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "SQLite Query Results\n\
             ====================\n\
             Rows returned: {}\n\
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            result.rows,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// List tables in a SQLite database.
    #[tool(description = "List all tables in a SQLite database.")]
    async fn db_sqlite_tables(
        &self,
        Parameters(request): Parameters<SqliteListTablesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let tables = match list_sqlite_tables(&request.db_path) {
            Ok(t) => t,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to list tables: {}",
                    e
                ))]));
            }
        };

        if tables.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No tables found in database.",
            )]));
        }

        let output = format!(
            "Tables in SQLite database:\n\n{}",
            tables.iter()
                .map(|t| format!("  - {}", t))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Get schema for a SQLite table.
    #[tool(description = "Get the schema (column names and types) for a table in a SQLite database.")]
    async fn db_sqlite_schema(
        &self,
        Parameters(request): Parameters<SqliteSchemaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let schema = match sqlite_table_schema(&request.db_path, &request.table_name) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to get schema: {}",
                    e
                ))]));
            }
        };

        if schema.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Table '{}' not found or has no columns.",
                request.table_name
            ))]));
        }

        let output = format!(
            "Schema for table '{}':\n\n{}",
            request.table_name,
            schema.iter()
                .map(|(name, dtype)| format!("  - {} ({})", name, dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Query a DuckDB database and load results as a dataset.
    #[tool(description = "Execute a SQL query against a DuckDB database and load the results as a dataset. DuckDB supports advanced analytics SQL including window functions, CTEs, and more.")]
    async fn db_duckdb_query(
        &self,
        Parameters(request): Parameters<DuckDBQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let result = match query_duckdb(&request.db_path, &request.query) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DuckDB query failed: {}",
                    e
                ))]));
            }
        };

        // Get preview before moving dataframe
        let preview = result.dataframe.head(Some(5));

        // Create dataset from result
        let dataset = Dataset::new(result.dataframe);

        // Generate name
        let name = request.name.unwrap_or_else(|| {
            format!("duckdb_query_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs())
        });

        // Store dataset
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "DuckDB Query Results\n\
             ====================\n\
             Rows returned: {}\n\
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            result.rows,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// List tables in a DuckDB database.
    #[tool(description = "List all tables in a DuckDB database.")]
    async fn db_duckdb_tables(
        &self,
        Parameters(request): Parameters<DuckDBListTablesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let tables = match list_duckdb_tables(&request.db_path) {
            Ok(t) => t,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to list tables: {}",
                    e
                ))]));
            }
        };

        if tables.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No tables found in database.",
            )]));
        }

        let output = format!(
            "Tables in DuckDB database:\n\n{}",
            tables.iter()
                .map(|t| format!("  - {}", t))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Get schema for a DuckDB table.
    #[tool(description = "Get the schema (column names and types) for a table in a DuckDB database.")]
    async fn db_duckdb_schema(
        &self,
        Parameters(request): Parameters<DuckDBSchemaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let schema = match duckdb_table_schema(&request.db_path, &request.table_name) {
            Ok(s) => s,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to get schema: {}",
                    e
                ))]));
            }
        };

        if schema.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "Table '{}' not found or has no columns.",
                request.table_name
            ))]));
        }

        let output = format!(
            "Schema for table '{}':\n\n{}",
            request.table_name,
            schema.iter()
                .map(|(name, dtype)| format!("  - {} ({})", name, dtype))
                .collect::<Vec<_>>()
                .join("\n")
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Query a Parquet or CSV file directly using DuckDB SQL.
    #[tool(description = "Execute a SQL query directly on a Parquet or CSV file using DuckDB. This is powerful for filtering, aggregating, or joining large files before loading them as datasets. Use {file} as a placeholder for the file path in your query.")]
    async fn db_query_file(
        &self,
        Parameters(request): Parameters<DuckDBFileQueryRequest>,
    ) -> Result<CallToolResult, McpError> {
        let result = match query_file_with_duckdb(&request.file_path, &request.query) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "DuckDB file query failed: {}",
                    e
                ))]));
            }
        };

        // Get preview before moving dataframe
        let preview = result.dataframe.head(Some(5));

        // Convert to Dataset
        let dataset = Dataset::new(result.dataframe);

        // Generate name
        let name = request.name.unwrap_or_else(|| {
            format!("file_query_{}", std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs())
        });

        // Store in datasets
        let mut datasets = self.datasets.write().await;
        datasets.insert(name.clone(), dataset);

        let output = format!(
            "DuckDB File Query Results\n\
             =========================\n\
             File: {}\n\
             Rows returned: {}\n\
             Columns: {}\n\n\
             Dataset stored as: '{}'\n\n\
             Preview (first 5 rows):\n{}",
            request.file_path,
            result.rows,
            result.columns.join(", "),
            name,
            preview
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Visualization Tools
    // ========================================================================

    /// Generate a histogram for a numeric column.
    #[tool(description = "Generate a histogram visualization for a numeric column. Returns a base64-encoded PNG image along with bin statistics.")]
    async fn viz_histogram(
        &self,
        Parameters(request): Parameters<HistogramRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();
        let col = match df.column(&request.column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Column '{}' not found: {}",
                    request.column, e
                ))]));
            }
        };

        let values: Vec<f64> = match col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().filter_map(|v| v).filter(|v| v.is_finite()).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Not a numeric column: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert to numeric: {}",
                    e
                ))]));
            }
        };

        if values.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "No valid numeric values in column",
            )]));
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.column.clone());
        config.y_label = Some("Frequency".to_string());

        match histogram(&values, request.bins, config) {
            Ok(result) => {
                let output = format!(
                    "Histogram of '{}'\n{}\n\nImage (base64 PNG, {} bytes):\n{}",
                    request.column,
                    "=".repeat(40),
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate histogram: {}",
                e
            ))])),
        }
    }

    /// Generate a scatter plot for two numeric columns.
    #[tool(description = "Generate a scatter plot visualization showing the relationship between two numeric columns. Returns a base64-encoded PNG image.")]
    async fn viz_scatter(
        &self,
        Parameters(request): Parameters<ScatterPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract X values
        let x_col = match df.column(&request.x_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "X column '{}' not found: {}",
                    request.x_column, e
                ))]));
            }
        };

        let x_values: Vec<f64> = match x_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert X to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract Y values
        let y_col = match df.column(&request.y_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Y column '{}' not found: {}",
                    request.y_column, e
                ))]));
            }
        };

        let y_values: Vec<f64> = match y_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert Y to numeric: {}",
                    e
                ))]));
            }
        };

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.x_column.clone());
        config.y_label = Some(request.y_column.clone());

        match scatter_plot(&x_values, &y_values, config) {
            Ok(result) => {
                let output = format!(
                    "Scatter Plot: {} vs {}\n{}\nPoints: {}\nCorrelation: {:.4}\n\nImage (base64 PNG, {} bytes):\n{}",
                    request.x_column,
                    request.y_column,
                    "=".repeat(40),
                    result.n_points,
                    result.correlation.unwrap_or(f64::NAN),
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate scatter plot: {}",
                e
            ))])),
        }
    }

    /// Generate a line chart for time series or sequential data.
    #[tool(description = "Generate a line chart visualization for time series or sequential data. Supports multiple Y series. Returns a base64-encoded PNG image.")]
    async fn viz_line(
        &self,
        Parameters(request): Parameters<LineChartRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract X values (shared across all series)
        let x_col = match df.column(&request.x_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "X column '{}' not found: {}",
                    request.x_column, e
                ))]));
            }
        };

        let x_values: Vec<f64> = match x_col.cast(&DataType::Float64) {
            Ok(c) => match c.f64() {
                Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "X column not numeric: {}",
                        e
                    ))]));
                }
            },
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Cannot convert X to numeric: {}",
                    e
                ))]));
            }
        };

        // Extract Y series - API expects (name, x_vals, y_vals) tuples
        let mut series: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
        let mut series_names = Vec::new();
        for y_col_name in &request.y_columns {
            let y_col = match df.column(y_col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Y column '{}' not found: {}",
                        y_col_name, e
                    ))]));
                }
            };

            let y_values: Vec<f64> = match y_col.cast(&DataType::Float64) {
                Ok(c) => match c.f64() {
                    Ok(f) => f.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Y column '{}' not numeric: {}",
                            y_col_name, e
                        ))]));
                    }
                },
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot convert Y column '{}' to numeric: {}",
                        y_col_name, e
                    ))]));
                }
            };
            series_names.push(y_col_name.clone());
            series.push((y_col_name.clone(), x_values.clone(), y_values));
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some(request.x_column.clone());

        match line_chart(&series, config) {
            Ok(result) => {
                let output = format!(
                    "Line Chart\n{}\nX: {}\nSeries: {}\nPoints: {}\n\nImage (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    request.x_column,
                    series_names.join(", "),
                    result.n_points,
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate line chart: {}",
                e
            ))])),
        }
    }

    /// Generate a box plot for comparing distributions.
    #[tool(description = "Generate a box plot visualization comparing the distributions of one or more numeric columns. Shows median, quartiles, and outliers. Returns a base64-encoded PNG image.")]
    async fn viz_boxplot(
        &self,
        Parameters(request): Parameters<BoxPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract data for each column
        let mut groups = Vec::new();
        for col_name in &request.columns {
            let col = match df.column(col_name) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Column '{}' not found: {}",
                        col_name, e
                    ))]));
                }
            };

            let values: Vec<f64> = match col.cast(&DataType::Float64) {
                Ok(c) => match c.f64() {
                    Ok(f) => f.into_iter().filter_map(|v| v).filter(|v| v.is_finite()).collect(),
                    Err(e) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Column '{}' not numeric: {}",
                            col_name, e
                        ))]));
                    }
                },
                Err(e) => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Cannot convert '{}' to numeric: {}",
                        col_name, e
                    ))]));
                }
            };
            groups.push((col_name.clone(), values));
        }

        if groups.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "No valid columns specified",
            )]));
        }

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.y_label = Some("Value".to_string());

        match box_plot(&groups, config) {
            Ok(result) => {
                let mut output = format!("Box Plot\n{}\n", "=".repeat(40));
                for stat in &result.statistics {
                    output.push_str(&format!(
                        "\n{}:\n  Min: {:.4}, Q1: {:.4}, Median: {:.4}, Q3: {:.4}, Max: {:.4}\n",
                        stat.label, stat.min, stat.q1, stat.median, stat.q3, stat.max
                    ));
                }
                output.push_str(&format!(
                    "\nImage (base64 PNG, {} bytes):\n{}",
                    result.image_base64.len(),
                    result.image_base64
                ));
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate box plot: {}",
                e
            ))])),
        }
    }

    /// Generate a correlation heatmap.
    #[tool(description = "Generate a correlation heatmap visualization for numeric columns. Uses a diverging blue-white-red colormap. Returns a base64-encoded PNG image.")]
    async fn viz_heatmap(
        &self,
        Parameters(request): Parameters<HeatmapRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        // Compute correlation matrix
        let corr_result = match p2a_core::stats::correlation_matrix(dataset) {
            Ok(r) => r,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Failed to compute correlation: {}",
                    e
                ))]));
            }
        };

        if corr_result.columns.len() < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "Need at least 2 numeric columns for correlation heatmap",
            )]));
        }

        // Filter to specified columns if provided
        let (matrix, columns) = if let Some(ref selected_cols) = request.columns {
            // Find indices of requested columns
            let indices: Vec<usize> = selected_cols.iter()
                .filter_map(|name| corr_result.columns.iter().position(|c| c == name))
                .collect();

            if indices.len() < 2 {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Need at least 2 valid numeric columns for correlation heatmap",
                )]));
            }

            // Build filtered matrix
            let filtered_matrix: Vec<Vec<f64>> = indices.iter()
                .map(|&i| indices.iter().map(|&j| corr_result.matrix[i][j]).collect())
                .collect();
            let filtered_cols: Vec<String> = indices.iter()
                .map(|&i| corr_result.columns[i].clone())
                .collect();

            (filtered_matrix, filtered_cols)
        } else {
            (corr_result.matrix.clone(), corr_result.columns.clone())
        };

        match correlation_heatmap(
            &matrix,
            &columns,
            &columns,
            request.title.as_deref(),
            None,
            None,
        ) {
            Ok(result) => {
                let output = format!(
                    "Correlation Heatmap\n{}\nVariables: {}\n\nImage (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    columns.join(", "),
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate heatmap: {}",
                e
            ))])),
        }
    }

    /// Generate an event study plot for treatment effect visualization.
    #[tool(description = "Generate an event study plot showing treatment effects over time with confidence intervals. Used for visualizing DiD or panel event study results. Shows point estimates with CI bands and reference lines at t=0 and y=0.")]
    async fn viz_event_study(
        &self,
        Parameters(request): Parameters<EventStudyRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df.column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col.cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted.f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let time = match extract_numeric(&request.time_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let estimates = match extract_numeric(&request.estimate_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_lower = match extract_numeric(&request.ci_lower_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_upper = match extract_numeric(&request.ci_upper_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let mut config = ChartConfig::default();
        config.title = request.title;
        config.x_label = Some("Time Relative to Treatment".to_string());
        config.y_label = Some("Effect".to_string());

        match event_study_plot(&time, &estimates, &ci_lower, &ci_upper, config) {
            Ok(result) => {
                let output = format!(
                    "Event Study Plot\n{}\nPeriods: {}\n\nImage (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    result.n_periods,
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate event study plot: {}",
                e
            ))])),
        }
    }

    /// Generate a coefficient plot with confidence intervals.
    #[tool(description = "Generate a coefficient plot showing regression coefficients with confidence intervals (error bars). Useful for visualizing regression results. Shows vertical zero line for reference.")]
    async fn viz_coefficient(
        &self,
        Parameters(request): Parameters<CoefficientPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Extract name column
        let name_col = match df.column(&request.name_column) {
            Ok(c) => c,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Name column '{}' not found: {}", request.name_column, e
                ))]));
            }
        };
        let names: Vec<String> = match name_col.str() {
            Ok(s) => s.into_iter().map(|v| v.unwrap_or("").to_string()).collect(),
            Err(_) => (0..name_col.len()).map(|i| format!("Var_{}", i)).collect(),
        };

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df.column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col.cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted.f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let estimates = match extract_numeric(&request.estimate_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_lower = match extract_numeric(&request.ci_lower_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let ci_upper = match extract_numeric(&request.ci_upper_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        let mut config = ChartConfig::default();
        config.title = request.title;

        let horizontal = request.horizontal.unwrap_or(true);

        match coefficient_plot(&names, &estimates, &ci_lower, &ci_upper, config, horizontal) {
            Ok(result) => {
                let output = format!(
                    "Coefficient Plot\n{}\nCoefficients: {}\nOrientation: {}\n\nImage (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    result.n_coefficients,
                    if horizontal { "horizontal" } else { "vertical" },
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate coefficient plot: {}",
                e
            ))])),
        }
    }

    /// Generate an IRF (Impulse Response Function) plot.
    #[tool(description = "Generate an Impulse Response Function (IRF) plot from VAR models. Shows how a variable responds to a shock over time. Optionally includes confidence bands.")]
    async fn viz_irf(
        &self,
        Parameters(request): Parameters<IrfPlotRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df.column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col.cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted.f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let horizon = match extract_numeric(&request.horizon_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let response = match extract_numeric(&request.response_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract optional CI columns
        let ci_lower: Option<Vec<f64>> = if let Some(ref col_name) = request.ci_lower_column {
            match extract_numeric(col_name) {
                Ok(v) => Some(v),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            }
        } else {
            None
        };

        let ci_upper: Option<Vec<f64>> = if let Some(ref col_name) = request.ci_upper_column {
            match extract_numeric(col_name) {
                Ok(v) => Some(v),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            }
        } else {
            None
        };

        let shock_label = request.shock_label.as_deref();
        let response_label = request.response_label.as_deref();
        let config = ChartConfig {
            title: request.title,
            ..ChartConfig::default()
        };

        let has_ci = ci_lower.is_some() && ci_upper.is_some();

        match irf_plot(&horizon, &response, ci_lower.as_deref(), ci_upper.as_deref(), shock_label, response_label, config) {
            Ok(result) => {
                let output = format!(
                    "IRF Plot\n{}\nHorizons: {}\nHas CI bands: {}\nShock: {}\nResponse: {}\n\nImage (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    result.n_horizons,
                    has_ci,
                    result.shock.as_deref().unwrap_or("unnamed"),
                    result.response.as_deref().unwrap_or("unnamed"),
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate IRF plot: {}",
                e
            ))])),
        }
    }

    /// Generate residual diagnostic plots for regression model validation.
    #[tool(description = "Generate four diagnostic plots for regression analysis: (1) Residuals vs Fitted, (2) Normal Q-Q plot, (3) Scale-Location, (4) Residuals vs Leverage. Also calculates Cook's distance for identifying influential observations. Returns four base64-encoded PNG images.")]
    async fn viz_residual_diagnostics(
        &self,
        Parameters(request): Parameters<ResidualDiagnosticsRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        let datasets = self.datasets.read().await;
        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found",
                    request.dataset
                ))]));
            }
        };

        let df = dataset.df();

        // Helper to extract numeric column
        let extract_numeric = |col_name: &str| -> Result<Vec<f64>, String> {
            let col = df.column(col_name)
                .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;
            let casted = col.cast(&DataType::Float64)
                .map_err(|e| format!("Column '{}' not numeric: {}", col_name, e))?;
            let arr = casted.f64()
                .map_err(|e| format!("Column '{}' error: {}", col_name, e))?;
            Ok(arr.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect())
        };

        let fitted = match extract_numeric(&request.fitted_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };
        let residuals = match extract_numeric(&request.residuals_column) {
            Ok(v) => v,
            Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
        };

        // Extract optional leverage column
        let leverage: Option<Vec<f64>> = if let Some(ref col_name) = request.leverage_column {
            match extract_numeric(col_name) {
                Ok(v) => Some(v),
                Err(e) => return Ok(CallToolResult::error(vec![Content::text(e)])),
            }
        } else {
            None
        };

        let config = ChartConfig::default();

        match residual_diagnostics(&fitted, &residuals, leverage.as_deref(), config) {
            Ok(result) => {
                // Find observations with high Cook's distance
                let high_influence: Vec<usize> = result.cooks_distance.iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0.5)
                    .map(|(i, _)| i)
                    .collect();

                let output = format!(
                    "Residual Diagnostics\n{}\n\
                     Observations: {}\n\
                     High influence points (Cook's D > 0.5): {}\n\n\
                     Plot 1: Residuals vs Fitted (base64 PNG, {} bytes):\n{}\n\n\
                     Plot 2: Normal Q-Q (base64 PNG, {} bytes):\n{}\n\n\
                     Plot 3: Scale-Location (base64 PNG, {} bytes):\n{}\n\n\
                     Plot 4: Residuals vs Leverage (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    result.n_observations,
                    if high_influence.is_empty() { "None".to_string() } else { format!("{:?}", high_influence) },
                    result.residuals_vs_fitted.len(),
                    result.residuals_vs_fitted,
                    result.qq_plot.len(),
                    result.qq_plot,
                    result.scale_location.len(),
                    result.scale_location,
                    result.residuals_vs_leverage.len(),
                    result.residuals_vs_leverage
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate residual diagnostics: {}",
                e
            ))])),
        }
    }

    /// Generate a dendrogram visualization from hierarchical clustering results.
    #[tool(description = "Generate a dendrogram (tree diagram) from hierarchical clustering results. Shows how clusters are merged at each level with merge distances. Takes a linkage matrix from hierarchical clustering output.")]
    async fn viz_dendrogram(
        &self,
        Parameters(request): Parameters<DendrogramRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::dendrogram;

        // Convert linkage matrix from Vec<Vec<f64>> to Vec<(usize, usize, f64, usize)>
        let linkage: Vec<(usize, usize, f64, usize)> = request.linkage_matrix
            .iter()
            .filter_map(|row| {
                if row.len() >= 4 {
                    Some((row[0] as usize, row[1] as usize, row[2], row[3] as usize))
                } else {
                    None
                }
            })
            .collect();

        if linkage.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "Invalid linkage matrix: must be array of [cluster1, cluster2, distance, size] tuples".to_string()
            )]));
        }

        let config = ChartConfig {
            width: request.width.unwrap_or(800),
            height: request.height.unwrap_or(600),
            title: request.title,
            x_label: None,
            y_label: Some("Distance".to_string()),
        };

        match dendrogram(&linkage, request.labels.as_deref(), config) {
            Ok(result) => {
                let output = format!(
                    "Dendrogram\n{}\n\
                     Samples: {}\n\
                     Merge steps: {}\n\
                     Max distance: {:.4}\n\n\
                     Image (base64 PNG, {} bytes):\n{}",
                    "=".repeat(40),
                    result.n_samples,
                    result.n_merges,
                    result.max_distance,
                    result.image_base64.len(),
                    result.image_base64
                );
                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
                "Failed to generate dendrogram: {}",
                e
            ))])),
        }
    }

    /// Batch process multiple datasets with the same operation.
    #[tool(description = "Run the same analysis (describe, correlation, or OLS regression) on multiple datasets at once. Useful for comparing results across datasets or processing survey waves.")]
    async fn batch_process(
        &self,
        Parameters(request): Parameters<BatchProcessRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        if request.datasets.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one dataset must be specified".to_string()
            )]));
        }

        let datasets = self.datasets.read().await;
        let mut results = Vec::new();
        let mut combined_stats: Option<Vec<(String, Vec<f64>)>> = if request.combine_results.unwrap_or(false) {
            Some(Vec::new())
        } else {
            None
        };

        for ds_name in &request.datasets {
            let dataset = match datasets.get(ds_name) {
                Some(ds) => ds,
                None => {
                    results.push(format!("Dataset '{}': NOT FOUND", ds_name));
                    continue;
                }
            };

            let df = dataset.df();
            let result = match request.operation.to_lowercase().as_str() {
                "describe" => {
                    // Get summary statistics
                    let columns: Vec<String> = if let Some(ref cols) = request.columns {
                        cols.clone()
                    } else {
                        // Get all numeric columns
                        df.get_columns().iter()
                            .filter(|c| c.dtype().is_primitive_numeric())
                            .map(|c| c.name().to_string())
                            .collect()
                    };

                    let mut stats_output = format!("Dataset: {}\n", ds_name);
                    stats_output.push_str(&format!("{:-<60}\n", ""));

                    for col_name in &columns {
                        if let Ok(col) = df.column(col_name) {
                            if let Ok(casted) = col.cast(&DataType::Float64) {
                                if let Ok(arr) = casted.f64() {
                                    let values: Vec<f64> = arr.into_iter()
                                        .filter_map(|v| v)
                                        .collect();
                                    if !values.is_empty() {
                                        let n = values.len();
                                        let sum: f64 = values.iter().sum();
                                        let mean = sum / n as f64;
                                        let variance: f64 = values.iter()
                                            .map(|v| (v - mean).powi(2))
                                            .sum::<f64>() / (n - 1).max(1) as f64;
                                        let std_dev = variance.sqrt();
                                        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                                        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                                        stats_output.push_str(&format!(
                                            "  {}: n={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}\n",
                                            col_name, n, mean, std_dev, min, max
                                        ));

                                        if let Some(ref mut combined) = combined_stats {
                                            combined.push((format!("{}:{}", ds_name, col_name), vec![n as f64, mean, std_dev, min, max]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    stats_output
                }
                "correlation" => {
                    // Get correlation matrix
                    match correlation_matrix(dataset) {
                        Ok(corr) => {
                            format!("Dataset: {}\n{:?}", ds_name, corr)
                        }
                        Err(e) => format!("Dataset '{}': Error - {}", ds_name, e)
                    }
                }
                "ols" => {
                    // Run OLS regression
                    if let Some(ref cols) = request.columns {
                        if cols.len() < 2 {
                            format!("Dataset '{}': OLS requires at least 2 columns (dependent + independent)", ds_name)
                        } else {
                            let y_col = &cols[0];
                            let x_cols: Vec<&str> = cols[1..].iter().map(|s| s.as_str()).collect();

                            match run_ols(dataset, y_col, &x_cols, true, CovarianceType::HC1) {
                                Ok(ols_result) => {
                                    let mut output = format!("Dataset: {}\n{:-<60}\n", ds_name, "");
                                    output.push_str(&format!("R²: {:.4}, Adj R²: {:.4}\n", ols_result.r_squared, ols_result.adj_r_squared));
                                    output.push_str(&format!("F-stat: {:.4}\n", ols_result.f_statistic));
                                    for coef in &ols_result.coefficients {
                                        output.push_str(&format!(
                                            "  {}: coef={:.4}, se={:.4}, t={:.4}, p={:.4}\n",
                                            coef.name, coef.estimate, coef.std_error, coef.t_value, coef.p_value
                                        ));
                                    }
                                    output
                                }
                                Err(e) => format!("Dataset '{}': Error - {}", ds_name, e)
                            }
                        }
                    } else {
                        format!("Dataset '{}': OLS requires columns to be specified", ds_name)
                    }
                }
                other => format!("Unknown operation: '{}'. Use 'describe', 'correlation', or 'ols'.", other)
            };

            results.push(result);
        }

        let mut output = format!("Batch Processing Results\n{}\n\n", "=".repeat(40));
        output.push_str(&format!("Datasets processed: {}\n", request.datasets.len()));
        output.push_str(&format!("Operation: {}\n\n", request.operation));

        for result in results {
            output.push_str(&result);
            output.push_str("\n\n");
        }

        // Add combined summary if requested
        if let Some(combined) = combined_stats {
            if !combined.is_empty() {
                output.push_str(&format!("Combined Summary\n{}\n", "-".repeat(40)));
                for (name, stats) in combined {
                    output.push_str(&format!(
                        "{}: n={}, mean={:.4}, std={:.4}, min={:.4}, max={:.4}\n",
                        name, stats[0] as usize, stats[1], stats[2], stats[3], stats[4]
                    ));
                }
            }
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Compare the same columns across multiple datasets.
    #[tool(description = "Compare statistics for specific columns across multiple datasets. Useful for comparing distributions, means, and correlations between different datasets (e.g., treatment vs control, before vs after).")]
    async fn compare_datasets(
        &self,
        Parameters(request): Parameters<CompareDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;

        if request.datasets.len() < 2 {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least two datasets must be specified for comparison".to_string()
            )]));
        }

        if request.columns.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one column must be specified for comparison".to_string()
            )]));
        }

        let datasets = self.datasets.read().await;
        let comparison_type = request.comparison_type.as_deref().unwrap_or("summary");

        let mut output = format!("Dataset Comparison\n{}\n\n", "=".repeat(40));
        output.push_str(&format!("Datasets: {:?}\n", request.datasets));
        output.push_str(&format!("Columns: {:?}\n", request.columns));
        output.push_str(&format!("Comparison type: {}\n\n", comparison_type));

        // Collect statistics for each dataset and column
        let mut all_stats: HashMap<String, HashMap<String, (usize, f64, f64, f64, f64)>> = HashMap::new();

        for ds_name in &request.datasets {
            let dataset = match datasets.get(ds_name) {
                Some(ds) => ds,
                None => {
                    output.push_str(&format!("Warning: Dataset '{}' not found\n", ds_name));
                    continue;
                }
            };

            let df = dataset.df();
            let mut ds_stats: HashMap<String, (usize, f64, f64, f64, f64)> = HashMap::new();

            for col_name in &request.columns {
                if let Ok(col) = df.column(col_name) {
                    if let Ok(casted) = col.cast(&DataType::Float64) {
                        if let Ok(arr) = casted.f64() {
                            let values: Vec<f64> = arr.into_iter()
                                .filter_map(|v| v)
                                .collect();
                            if !values.is_empty() {
                                let n = values.len();
                                let sum: f64 = values.iter().sum();
                                let mean = sum / n as f64;
                                let variance: f64 = values.iter()
                                    .map(|v| (v - mean).powi(2))
                                    .sum::<f64>() / (n - 1).max(1) as f64;
                                let std_dev = variance.sqrt();
                                let min = values.iter().copied().fold(f64::INFINITY, f64::min);
                                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                                ds_stats.insert(col_name.clone(), (n, mean, std_dev, min, max));
                            }
                        }
                    }
                }
            }

            all_stats.insert(ds_name.clone(), ds_stats);
        }

        match comparison_type {
            "summary" => {
                // Side-by-side comparison table
                for col_name in &request.columns {
                    output.push_str(&format!("\nColumn: {}\n{}\n", col_name, "-".repeat(60)));
                    output.push_str(&format!("{:<20} {:>10} {:>12} {:>12} {:>12} {:>12}\n",
                        "Dataset", "N", "Mean", "Std Dev", "Min", "Max"));
                    output.push_str(&format!("{}\n", "-".repeat(80)));

                    for ds_name in &request.datasets {
                        if let Some(ds_stats) = all_stats.get(ds_name) {
                            if let Some((n, mean, std, min, max)) = ds_stats.get(col_name) {
                                output.push_str(&format!("{:<20} {:>10} {:>12.4} {:>12.4} {:>12.4} {:>12.4}\n",
                                    ds_name, n, mean, std, min, max));
                            } else {
                                output.push_str(&format!("{:<20} Column not found or not numeric\n", ds_name));
                            }
                        }
                    }

                    // Calculate and show differences between first two datasets
                    if request.datasets.len() >= 2 {
                        let ds1 = &request.datasets[0];
                        let ds2 = &request.datasets[1];
                        if let (Some(stats1), Some(stats2)) = (
                            all_stats.get(ds1).and_then(|s| s.get(col_name)),
                            all_stats.get(ds2).and_then(|s| s.get(col_name))
                        ) {
                            let mean_diff = stats2.1 - stats1.1;
                            let pct_diff = if stats1.1.abs() > 1e-10 {
                                (mean_diff / stats1.1) * 100.0
                            } else {
                                f64::NAN
                            };
                            output.push_str(&format!("\nDifference ({} - {}): mean diff = {:.4} ({:.2}%)\n",
                                ds2, ds1, mean_diff, pct_diff));
                        }
                    }
                }
            }
            "distribution" => {
                // Distribution comparison (basic)
                for col_name in &request.columns {
                    output.push_str(&format!("\nColumn: {} - Distribution Comparison\n{}\n", col_name, "-".repeat(60)));

                    for ds_name in &request.datasets {
                        if let Some(ds_stats) = all_stats.get(ds_name) {
                            if let Some((n, mean, std, min, max)) = ds_stats.get(col_name) {
                                let range = max - min;
                                let cv = if mean.abs() > 1e-10 { std / mean.abs() } else { f64::NAN };
                                output.push_str(&format!(
                                    "{}: n={}, range={:.4}, CV={:.4}\n",
                                    ds_name, n, range, cv
                                ));
                            }
                        }
                    }
                }
            }
            "correlation" => {
                // Correlation comparison (if multiple columns)
                if request.columns.len() < 2 {
                    output.push_str("Correlation comparison requires at least 2 columns\n");
                } else {
                    for ds_name in &request.datasets {
                        if let Some(dataset) = datasets.get(ds_name) {
                            match correlation_matrix(dataset) {
                                Ok(corr) => {
                                    output.push_str(&format!("\n{}\n{:?}\n", ds_name, corr));
                                }
                                Err(e) => {
                                    output.push_str(&format!("\n{}: Error - {}\n", ds_name, e));
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                output.push_str(&format!("Unknown comparison type: '{}'. Use 'summary', 'distribution', or 'correlation'.\n", comparison_type));
            }
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    // ========================================================================
    // Data Munging Tools
    // ========================================================================

    /// Filter rows in a dataset based on a condition.
    #[tool(description = "Filter rows in a dataset based on a column condition. Supports operators: 'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'contains', 'starts_with', 'ends_with'. The value is parsed based on the column type.")]
    async fn munge_filter(
        &self,
        Parameters(request): Parameters<FilterDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found. Use 'list_datasets' to see available datasets.",
                    request.dataset
                ))]));
            }
        };

        let result = match filter(dataset, &request.column, &request.op, &request.value) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Filter failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Filtered dataset saved as '{}' ({} rows, {} columns)",
            result_name, n_rows, n_cols
        ))]))
    }

    /// Select specific columns from a dataset.
    #[tool(description = "Select (keep) specific columns from a dataset, dropping all others.")]
    async fn munge_select(
        &self,
        Parameters(request): Parameters<SelectColumnsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let cols: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let result = match select(dataset, &cols) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Select failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Selected {} columns, saved as '{}' ({} rows)",
            n_cols, result_name, n_rows
        ))]))
    }

    /// Drop columns from a dataset.
    #[tool(description = "Drop (remove) specific columns from a dataset.")]
    async fn munge_drop_columns(
        &self,
        Parameters(request): Parameters<DropColumnsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let cols: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let result = match drop_columns(dataset, &cols) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Drop columns failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Dropped {} columns, saved as '{}' ({} rows, {} columns remaining)",
            request.columns.len(), result_name, n_rows, n_cols
        ))]))
    }

    /// Rename columns in a dataset.
    #[tool(description = "Rename columns in a dataset. Provide pairs of [old_name, new_name].")]
    async fn munge_rename(
        &self,
        Parameters(request): Parameters<RenameColumnsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let renames: Vec<(&str, &str)> = request.renames.iter()
            .filter_map(|pair| {
                if pair.len() >= 2 {
                    Some((pair[0].as_str(), pair[1].as_str()))
                } else {
                    None
                }
            })
            .collect();

        let result = match rename(dataset, &renames) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Rename failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Renamed {} columns, saved as '{}'",
            renames.len(), result_name
        ))]))
    }

    /// Sort a dataset by one or more columns.
    #[tool(description = "Sort a dataset by one or more columns in ascending or descending order.")]
    async fn munge_sort(
        &self,
        Parameters(request): Parameters<SortDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let by_cols: Vec<&str> = request.by.iter().map(|s| s.as_str()).collect();
        let descending = request.descending.unwrap_or(false);
        let descending_flags: Vec<bool> = vec![descending; by_cols.len()];

        let result = match sort(dataset, &by_cols, &descending_flags) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sort failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Sorted by {:?} ({}), saved as '{}' ({} rows)",
            request.by,
            if descending { "descending" } else { "ascending" },
            result_name,
            n_rows
        ))]))
    }

    /// Join two datasets on key columns.
    #[tool(description = "Join two datasets on key columns. Supports 'left', 'right', 'inner', and 'full' join types.")]
    async fn munge_join(
        &self,
        Parameters(request): Parameters<JoinDatasetsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let left_ds = match datasets.get(&request.left) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Left dataset '{}' not found.",
                    request.left
                ))]));
            }
        };

        let right_ds = match datasets.get(&request.right) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Right dataset '{}' not found.",
                    request.right
                ))]));
            }
        };

        let left_on: Vec<&str> = request.left_on.iter().map(|s| s.as_str()).collect();
        let right_on_vec: Option<Vec<&str>> = request.right_on.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let right_on: Option<&[&str]> = right_on_vec.as_deref();
        let suffix: Option<&str> = request.suffix.as_deref();

        let join_type = request.join_type.as_deref().unwrap_or("left");
        let result = match join_type {
            "left" => left_join(left_ds, right_ds, &left_on, right_on, suffix),
            "right" => right_join(left_ds, right_ds, &left_on, right_on, suffix),
            "inner" => inner_join(left_ds, right_ds, &left_on, right_on, suffix),
            "full" => full_join(left_ds, right_ds, &left_on, right_on, suffix),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown join type: '{}'. Use 'left', 'right', 'inner', or 'full'.",
                    join_type
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Join failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| format!("{}_{}", request.left, request.right));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} join completed, saved as '{}' ({} rows, {} columns)",
            join_type, result_name, n_rows, n_cols
        ))]))
    }

    /// Concatenate multiple datasets vertically.
    #[tool(description = "Concatenate (row-bind) multiple datasets vertically. All datasets must have the same columns.")]
    async fn munge_concat(
        &self,
        Parameters(request): Parameters<ConcatDatasetsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let mut ds_list: Vec<&Dataset> = Vec::new();
        for name in &request.datasets {
            match datasets.get(name) {
                Some(ds) => ds_list.push(ds),
                None => {
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Dataset '{}' not found.",
                        name
                    ))]));
                }
            }
        }

        let result = match concat(&ds_list) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Concat failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| "concatenated".to_string());
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Concatenated {} datasets, saved as '{}' ({} rows, {} columns)",
            request.datasets.len(), result_name, n_rows, n_cols
        ))]))
    }

    /// Group by columns and compute aggregations.
    #[tool(description = "Group a dataset by columns and compute aggregations. Supported functions: 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var', 'first', 'last'.")]
    async fn munge_group_by(
        &self,
        Parameters(request): Parameters<GroupByRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let by_cols: Vec<&str> = request.by.iter().map(|s| s.as_str()).collect();

        // Parse aggregation specs
        let mut agg_specs: Vec<AggSpec> = Vec::new();
        for spec in &request.aggs {
            if spec.len() >= 2 {
                let col = &spec[0];
                let func_str = spec[1].to_lowercase();
                let agg_fn = match func_str.as_str() {
                    "count" => AggFn::Count,
                    "sum" => AggFn::Sum,
                    "mean" => AggFn::Mean,
                    "median" => AggFn::Median,
                    "min" => AggFn::Min,
                    "max" => AggFn::Max,
                    "std" => AggFn::Std,
                    "var" => AggFn::Var,
                    "first" => AggFn::First,
                    "last" => AggFn::Last,
                    _ => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Unknown aggregation function: '{}'. Use: count, sum, mean, median, min, max, std, var, first, last.",
                            func_str
                        ))]));
                    }
                };
                agg_specs.push(AggSpec::new(col, agg_fn));
            }
        }

        if agg_specs.is_empty() {
            return Ok(CallToolResult::error(vec![Content::text(
                "At least one aggregation spec is required. Format: [[\"column\", \"function\"], ...]".to_string()
            )]));
        }

        let result = match group_by(dataset, &by_cols, &agg_specs) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Group by failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| format!("{}_grouped", request.dataset));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Grouped by {:?} with {} aggregations, saved as '{}' ({} groups, {} columns)",
            request.by, agg_specs.len(), result_name, n_rows, n_cols
        ))]))
    }

    /// Compute value counts for a column.
    #[tool(description = "Compute frequency counts for unique values in a column. Optionally normalize to percentages.")]
    async fn munge_value_counts(
        &self,
        Parameters(request): Parameters<ValueCountsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match value_counts(dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Value counts failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| format!("{}_value_counts", request.column));
        let n_rows = result.nrows();

        // Format output for display
        let mut output = format!(
            "Value Counts for '{}'\n{}\n",
            request.column, "=".repeat(40)
        );

        // Show first few rows
        let show_n = 10.min(n_rows);
        output.push_str(&format!("Showing top {} of {} unique values:\n\n", show_n, n_rows));

        let df = result.df();
        for i in 0..show_n {
            let val_col = df.column(&request.column).ok();
            let count_col = df.column("count").ok();

            if let (Some(v), Some(c)) = (val_col, count_col) {
                if let (Ok(val), Ok(cnt)) = (v.get(i), c.get(i)) {
                    output.push_str(&format!("  {:?}: {}\n", val, cnt));
                }
            }
        }

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        output.push_str(&format!("\nFull result saved as '{}'", result_name));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Pivot a dataset from long to wide format.
    #[tool(description = "Pivot a dataset from long to wide format. Index columns remain as rows, 'on' column values become new column names, and 'values' column fills those columns.")]
    async fn munge_pivot(
        &self,
        Parameters(request): Parameters<PivotDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let index: Vec<&str> = request.index.iter().map(|s| s.as_str()).collect();

        let result = match pivot(dataset, &index, &request.on, &request.values) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Pivot failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| format!("{}_pivoted", request.dataset));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Pivoted to wide format, saved as '{}' ({} rows, {} columns)",
            result_name, n_rows, n_cols
        ))]))
    }

    /// Melt a dataset from wide to long format.
    #[tool(description = "Melt a dataset from wide to long format. ID variables remain as-is, value variables are unpivoted into rows.")]
    async fn munge_melt(
        &self,
        Parameters(request): Parameters<MeltDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let id_vars: Vec<&str> = request.id_vars.iter().map(|s| s.as_str()).collect();
        let value_vars: Vec<&str> = request.value_vars.iter().map(|s| s.as_str()).collect();
        let variable_name = request.variable_name.as_deref().unwrap_or("variable");
        let value_name = request.value_name.as_deref().unwrap_or("value");

        let result = match melt(dataset, &id_vars, &value_vars, variable_name, value_name) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Melt failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| format!("{}_melted", request.dataset));
        let n_rows = result.nrows();
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Melted to long format, saved as '{}' ({} rows, {} columns)",
            result_name, n_rows, n_cols
        ))]))
    }

    /// Drop rows with null values.
    #[tool(description = "Drop rows containing null values. Use 'any' to drop if any column is null, 'all' to drop only if all columns are null.")]
    async fn munge_drop_na(
        &self,
        Parameters(request): Parameters<DropNaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Option<Vec<&str>> = request.columns.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let how = request.how.as_deref().unwrap_or("any");

        let orig_rows = dataset.nrows();
        let result = match drop_na(dataset, columns.as_deref(), how) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Drop NA failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let dropped = orig_rows - n_rows;

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Dropped {} rows with null values, saved as '{}' ({} rows remaining)",
            dropped, result_name, n_rows
        ))]))
    }

    /// Fill null values using a strategy.
    #[tool(description = "Fill null values using a strategy: 'mean', 'median', 'mode', 'forward', 'backward', or a constant value.")]
    async fn munge_fill_na(
        &self,
        Parameters(request): Parameters<FillNaRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Option<Vec<&str>> = request.columns.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let strategy = match request.strategy.to_lowercase().as_str() {
            "mean" => FillStrategy::Mean,
            "median" => FillStrategy::Median,
            "forward" => FillStrategy::Forward,
            "backward" => FillStrategy::Backward,
            "zero" => FillStrategy::Zero,
            val => {
                // Try to use as a constant value string
                FillStrategy::Constant(val.to_string())
            }
        };

        let result = match fill_na(dataset, columns.as_deref(), strategy) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Fill NA failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Filled null values with strategy '{}', saved as '{}'",
            request.strategy, result_name
        ))]))
    }

    /// Remove duplicate rows.
    #[tool(description = "Remove duplicate rows from a dataset. Specify which duplicate to keep: 'first', 'last', or 'none'.")]
    async fn munge_deduplicate(
        &self,
        Parameters(request): Parameters<DeduplicateRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Option<Vec<&str>> = request.columns.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());
        let keep = request.keep.as_deref().unwrap_or("first");

        let orig_rows = dataset.nrows();
        let result = match deduplicate(dataset, columns.as_deref(), keep) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Deduplicate failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_rows = result.nrows();
        let removed = orig_rows - n_rows;

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Removed {} duplicate rows, saved as '{}' ({} rows remaining)",
            removed, result_name, n_rows
        ))]))
    }

    // =========================================================================
    // STRING CLEANING TOOLS
    // =========================================================================

    /// Trim whitespace from string columns.
    #[tool(description = "Trim leading and trailing whitespace from string columns. If no columns specified, trims all string columns.")]
    async fn str_trim(
        &self,
        Parameters(request): Parameters<TrimRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Option<Vec<&str>> = request.columns.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let result = match trim(dataset, columns.as_deref()) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Trim failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        let cols_desc = request.columns
            .as_ref()
            .map(|c| c.join(", "))
            .unwrap_or_else(|| "all string columns".to_string());

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Trimmed whitespace from {}, saved as '{}'",
            cols_desc, result_name
        ))]))
    }

    /// Convert string column to lowercase.
    #[tool(description = "Convert all characters in a string column to lowercase.")]
    async fn str_to_lowercase(
        &self,
        Parameters(request): Parameters<ToLowercaseRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match to_lowercase(dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "To lowercase failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Converted '{}' to lowercase, saved as '{}'",
            request.column, result_name
        ))]))
    }

    /// Convert string column to uppercase.
    #[tool(description = "Convert all characters in a string column to uppercase.")]
    async fn str_to_uppercase(
        &self,
        Parameters(request): Parameters<ToUppercaseRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match to_uppercase(dataset, &request.column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "To uppercase failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Converted '{}' to uppercase, saved as '{}'",
            request.column, result_name
        ))]))
    }

    /// Replace exact values in a column.
    #[tool(description = "Replace exact values in a column with a new value. For pattern-based replacement, use str_regex_replace.")]
    async fn str_replace_value(
        &self,
        Parameters(request): Parameters<ReplaceValueRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match replace(dataset, &request.column, &request.old_value, &request.new_value) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Replace failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Replaced '{}' with '{}' in column '{}', saved as '{}'",
            request.old_value, request.new_value, request.column, result_name
        ))]))
    }

    // =========================================================================
    // REGEX TOOLS
    // =========================================================================

    /// Replace substrings matching a regex pattern.
    #[tool(description = "Replace substrings matching a regex pattern with a replacement string. Supports capture groups ($1, $2, etc.) in the replacement.")]
    async fn str_regex_replace(
        &self,
        Parameters(request): Parameters<RegexReplaceRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match regex_replace(dataset, &request.column, &request.pattern, &request.replacement) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regex replace failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Replaced pattern '{}' in '{}', saved as '{}'",
            request.pattern, request.column, result_name
        ))]))
    }

    /// Extract substrings matching a regex pattern into a new column.
    #[tool(description = "Extract substrings matching a regex pattern into a new column. Use capture groups () to specify what to extract, or extract the whole match.")]
    async fn str_regex_extract(
        &self,
        Parameters(request): Parameters<RegexExtractRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let group = request.group.unwrap_or(1);

        let result = match regex_extract(dataset, &request.column, &request.pattern, &request.new_column, group) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regex extract failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Extracted pattern '{}' from '{}' into '{}', saved as '{}'",
            request.pattern, request.column, request.new_column, result_name
        ))]))
    }

    /// Count regex pattern matches in each row.
    #[tool(description = "Count the number of times a regex pattern matches in each row, creating a new integer column with the counts.")]
    async fn str_regex_count(
        &self,
        Parameters(request): Parameters<RegexCountRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match regex_count(dataset, &request.column, &request.pattern, &request.new_column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Regex count failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Counted pattern '{}' matches from '{}' into '{}', saved as '{}'",
            request.pattern, request.column, request.new_column, result_name
        ))]))
    }

    // =========================================================================
    // STRING MANIPULATION TOOLS
    // =========================================================================

    /// Split a string column into multiple columns.
    #[tool(description = "Split a string column by a pattern (supports regex) into multiple columns named prefix_0, prefix_1, etc.")]
    async fn str_split(
        &self,
        Parameters(request): Parameters<StrSplitRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match str_split(dataset, &request.column, &request.pattern, request.max_splits, &request.prefix) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String split failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Split '{}' by '{}' into columns with prefix '{}', saved as '{}'",
            request.column, request.pattern, request.prefix, result_name
        ))]))
    }

    /// Concatenate multiple string columns.
    #[tool(description = "Concatenate multiple string columns into a new column, optionally with a separator between values.")]
    async fn str_concat(
        &self,
        Parameters(request): Parameters<StrConcatRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let columns: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let separator = request.separator.as_deref();

        let result = match str_concat(dataset, &columns, &request.new_column, separator) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String concat failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Concatenated {} columns into '{}', saved as '{}'",
            request.columns.len(), request.new_column, result_name
        ))]))
    }

    /// Get string lengths.
    #[tool(description = "Create a new column containing the length (number of characters) of each string in the source column.")]
    async fn str_length(
        &self,
        Parameters(request): Parameters<StrLengthRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match str_length(dataset, &request.column, &request.new_column) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String length failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created length column '{}' from '{}', saved as '{}'",
            request.new_column, request.column, result_name
        ))]))
    }

    /// Extract a substring from a string column.
    #[tool(description = "Extract a substring from each string in a column. Supports negative indices to count from end.")]
    async fn str_substring(
        &self,
        Parameters(request): Parameters<StrSubstringRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let result = match str_substring(dataset, &request.column, request.start, request.length) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "String substring failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        let length_desc = request.length
            .map(|l| format!(", length {}", l))
            .unwrap_or_else(|| " to end".to_string());

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Extracted substring from '{}' (start: {}{}), saved as '{}'",
            request.column, request.start, length_desc, result_name
        ))]))
    }

    /// Create lag or lead columns for time series data.
    #[tool(description = "Create lag or lead columns for time series or panel data. Lag shifts values forward (past values), lead shifts values backward (future values).")]
    async fn munge_lag_lead(
        &self,
        Parameters(request): Parameters<LagLeadRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let periods = request.periods.unsigned_abs() as usize;
        let group_by_cols: Option<Vec<&str>> = request.group_by.as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let direction = request.direction.as_deref().unwrap_or("lag");
        let result = match direction {
            "lag" => lag(dataset, &request.column, periods, group_by_cols.as_deref()),
            "lead" => lead(dataset, &request.column, periods, group_by_cols.as_deref()),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown direction: '{}'. Use 'lag' or 'lead'.",
                    direction
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Lag/lead failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let new_col = format!("{}_{}{}", request.column, direction, periods);

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created '{}' column, saved as '{}'",
            new_col, result_name
        ))]))
    }

    /// Standardize or normalize columns.
    #[tool(description = "Standardize (z-score) or normalize (0-1 range) numeric columns. Standardize subtracts mean and divides by std. Normalize scales to [0, 1].")]
    async fn munge_standardize(
        &self,
        Parameters(request): Parameters<StandardizeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let cols: Vec<&str> = request.columns.iter().map(|s| s.as_str()).collect();
        let method = request.method.as_deref().unwrap_or("standardize");

        let result = match method {
            "standardize" => standardize(dataset, &cols),
            "normalize" => normalize(dataset, &cols),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown method: '{}'. Use 'standardize' or 'normalize'.",
                    method
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "{} failed: {}",
                    method, e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Applied {} to {} columns, saved as '{}'",
            method, request.columns.len(), result_name
        ))]))
    }

    /// Bin a continuous variable into discrete categories.
    #[tool(description = "Bin a continuous variable into discrete categories. Strategies: 'uniform' (equal width), 'quantile' (equal frequency), or 'custom' (specify break points).")]
    async fn munge_bin(
        &self,
        Parameters(request): Parameters<BinColumnRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let strategy = match request.strategy.to_lowercase().as_str() {
            "uniform" | "equal_width" => {
                let n_bins = request.bins.first().map(|&v| v as usize).unwrap_or(5);
                BinStrategy::EqualWidth(n_bins)
            }
            "quantile" => {
                let n_bins = request.bins.first().map(|&v| v as usize).unwrap_or(5);
                BinStrategy::Quantile(n_bins)
            }
            "custom" => {
                BinStrategy::Custom(request.bins.clone())
            }
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown strategy: '{}'. Use 'uniform', 'quantile', or 'custom'.",
                    request.strategy
                ))]));
            }
        };

        let labels = request.labels.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect::<Vec<_>>());

        let result = match bin(dataset, &request.column, strategy, labels.as_deref()) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Bin failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Binned '{}' using {} strategy, saved as '{}'",
            request.column, request.strategy, result_name
        ))]))
    }

    /// One-hot encode a categorical column.
    #[tool(description = "One-hot encode a categorical column, creating binary indicator columns for each category. Use drop_first=true to avoid multicollinearity in regression.")]
    async fn munge_one_hot_encode(
        &self,
        Parameters(request): Parameters<OneHotEncodeRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let drop_first = request.drop_first.unwrap_or(false);

        let result = match one_hot_encode(dataset, &request.column, drop_first) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "One-hot encode failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let n_cols = result.ncols();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "One-hot encoded '{}' (drop_first={}), saved as '{}' ({} total columns)",
            request.column, drop_first, result_name, n_cols
        ))]))
    }

    /// Compute differences or percent changes.
    #[tool(description = "Compute differences or percent changes for a column. Useful for time series and panel data analysis.")]
    async fn munge_diff(
        &self,
        Parameters(request): Parameters<DiffRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let periods = request.periods.unwrap_or(1) as usize;
        let diff_type = request.diff_type.as_deref().unwrap_or("diff");

        let result = match diff_type {
            "diff" => diff(dataset, &request.column, periods),
            "pct_change" => pct_change(dataset, &request.column, periods),
            _ => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown diff type: '{}'. Use 'diff' or 'pct_change'.",
                    diff_type
                ))]));
            }
        };

        let result = match result {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Diff/pct_change failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());
        let new_col = format!("{}_{}{}", request.column, diff_type, periods);

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created '{}' column, saved as '{}'",
            new_col, result_name
        ))]))
    }

    /// Sample rows from a dataset.
    #[tool(description = "Randomly sample rows from a dataset. Useful for creating training/test splits or working with large datasets.")]
    async fn munge_sample(
        &self,
        Parameters(request): Parameters<SampleDatasetRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let replace = request.replace.unwrap_or(false);
        let seed = request.seed;

        let result = match sample(dataset, Some(request.n), None, replace, seed) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Sample failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| format!("{}_sample", request.dataset));
        let n_rows = result.nrows();

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Sampled {} rows (replace={}), saved as '{}'",
            n_rows, replace, result_name
        ))]))
    }

    /// Create a new column by computation.
    #[tool(description = "Create a new column by applying arithmetic operations or functions. Supports: arithmetic (+, -, *, /), functions (log, exp, sqrt, abs, square), or constant values.")]
    async fn munge_mutate(
        &self,
        Parameters(request): Parameters<MutateColumnRequest>,
    ) -> Result<CallToolResult, McpError> {
        let datasets = self.datasets.read().await;

        let dataset = match datasets.get(&request.dataset) {
            Some(ds) => ds,
            None => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Dataset '{}' not found.",
                    request.dataset
                ))]));
            }
        };

        let expr = match request.expr_type.as_str() {
            "arithmetic" => {
                let op = match request.operator.as_deref() {
                    Some("+") => ArithOp::Add,
                    Some("-") => ArithOp::Sub,
                    Some("*") => ArithOp::Mul,
                    Some("/") => ArithOp::Div,
                    Some(other) => {
                        return Ok(CallToolResult::error(vec![Content::text(format!(
                            "Unknown operator: '{}'. Use '+', '-', '*', or '/'.",
                            other
                        ))]));
                    }
                    None => {
                        return Ok(CallToolResult::error(vec![Content::text(
                            "Arithmetic expressions require an 'operator' field.".to_string()
                        )]));
                    }
                };

                let right = request.right.as_deref().ok_or_else(|| {
                    McpError::invalid_request("Arithmetic expressions require a 'right' field", None)
                })?;

                MutateExpr::Arithmetic(request.left.clone(), op, right.to_string())
            }
            "function" => {
                let func = request.operator.as_deref().unwrap_or("log");
                MutateExpr::Function(func.to_string(), request.left.clone())
            }
            "constant" => {
                MutateExpr::Constant(request.left.clone())
            }
            other => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Unknown expression type: '{}'. Use 'arithmetic', 'function', or 'constant'.",
                    other
                ))]));
            }
        };

        let result = match mutate(dataset, &request.new_column, expr) {
            Ok(ds) => ds,
            Err(e) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "Mutate failed: {}",
                    e
                ))]));
            }
        };

        let result_name = request.result_name.unwrap_or_else(|| request.dataset.clone());

        drop(datasets);
        let mut datasets = self.datasets.write().await;
        datasets.insert(result_name.clone(), result);

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Created column '{}', saved as '{}'",
            request.new_column, result_name
        ))]))
    }

    /// Export the current analysis session to a JSON file.
    #[tool(description = "Export the current session including all loaded datasets and their metadata. Can save to file or return as string. Useful for saving your analysis state to resume later.")]
    async fn export_session(
        &self,
        Parameters(request): Parameters<ExportSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;
        use std::fs;

        let datasets = self.datasets.read().await;
        let include_data = request.include_data.unwrap_or(true);

        let mut session_data = serde_json::Map::new();
        session_data.insert("version".to_string(), serde_json::json!("1.0"));
        session_data.insert("created_at".to_string(), serde_json::json!(chrono::Utc::now().to_rfc3339()));

        let mut datasets_json = serde_json::Map::new();
        for (name, dataset) in datasets.iter() {
            let df = dataset.df();
            let mut ds_info = serde_json::Map::new();

            // Save schema
            let schema: Vec<serde_json::Value> = df.get_columns()
                .iter()
                .map(|col| {
                    serde_json::json!({
                        "name": col.name().to_string(),
                        "dtype": format!("{:?}", col.dtype())
                    })
                })
                .collect();
            ds_info.insert("schema".to_string(), serde_json::json!(schema));
            ds_info.insert("n_rows".to_string(), serde_json::json!(df.height()));
            ds_info.insert("n_cols".to_string(), serde_json::json!(df.width()));

            if include_data {
                // Serialize actual data
                let mut columns_data = serde_json::Map::new();
                for col in df.get_columns() {
                    let col_name = col.name().to_string();
                    let values: Vec<serde_json::Value> = (0..col.len())
                        .map(|i| {
                            match col.get(i) {
                                Ok(av) => match av {
                                    AnyValue::Null => serde_json::Value::Null,
                                    AnyValue::Boolean(b) => serde_json::json!(b),
                                    AnyValue::Int8(v) => serde_json::json!(v),
                                    AnyValue::Int16(v) => serde_json::json!(v),
                                    AnyValue::Int32(v) => serde_json::json!(v),
                                    AnyValue::Int64(v) => serde_json::json!(v),
                                    AnyValue::UInt8(v) => serde_json::json!(v),
                                    AnyValue::UInt16(v) => serde_json::json!(v),
                                    AnyValue::UInt32(v) => serde_json::json!(v),
                                    AnyValue::UInt64(v) => serde_json::json!(v),
                                    AnyValue::Float32(v) => serde_json::json!(v),
                                    AnyValue::Float64(v) => serde_json::json!(v),
                                    AnyValue::String(s) => serde_json::json!(s),
                                    _ => serde_json::json!(format!("{:?}", av)),
                                },
                                Err(_) => serde_json::Value::Null,
                            }
                        })
                        .collect();
                    columns_data.insert(col_name, serde_json::json!(values));
                }
                ds_info.insert("data".to_string(), serde_json::json!(columns_data));
            }

            datasets_json.insert(name.clone(), serde_json::json!(ds_info));
        }
        session_data.insert("datasets".to_string(), serde_json::json!(datasets_json));

        let json_output = serde_json::to_string_pretty(&session_data)
            .map_err(|e| McpError::internal_error(format!("JSON serialization failed: {}", e), None))?;

        if let Some(file_path) = request.file_path {
            fs::write(&file_path, &json_output)
                .map_err(|e| McpError::internal_error(format!("Failed to write session file: {}", e), None))?;

            Ok(CallToolResult::success(vec![Content::text(format!(
                "Session exported successfully to: {}\n\
                 Datasets saved: {}\n\
                 Include data: {}",
                file_path,
                datasets.len(),
                include_data
            ))]))
        } else {
            Ok(CallToolResult::success(vec![Content::text(format!(
                "Session Export\n{}\n\
                 Datasets: {}\n\n{}",
                "=".repeat(40),
                datasets.len(),
                json_output
            ))]))
        }
    }

    /// Import a previously exported analysis session.
    #[tool(description = "Import a previously exported session from a JSON file. Can merge with existing session or replace it. Restores all datasets with their original names.")]
    async fn import_session(
        &self,
        Parameters(request): Parameters<ImportSessionRequest>,
    ) -> Result<CallToolResult, McpError> {
        use p2a_core::polars::prelude::*;
        use std::fs;

        let json_content = fs::read_to_string(&request.file_path)
            .map_err(|e| McpError::internal_error(format!("Failed to read session file: {}", e), None))?;

        let session: serde_json::Value = serde_json::from_str(&json_content)
            .map_err(|e| McpError::internal_error(format!("Invalid JSON: {}", e), None))?;

        let datasets_obj = session.get("datasets")
            .and_then(|v| v.as_object())
            .ok_or_else(|| McpError::internal_error("Invalid session format: missing 'datasets' field", None))?;

        let merge = request.merge.unwrap_or(false);
        let mut datasets = self.datasets.write().await;

        if !merge {
            datasets.clear();
        }

        let mut imported_count = 0;
        let mut errors = Vec::new();

        for (name, ds_info) in datasets_obj {
            let ds_obj = match ds_info.as_object() {
                Some(obj) => obj,
                None => {
                    errors.push(format!("{}: invalid format", name));
                    continue;
                }
            };

            // Check if we have data to restore
            if let Some(data) = ds_obj.get("data").and_then(|v| v.as_object()) {
                // Reconstruct DataFrame from stored columns
                let mut columns_vec: Vec<Column> = Vec::new();

                for (col_name, values) in data {
                    if let Some(arr) = values.as_array() {
                        // Try to determine column type from first non-null value
                        let first_non_null = arr.iter().find(|v| !v.is_null());

                        let series: Series = match first_non_null {
                            Some(serde_json::Value::Number(n)) if n.is_f64() => {
                                let vals: Vec<Option<f64>> = arr.iter()
                                    .map(|v| v.as_f64())
                                    .collect();
                                Series::new(col_name.into(), vals)
                            }
                            Some(serde_json::Value::Number(_)) => {
                                let vals: Vec<Option<i64>> = arr.iter()
                                    .map(|v| v.as_i64())
                                    .collect();
                                Series::new(col_name.into(), vals)
                            }
                            Some(serde_json::Value::Bool(_)) => {
                                let vals: Vec<Option<bool>> = arr.iter()
                                    .map(|v| v.as_bool())
                                    .collect();
                                Series::new(col_name.into(), vals)
                            }
                            _ => {
                                // Default to string
                                let vals: Vec<Option<String>> = arr.iter()
                                    .map(|v| {
                                        if v.is_null() { None }
                                        else if let Some(s) = v.as_str() { Some(s.to_string()) }
                                        else { Some(v.to_string()) }
                                    })
                                    .collect();
                                Series::new(col_name.into(), vals)
                            }
                        };
                        columns_vec.push(series.into());
                    }
                }

                if !columns_vec.is_empty() {
                    match DataFrame::new(columns_vec) {
                        Ok(df) => {
                            let dataset = p2a_core::Dataset::new(df);
                            datasets.insert(name.clone(), dataset);
                            imported_count += 1;
                        }
                        Err(e) => {
                            errors.push(format!("{}: DataFrame error - {}", name, e));
                        }
                    }
                } else {
                    errors.push(format!("{}: no column data found", name));
                }
            } else {
                errors.push(format!("{}: no data field (metadata-only session)", name));
            }
        }

        let mut output = format!(
            "Session Import\n{}\n\
             File: {}\n\
             Mode: {}\n\
             Datasets imported: {}\n",
            "=".repeat(40),
            request.file_path,
            if merge { "merge" } else { "replace" },
            imported_count
        );

        if !errors.is_empty() {
            output.push_str(&format!("\nErrors:\n{}", errors.join("\n")));
        }

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Set the global random seed for ML reproducibility.
    #[tool(description = "Set a global random seed for ML operations (kmeans, random_forest, tsne). When set, ML tools will use this seed as a fallback if no per-tool seed is specified. Clear by calling with no seed value.")]
    async fn set_seed(
        &self,
        Parameters(request): Parameters<SetSeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut global_seed = self.global_seed.write().await;
        *global_seed = request.seed;

        match request.seed {
            Some(seed) => Ok(CallToolResult::success(vec![Content::text(format!(
                "Global random seed set to: {}\n\
                 This seed will be used by ML tools (kmeans, random_forest, tsne) unless overridden per-tool.",
                seed
            ))])),
            None => Ok(CallToolResult::success(vec![Content::text(
                "Global random seed cleared. ML tools will use random initialization unless a per-tool seed is specified.".to_string()
            )])),
        }
    }

    /// Get the current global random seed.
    #[tool(description = "Get the current global random seed setting and list which ML tools support seeded reproducibility.")]
    async fn get_seed(
        &self,
        Parameters(_request): Parameters<GetSeedRequest>,
    ) -> Result<CallToolResult, McpError> {
        let global_seed = self.global_seed.read().await;

        let seed_status = match *global_seed {
            Some(seed) => format!("Current global seed: {}", seed),
            None => "No global seed set (using random initialization)".to_string(),
        };

        let output = format!(
            "Seed Management\n{}\n\
             {}\n\n\
             ML tools supporting reproducibility:\n\
             - ml_kmeans: Uses seed for centroid initialization\n\
             - ml_random_forest: Uses seed for bootstrap sampling and feature selection\n\
             - ml_tsne: Uses seed for initial embedding\n\n\
             Per-tool seeds override the global seed.",
            "=".repeat(40),
            seed_status
        );

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }
}

/// Helper function to extract numeric columns into an ndarray matrix.
fn extract_numeric_matrix(
    dataset: &Dataset,
    columns: &[String],
) -> Result<ndarray::Array2<f64>, String> {
    use p2a_core::polars::prelude::*;

    let df = dataset.df();
    let n_rows = df.height();
    let n_cols = columns.len();

    if columns.is_empty() {
        return Err("At least one column must be specified".to_string());
    }

    let mut data = ndarray::Array2::zeros((n_rows, n_cols));

    for (j, col_name) in columns.iter().enumerate() {
        let col = df.column(col_name)
            .map_err(|e| format!("Column '{}' not found: {}", col_name, e))?;

        let values: Vec<f64> = col.cast(&DataType::Float64)
            .map_err(|e| format!("Cannot convert column '{}' to numeric: {}", col_name, e))?
            .f64()
            .map_err(|e| format!("Column '{}' is not numeric: {}", col_name, e))?
            .into_iter()
            .map(|v: Option<f64>| v.unwrap_or(f64::NAN))
            .collect();

        for (i, &val) in values.iter().enumerate() {
            data[[i, j]] = val;
        }
    }

    Ok(data)
}

// ============================================================================
// ServerHandler Implementation
// ============================================================================

#[tool_handler(router = self.tool_router)]
impl ServerHandler for AnalyticsServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "prompt2analytics".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                ..Default::default()
            },
            instructions: Some(
                "prompt2analytics is a local data analytics engine. \
                 Use 'load_dataset' to load a CSV or Parquet file, or \
                 'db_sqlite_query'/'db_duckdb_query' to query databases. \
                 Then use 'describe_dataset' for summary statistics, \
                 'compute_correlation' for correlations, 'regression_ols' \
                 for linear regression, 'regression_diagnostics' for model validation, \
                 'panel_fixed_effects' or 'panel_random_effects' for panel data, \
                 'hausman_test' to choose between FE/RE, 'iv_2sls' for instrumental \
                 variables, 'diff_in_diff' for difference-in-differences, \
                 'logit' or 'probit' for binary outcomes, \
                 'ts_var' for VAR models, 'ts_varma' for VARMA models, \
                 'ts_vecm' for cointegration analysis, 'ts_var_irf' for impulse responses, \
                 'ml_kmeans' for K-means clustering, 'ml_dbscan' for DBSCAN clustering, \
                 'ml_pca' for principal component analysis, or visualization tools: \
                 'viz_histogram', 'viz_scatter', 'viz_line', 'viz_boxplot', 'viz_heatmap'."
                    .to_string(),
            ),
        }
    }
}
