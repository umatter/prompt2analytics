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
    },
    regression::{run_ols, run_ols_clustered, run_diagnostics},
    stats::{correlation_matrix, DescriptiveStats},
    // Econometrics
    run_fixed_effects, run_random_effects, run_hausman_test, run_iv2sls, run_did,
    run_logit, run_probit, run_first_stage_diagnostics,
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

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'y ~ x1 + x2').")]
    pub formula: String,
}

/// Request for OLS with clustered standard errors.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct OlsClusteredRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'y ~ x1 + x2').")]
    pub formula: String,

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

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'y ~ x1 + x2').")]
    pub formula: String,

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

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'y ~ x1 + x2').")]
    pub formula: String,

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

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'y ~ x1 + x2').")]
    pub formula: String,

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

    /// Endogenous model formula (e.g., "y ~ x1 + x2 + endog_var")
    #[schemars(description = "Formula for the structural equation (e.g., 'wage ~ experience + education').")]
    pub endog_formula: String,

    /// Instrument formula (e.g., "endog_var ~ z1 + z2")
    #[schemars(description = "Formula for instruments (e.g., 'education ~ parents_edu + distance_to_college').")]
    pub instrument_formula: String,

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

/// Request for Logit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'outcome ~ treatment + control_var').")]
    pub formula: String,
}

/// Request for Probit regression.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProbitRequest {
    /// Name/ID of the dataset
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,

    /// R-style formula (e.g., "y ~ x1 + x2")
    #[schemars(description = "R-style formula specifying the model (e.g., 'outcome ~ treatment + control_var').")]
    pub formula: String,
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

// ============================================================================
// Tool Router Implementation
// ============================================================================

#[tool_router]
impl AnalyticsServer {
    /// Create a new AnalyticsServer instance.
    pub fn new() -> Self {
        Self {
            datasets: Arc::new(RwLock::new(HashMap::new())),
            tool_router: Self::tool_router(),
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

        let result = match run_ols(dataset, &request.y, &x_refs) {
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

        let result = match run_diagnostics(dataset, &request.formula) {
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

        let result = match run_ols_clustered(
            dataset,
            &request.formula,
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

        let result = match run_fixed_effects(dataset, &request.formula, &request.entity_var) {
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

        let result = match run_random_effects(dataset, &request.formula, &request.entity_var) {
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

        let result = match run_hausman_test(dataset, &request.formula, &request.entity_var) {
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

        let result = match run_iv2sls(dataset, &request.endog_formula, &request.instrument_formula, robust) {
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

        let result = match run_did(dataset, &request.dep_var, &request.treatment_var, &request.post_var) {
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

        let result = match run_logit(dataset, &request.formula) {
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

        let result = match run_probit(dataset, &request.formula) {
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

        let result = match kmeans(
            data.view(),
            request.k,
            request.max_iterations,
            None, // tolerance
            request.n_init,
            request.seed,
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

        let result = match tsne(
            data.view(),
            request.n_components,
            request.perplexity,
            request.max_iterations,
            request.learning_rate,
            request.seed,
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

        let result = match random_forest(
            data.view(),
            target.view(),
            request.n_trees,
            request.max_depth,
            request.min_samples_split,
            request.max_features.as_deref(),
            request.seed,
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

                            match run_ols(dataset, y_col, &x_cols) {
                                Ok(ols_result) => {
                                    let mut output = format!("Dataset: {}\n{:-<60}\n", ds_name, "");
                                    output.push_str(&format!("R²: {:.4}, Adj R²: {:.4}\n", ols_result.r_squared, ols_result.adj_r_squared));
                                    output.push_str(&format!("F-stat: {:.4}\n", ols_result.f_statistic));
                                    output.push_str(&format!("Intercept: {:.4}\n", ols_result.intercept));
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
