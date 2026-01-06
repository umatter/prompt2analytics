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
    // Machine Learning
    kmeans, dbscan, pca,
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
                 or 'ml_pca' for principal component analysis."
                    .to_string(),
            ),
        }
    }
}
