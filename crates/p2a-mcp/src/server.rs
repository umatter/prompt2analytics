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
    data::{DataLoader, Dataset, DatasetInfo},
    regression::run_ols,
    stats::{correlation_matrix, DescriptiveStats},
    // Econometrics
    run_fixed_effects, run_random_effects, run_iv2sls, run_did,
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
    /// Path to the data file (CSV or Parquet)
    #[schemars(description = "Absolute or relative path to the data file. Supports CSV and Parquet formats.")]
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
    #[tool(description = "Load a dataset from a file. Supports CSV and Parquet formats. Returns dataset information including dimensions and column types.")]
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
                 Use 'load_dataset' to load a CSV or Parquet file. \
                 Then use 'describe_dataset' for summary statistics, \
                 'compute_correlation' for correlations, 'regression_ols' \
                 for linear regression, 'panel_fixed_effects' or 'panel_random_effects' \
                 for panel data, 'iv_2sls' for instrumental variables, or \
                 'diff_in_diff' for difference-in-differences causal analysis."
                    .to_string(),
            ),
        }
    }
}
