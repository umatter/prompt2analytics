//! MCP tool definitions for LLM function calling.
//!
//! This module provides the system prompt and tool definitions that are passed
//! to LLM providers to enable them to invoke p2a-mcp analytics tools.

use super::ToolDefinition;
use serde_json::json;

/// Returns the base system prompt for the data analytics assistant.
fn get_base_system_prompt() -> &'static str {
    r#"You are a data analytics assistant for prompt2analytics. You help users analyze data by invoking specialized tools.

Available capabilities:
- Load datasets from CSV, Parquet, Excel, Stata, or SAS files
- Query SQLite and DuckDB databases
- Compute descriptive statistics and correlations
- Run regression analyses (OLS, Panel FE/RE, 2SLS, DiD, Logit/Probit)
- Time series modeling (ARIMA, VAR, VECM)
- Machine learning (K-means, DBSCAN, PCA)
- Generate visualizations (histograms, scatter plots, line charts, box plots, heatmaps)

When a user asks for analysis:
1. If no dataset is loaded, help them load one first
2. Choose the appropriate tool(s) for the analysis
3. Execute tools and interpret results clearly
4. Suggest follow-up analyses when relevant

Always explain statistical results in plain language alongside technical output."#
}

/// Returns the system prompt for the data analytics assistant.
pub fn get_system_prompt() -> String {
    get_base_system_prompt().to_string()
}

/// Returns the system prompt with dataset context included.
///
/// When datasets are loaded, includes their names and columns in the prompt
/// so the LLM knows what data is available for analysis.
pub fn get_system_prompt_with_context(dataset_context: Option<&str>) -> String {
    let base = get_base_system_prompt();

    match dataset_context {
        Some(context) if !context.is_empty() => {
            format!(
                "{}\n\n## Currently Loaded Datasets\n\n{}",
                base, context
            )
        }
        _ => base.to_string(),
    }
}

/// Returns the complete set of MCP tool definitions for LLM function calling.
/// These definitions match the tools exposed by the p2a-mcp server.
pub fn get_mcp_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // Data Loading & Management
        ToolDefinition {
            name: "load_dataset".to_string(),
            description: "Load a dataset from a file. Supports CSV, Parquet, Excel (xlsx, xls, xlsb, ods), Stata (dta), and SAS (sas7bdat) formats.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the data file."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name to identify this dataset. If not provided, the filename will be used."
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "list_datasets".to_string(),
            description: "List all currently loaded datasets with their basic information.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "describe_dataset".to_string(),
            description: "Compute descriptive statistics for all columns in a dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    }
                },
                "required": ["dataset"]
            }),
        },
        ToolDefinition {
            name: "head_dataset".to_string(),
            description: "Show the first N rows of a dataset. Default is 5 rows.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to return. Default is 5."
                    }
                },
                "required": ["dataset"]
            }),
        },
        ToolDefinition {
            name: "compute_correlation".to_string(),
            description: "Compute the Pearson correlation matrix for all numeric columns in a dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    }
                },
                "required": ["dataset"]
            }),
        },

        // Database Queries
        ToolDefinition {
            name: "db_sqlite_tables".to_string(),
            description: "List all tables in a SQLite database.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the SQLite database file."
                    }
                },
                "required": ["db_path"]
            }),
        },
        ToolDefinition {
            name: "db_sqlite_schema".to_string(),
            description: "Get the schema for a table in a SQLite database.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the SQLite database file."
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to get schema for."
                    }
                },
                "required": ["db_path", "table_name"]
            }),
        },
        ToolDefinition {
            name: "db_sqlite_query".to_string(),
            description: "Execute a SQL query against a SQLite database and load the results as a dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the SQLite database file."
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the resulting dataset."
                    }
                },
                "required": ["db_path", "query"]
            }),
        },
        ToolDefinition {
            name: "db_duckdb_tables".to_string(),
            description: "List all tables in a DuckDB database.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the DuckDB database file."
                    }
                },
                "required": ["db_path"]
            }),
        },
        ToolDefinition {
            name: "db_duckdb_schema".to_string(),
            description: "Get the schema for a table in a DuckDB database.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the DuckDB database file."
                    },
                    "table_name": {
                        "type": "string",
                        "description": "Name of the table to get schema for."
                    }
                },
                "required": ["db_path", "table_name"]
            }),
        },
        ToolDefinition {
            name: "db_duckdb_query".to_string(),
            description: "Execute a SQL query against a DuckDB database and load the results as a dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the DuckDB database file."
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute."
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional name for the resulting dataset."
                    }
                },
                "required": ["db_path", "query"]
            }),
        },

        // Regression Analysis
        ToolDefinition {
            name: "regression_ols".to_string(),
            description: "Run Ordinary Least Squares (OLS) regression. Returns coefficients, standard errors, t-values, p-values, R-squared, and F-statistic.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "y": {
                        "type": "string",
                        "description": "Name of the dependent variable (Y) column."
                    },
                    "x": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Names of the independent variable (X) columns."
                    }
                },
                "required": ["dataset", "y", "x"]
            }),
        },
        ToolDefinition {
            name: "regression_clustered".to_string(),
            description: "Run OLS regression with clustered standard errors. Supports one-way or two-way clustering.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model (e.g., 'y ~ x1 + x2')."
                    },
                    "cluster1": {
                        "type": "string",
                        "description": "Column name for first clustering dimension."
                    },
                    "cluster2": {
                        "type": "string",
                        "description": "Optional column for second clustering dimension."
                    }
                },
                "required": ["dataset", "formula", "cluster1"]
            }),
        },
        ToolDefinition {
            name: "regression_diagnostics".to_string(),
            description: "Run comprehensive regression diagnostics including Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF, and condition number.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model."
                    }
                },
                "required": ["dataset", "formula"]
            }),
        },
        ToolDefinition {
            name: "panel_fixed_effects".to_string(),
            description: "Run Fixed Effects (within) panel regression. Controls for time-invariant unobserved heterogeneity.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model."
                    },
                    "entity_var": {
                        "type": "string",
                        "description": "Column name for entity/individual identifier."
                    }
                },
                "required": ["dataset", "formula", "entity_var"]
            }),
        },
        ToolDefinition {
            name: "panel_random_effects".to_string(),
            description: "Run Random Effects (GLS) panel regression.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model."
                    },
                    "entity_var": {
                        "type": "string",
                        "description": "Column name for entity/individual identifier."
                    }
                },
                "required": ["dataset", "formula", "entity_var"]
            }),
        },
        ToolDefinition {
            name: "hausman_test".to_string(),
            description: "Run Hausman specification test to choose between Fixed Effects and Random Effects.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model."
                    },
                    "entity_var": {
                        "type": "string",
                        "description": "Column name for entity/individual identifier."
                    }
                },
                "required": ["dataset", "formula", "entity_var"]
            }),
        },
        ToolDefinition {
            name: "iv_2sls".to_string(),
            description: "Run Instrumental Variables (2SLS) regression.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "endog_formula": {
                        "type": "string",
                        "description": "Formula for the structural equation."
                    },
                    "instrument_formula": {
                        "type": "string",
                        "description": "Formula for instruments."
                    },
                    "robust": {
                        "type": "boolean",
                        "description": "Whether to use heteroskedasticity-robust standard errors."
                    }
                },
                "required": ["dataset", "endog_formula", "instrument_formula"]
            }),
        },
        ToolDefinition {
            name: "iv_first_stage".to_string(),
            description: "Run first-stage diagnostics to test instrument strength.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "endogenous_var": {
                        "type": "string",
                        "description": "Name of the endogenous variable."
                    },
                    "instruments": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Names of the instrumental variables."
                    },
                    "controls": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional control variables."
                    }
                },
                "required": ["dataset", "endogenous_var", "instruments"]
            }),
        },
        ToolDefinition {
            name: "diff_in_diff".to_string(),
            description: "Run Difference-in-Differences (DiD) estimation.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "dep_var": {
                        "type": "string",
                        "description": "Name of the outcome/dependent variable column."
                    },
                    "treatment_var": {
                        "type": "string",
                        "description": "Column indicating treatment group (1 = treated, 0 = control)."
                    },
                    "post_var": {
                        "type": "string",
                        "description": "Column indicating post-treatment period (1 = post, 0 = pre)."
                    }
                },
                "required": ["dataset", "dep_var", "treatment_var", "post_var"]
            }),
        },
        ToolDefinition {
            name: "logit".to_string(),
            description: "Run Logit (logistic) regression for binary outcomes.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model."
                    }
                },
                "required": ["dataset", "formula"]
            }),
        },
        ToolDefinition {
            name: "probit".to_string(),
            description: "Run Probit regression for binary outcomes.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "formula": {
                        "type": "string",
                        "description": "R-style formula specifying the model."
                    }
                },
                "required": ["dataset", "formula"]
            }),
        },

        // Time Series
        ToolDefinition {
            name: "ts_arima_fit".to_string(),
            description: "Fit an ARIMA(p,d,q) model to a univariate time series.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "column": {
                        "type": "string",
                        "description": "Name of the column containing the time series values."
                    },
                    "p": {
                        "type": "integer",
                        "description": "Number of autoregressive (AR) terms."
                    },
                    "d": {
                        "type": "integer",
                        "description": "Number of differences."
                    },
                    "q": {
                        "type": "integer",
                        "description": "Number of moving average (MA) terms."
                    }
                },
                "required": ["dataset", "column", "p", "d", "q"]
            }),
        },
        ToolDefinition {
            name: "ts_arima_forecast".to_string(),
            description: "Forecast future values using an ARIMA model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "column": {
                        "type": "string",
                        "description": "Name of the column containing the time series values."
                    },
                    "p": { "type": "integer" },
                    "d": { "type": "integer" },
                    "q": { "type": "integer" },
                    "horizon": {
                        "type": "integer",
                        "description": "Number of periods to forecast ahead."
                    }
                },
                "required": ["dataset", "column", "p", "d", "q", "horizon"]
            }),
        },
        ToolDefinition {
            name: "ts_mstl".to_string(),
            description: "Perform MSTL decomposition on a time series.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "column": {
                        "type": "string",
                        "description": "Name of the column containing the time series values."
                    },
                    "periods": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "Seasonal periods to extract."
                    }
                },
                "required": ["dataset", "column", "periods"]
            }),
        },
        ToolDefinition {
            name: "ts_var".to_string(),
            description: "Run Vector Autoregression (VAR) model for multivariate time series.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": {
                        "type": "string",
                        "description": "Name or ID of a previously loaded dataset."
                    },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Names of the columns to include in the VAR model."
                    },
                    "lags": {
                        "type": "integer",
                        "description": "Number of lags."
                    }
                },
                "required": ["dataset", "columns", "lags"]
            }),
        },
        ToolDefinition {
            name: "ts_varma".to_string(),
            description: "Run VARMA(p,q) model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "p": { "type": "integer" },
                    "q": { "type": "integer" }
                },
                "required": ["dataset", "columns", "p", "q"]
            }),
        },
        ToolDefinition {
            name: "ts_vecm".to_string(),
            description: "Run VECM for cointegrated time series.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "lags": { "type": "integer" },
                    "rank": { "type": "integer" }
                },
                "required": ["dataset", "columns", "lags", "rank"]
            }),
        },
        ToolDefinition {
            name: "ts_var_irf".to_string(),
            description: "Compute Impulse Response Functions from a VAR model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "lags": { "type": "integer" },
                    "steps": { "type": "integer" }
                },
                "required": ["dataset", "columns", "lags", "steps"]
            }),
        },

        // Machine Learning
        ToolDefinition {
            name: "ml_kmeans".to_string(),
            description: "Run K-means clustering.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "k": { "type": "integer", "description": "Number of clusters." },
                    "max_iterations": { "type": "integer" },
                    "n_init": { "type": "integer" },
                    "seed": { "type": "integer" }
                },
                "required": ["dataset", "columns", "k"]
            }),
        },
        ToolDefinition {
            name: "ml_dbscan".to_string(),
            description: "Run DBSCAN clustering.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "eps": { "type": "number" },
                    "min_samples": { "type": "integer" }
                },
                "required": ["dataset", "columns", "eps", "min_samples"]
            }),
        },
        ToolDefinition {
            name: "ml_pca".to_string(),
            description: "Run Principal Component Analysis (PCA).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "n_components": { "type": "integer" }
                },
                "required": ["dataset", "columns"]
            }),
        },

        // Visualizations
        ToolDefinition {
            name: "viz_histogram".to_string(),
            description: "Generate a histogram visualization.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" },
                    "bins": { "type": "integer" },
                    "title": { "type": "string" }
                },
                "required": ["dataset", "column"]
            }),
        },
        ToolDefinition {
            name: "viz_scatter".to_string(),
            description: "Generate a scatter plot visualization.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x_column": { "type": "string" },
                    "y_column": { "type": "string" },
                    "title": { "type": "string" }
                },
                "required": ["dataset", "x_column", "y_column"]
            }),
        },
        ToolDefinition {
            name: "viz_line".to_string(),
            description: "Generate a line chart visualization.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x_column": { "type": "string" },
                    "y_columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "title": { "type": "string" }
                },
                "required": ["dataset", "x_column", "y_columns"]
            }),
        },
        ToolDefinition {
            name: "viz_boxplot".to_string(),
            description: "Generate a box plot visualization.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "title": { "type": "string" }
                },
                "required": ["dataset", "columns"]
            }),
        },
        ToolDefinition {
            name: "viz_heatmap".to_string(),
            description: "Generate a correlation heatmap visualization.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "title": { "type": "string" }
                },
                "required": ["dataset"]
            }),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_prompt_not_empty() {
        let prompt = get_system_prompt();
        assert!(!prompt.is_empty());
        assert!(prompt.contains("data analytics"));
    }

    #[test]
    fn test_tool_definitions_complete() {
        let tools = get_mcp_tool_definitions();
        // We should have ~38 tools matching the MCP server
        assert!(tools.len() >= 30, "Expected at least 30 tools, got {}", tools.len());
        
        // Check some key tools exist
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"load_dataset"));
        assert!(tool_names.contains(&"regression_ols"));
        assert!(tool_names.contains(&"viz_histogram"));
    }
}
