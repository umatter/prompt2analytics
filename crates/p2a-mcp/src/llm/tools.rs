//! MCP tool definitions for LLM function calling.
//!
//! Provides the system prompt and tool definitions for the analytics assistant.

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
pub fn get_mcp_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // Data Loading & Management
        ToolDefinition {
            name: "load_dataset".to_string(),
            description: "Load a dataset from a file. Supports CSV, Parquet, Excel, Stata, and SAS formats.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the data file." },
                    "name": { "type": "string", "description": "Optional name for the dataset." }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "list_datasets".to_string(),
            description: "List all currently loaded datasets.".to_string(),
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolDefinition {
            name: "describe_dataset".to_string(),
            description: "Compute descriptive statistics for all columns in a dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "dataset": { "type": "string" } },
                "required": ["dataset"]
            }),
        },
        ToolDefinition {
            name: "head_dataset".to_string(),
            description: "Show the first N rows of a dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "n": { "type": "integer" }
                },
                "required": ["dataset"]
            }),
        },
        ToolDefinition {
            name: "compute_correlation".to_string(),
            description: "Compute the correlation matrix for numeric columns.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "dataset": { "type": "string" } },
                "required": ["dataset"]
            }),
        },
        // Regression Analysis
        ToolDefinition {
            name: "regression_ols".to_string(),
            description: "Run OLS regression. Returns coefficients, standard errors, t-values, p-values, R-squared.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "y": { "type": "string", "description": "Dependent variable column" },
                    "x": { "type": "array", "items": { "type": "string" }, "description": "Independent variable columns" }
                },
                "required": ["dataset", "y", "x"]
            }),
        },
        ToolDefinition {
            name: "regression_clustered".to_string(),
            description: "Run OLS with clustered standard errors.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" },
                    "cluster1": { "type": "string" },
                    "cluster2": { "type": "string" }
                },
                "required": ["dataset", "formula", "cluster1"]
            }),
        },
        ToolDefinition {
            name: "regression_diagnostics".to_string(),
            description: "Run regression diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" }
                },
                "required": ["dataset", "formula"]
            }),
        },
        // Panel Data
        ToolDefinition {
            name: "panel_fixed_effects".to_string(),
            description: "Run Fixed Effects panel regression.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" },
                    "entity_var": { "type": "string" }
                },
                "required": ["dataset", "formula", "entity_var"]
            }),
        },
        ToolDefinition {
            name: "panel_random_effects".to_string(),
            description: "Run Random Effects panel regression.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" },
                    "entity_var": { "type": "string" }
                },
                "required": ["dataset", "formula", "entity_var"]
            }),
        },
        ToolDefinition {
            name: "hausman_test".to_string(),
            description: "Run Hausman test to choose between Fixed and Random Effects.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" },
                    "entity_var": { "type": "string" }
                },
                "required": ["dataset", "formula", "entity_var"]
            }),
        },
        // Causal Inference
        ToolDefinition {
            name: "iv_2sls".to_string(),
            description: "Run Instrumental Variables (2SLS) regression.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "endog_formula": { "type": "string" },
                    "instrument_formula": { "type": "string" },
                    "robust": { "type": "boolean" }
                },
                "required": ["dataset", "endog_formula", "instrument_formula"]
            }),
        },
        ToolDefinition {
            name: "diff_in_diff".to_string(),
            description: "Run Difference-in-Differences estimation.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "dep_var": { "type": "string" },
                    "treatment_var": { "type": "string" },
                    "post_var": { "type": "string" }
                },
                "required": ["dataset", "dep_var", "treatment_var", "post_var"]
            }),
        },
        // Discrete Choice
        ToolDefinition {
            name: "logit".to_string(),
            description: "Run Logit regression for binary outcomes.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" }
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
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" }
                },
                "required": ["dataset", "formula"]
            }),
        },
        // Time Series
        ToolDefinition {
            name: "ts_arima_fit".to_string(),
            description: "Fit an ARIMA(p,d,q) model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" },
                    "p": { "type": "integer" },
                    "d": { "type": "integer" },
                    "q": { "type": "integer" }
                },
                "required": ["dataset", "column", "p", "d", "q"]
            }),
        },
        ToolDefinition {
            name: "ts_arima_forecast".to_string(),
            description: "Forecast with ARIMA model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" },
                    "p": { "type": "integer" },
                    "d": { "type": "integer" },
                    "q": { "type": "integer" },
                    "horizon": { "type": "integer" }
                },
                "required": ["dataset", "column", "p", "d", "q", "horizon"]
            }),
        },
        ToolDefinition {
            name: "ts_var".to_string(),
            description: "Run Vector Autoregression (VAR) model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "lags": { "type": "integer" }
                },
                "required": ["dataset", "columns", "lags"]
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
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "k": { "type": "integer" }
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
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "eps": { "type": "number" },
                    "min_samples": { "type": "integer" }
                },
                "required": ["dataset", "columns", "eps", "min_samples"]
            }),
        },
        ToolDefinition {
            name: "ml_pca".to_string(),
            description: "Run Principal Component Analysis.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "n_components": { "type": "integer" }
                },
                "required": ["dataset", "columns"]
            }),
        },
        // Visualizations
        ToolDefinition {
            name: "viz_histogram".to_string(),
            description: "Generate a histogram.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" },
                    "bins": { "type": "integer" }
                },
                "required": ["dataset", "column"]
            }),
        },
        ToolDefinition {
            name: "viz_scatter".to_string(),
            description: "Generate a scatter plot.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x_column": { "type": "string" },
                    "y_column": { "type": "string" }
                },
                "required": ["dataset", "x_column", "y_column"]
            }),
        },
        ToolDefinition {
            name: "viz_line".to_string(),
            description: "Generate a line chart.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x_column": { "type": "string" },
                    "y_columns": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["dataset", "x_column", "y_columns"]
            }),
        },
        ToolDefinition {
            name: "viz_boxplot".to_string(),
            description: "Generate a box plot.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["dataset", "columns"]
            }),
        },
        ToolDefinition {
            name: "viz_heatmap".to_string(),
            description: "Generate a correlation heatmap.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } }
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
        assert!(tools.len() >= 20, "Expected at least 20 tools, got {}", tools.len());

        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"load_dataset"));
        assert!(tool_names.contains(&"regression_ols"));
        assert!(tool_names.contains(&"viz_histogram"));
    }
}
