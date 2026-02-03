//! MCP tool definitions for LLM function calling.
//!
//! Provides the system prompt and tool definitions for the analytics assistant.

use super::ToolDefinition;
use serde_json::json;

/// Returns the base system prompt for the data analytics assistant.
fn get_base_system_prompt() -> &'static str {
    r#"You are a data analytics assistant for prompt2analytics. You help users analyze data by invoking specialized Rust-powered tools.

## CRITICAL RULES

1. **ALWAYS use tools for any statistical computation or analysis.** You have access to high-performance Rust implementations - USE THEM.

2. **NEVER perform calculations yourself.** Do not:
   - Calculate means, standard deviations, or any statistics manually
   - Estimate regression coefficients or p-values in your head
   - Make up or approximate numerical results
   - Describe what an analysis "would show" without running it

3. **When in doubt, use a tool.** If a user asks anything that could be answered by a tool, call that tool.

4. **ALWAYS USE EXISTING DATASETS.** Before creating a new dataset:
   - Check the "Currently Loaded Datasets" section below (if present) to see what data is already available
   - If a dataset with the data you need already exists, USE IT - do NOT call `create_dataset` again
   - Only call `create_dataset` if the user explicitly asks to create NEW data or no suitable dataset exists
   - When referencing a dataset in a tool call, use the EXACT name shown in the loaded datasets list

5. **Be aware of conversation context.** The user may be continuing a previous analysis:
   - Refer back to datasets, analyses, or results from earlier in the conversation
   - Don't repeat tool calls unnecessarily if the result is already available
   - Build on previous work rather than starting over

## AVAILABLE TOOLS BY CATEGORY

### Data Management
- `load_dataset` - Load CSV, Parquet, Excel (.xlsx/.xls), Stata (.dta), SAS (.sas7bdat)
- `create_dataset` - Create dataset from inline CSV (ONLY for NEW test/generated data - check existing datasets first!)
- `list_datasets` - List all loaded datasets
- `describe_dataset` - Descriptive statistics for all columns
- `head_dataset` - Preview first N rows

### Regression Analysis
- `regression_ols` - OLS with robust standard errors (HC0-HC3)
- `regression_clustered` - OLS with clustered standard errors
- `regression_diagnostics` - Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF

### Panel Data Econometrics
- `panel_fixed_effects` - Fixed Effects estimation
- `panel_random_effects` - Random Effects estimation
- `hausman_test` - Choose between FE and RE

### Causal Inference
- `iv_2sls` - Instrumental Variables (2SLS) with first-stage diagnostics
- `diff_in_diff` - Difference-in-Differences

### Discrete Choice Models
- `logit` - Logistic regression for binary outcomes
- `probit` - Probit regression for binary outcomes

### Time Series
- `ts_arima_fit` - Fit ARIMA(p,d,q) model
- `ts_arima_forecast` - Forecast with ARIMA
- `ts_var` - Vector Autoregression
- `ts_varma` - VARMA model
- `ts_vecm` - Vector Error Correction Model
- `ts_var_irf` - Impulse Response Functions
- `ts_mstl` - MSTL decomposition
- `ts_changepoint` - Changepoint detection

### Machine Learning
- `ml_kmeans` - K-means clustering
- `ml_dbscan` - DBSCAN clustering
- `ml_hierarchical` - Hierarchical clustering
- `ml_pca` - Principal Component Analysis
- `ml_tsne` - t-SNE dimensionality reduction
- `ml_random_forest` - Random Forest classification/regression
- `ml_svm` - Support Vector Machine

### Database Queries
- `db_sqlite_query` - Query SQLite database
- `db_sqlite_tables` - List SQLite tables
- `db_sqlite_schema` - Get SQLite table schema
- `db_duckdb_query` - Query DuckDB (can query Parquet/CSV directly)
- `db_duckdb_tables` - List DuckDB tables
- `db_duckdb_schema` - Get DuckDB table schema

### Visualization
- `viz_histogram` - Histogram
- `viz_scatter` - Scatter plot
- `viz_line` - Line chart
- `viz_boxplot` - Box plot
- `viz_heatmap` - Correlation heatmap
- `viz_coefficient` - Coefficient plot with confidence intervals
- `viz_residual_diagnostics` - Residual diagnostic plots
- `viz_event_study` - Event study plot
- `viz_irf` - Impulse response function plot
- `viz_dendrogram` - Hierarchical clustering dendrogram

### Statistics
- `compute_correlation` - Correlation matrix

## WORKFLOW

1. **Check existing datasets first** → Look at the "Currently Loaded Datasets" section to see what's available
2. **No dataset?** → Help user load one with `load_dataset` OR create sample data with `create_dataset`
3. **Dataset exists?** → USE IT directly - do NOT recreate it
4. **User asks for analysis?** → Find the matching tool and call it with the existing dataset
5. **Got results?** → Explain them in plain language, suggest follow-up analyses
6. **User wants visualization?** → Use the appropriate viz_* tool

## EXAMPLES OF CORRECT BEHAVIOR

User: "What's the average income in my dataset?"
✓ CORRECT: Call `describe_dataset` to get statistics
✗ WRONG: Try to calculate or estimate the average yourself

User: "Run a regression of price on sqft and bedrooms"
✓ CORRECT: Call `regression_ols` with y="price", x=["sqft", "bedrooms"]
✗ WRONG: Describe what regression would do without calling the tool

User: "Generate some test data for regression"
✓ CORRECT: Call `create_dataset` with actual CSV content
✗ WRONG: Just describe what data would look like

User: "Now run OLS on that data" (after data was already created)
✓ CORRECT: Call `regression_ols` using the EXISTING dataset name
✗ WRONG: Call `create_dataset` again to recreate the same data

User: "Is there heteroskedasticity in my model?"
✓ CORRECT: Call `regression_diagnostics` which includes Breusch-Pagan test
✗ WRONG: Speculate about heteroskedasticity without testing

Remember: Your value is in orchestrating these powerful Rust tools, not in doing mental math. The tools are fast, accurate, and provide publication-quality output. USE THEM. Always check what datasets are already loaded before creating new ones."#
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
            format!("{}\n\n## Currently Loaded Datasets\n\n{}", base, context)
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
            name: "create_dataset".to_string(),
            description: "Create a dataset from inline CSV content. Use this to create datasets on-the-fly from generated or inline data without needing a file.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Name for the dataset (e.g., 'my_data')" },
                    "csv_content": { "type": "string", "description": "CSV content with headers in first row (e.g., 'x,y\\n1,2\\n3,4')" }
                },
                "required": ["name", "csv_content"]
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
        // Statistical Tests
        ToolDefinition {
            name: "t_test".to_string(),
            description: "Run Student's t-test. Supports one-sample, two-sample (Welch's), and paired t-tests.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x": { "type": "string", "description": "First column" },
                    "y": { "type": "string", "description": "Second column (for two-sample/paired)" },
                    "mu": { "type": "number", "description": "Hypothesized mean (for one-sample)" },
                    "paired": { "type": "boolean", "description": "Whether paired test" },
                    "alternative": { "type": "string", "description": "two.sided, less, or greater" }
                },
                "required": ["dataset", "x"]
            }),
        },
        ToolDefinition {
            name: "anova".to_string(),
            description: "Run one-way ANOVA to test whether means differ across groups.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "response": { "type": "string", "description": "Response variable" },
                    "group": { "type": "string", "description": "Grouping variable" }
                },
                "required": ["dataset", "response", "group"]
            }),
        },
        ToolDefinition {
            name: "shapiro_wilk".to_string(),
            description: "Run Shapiro-Wilk test for normality.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" }
                },
                "required": ["dataset", "column"]
            }),
        },
        ToolDefinition {
            name: "chi_squared_test".to_string(),
            description: "Run chi-squared test of independence for two categorical variables.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x": { "type": "string" },
                    "y": { "type": "string" }
                },
                "required": ["dataset", "x", "y"]
            }),
        },
        ToolDefinition {
            name: "wilcoxon_test".to_string(),
            description: "Run Wilcoxon non-parametric test (Mann-Whitney U for two-sample).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x": { "type": "string" },
                    "y": { "type": "string" },
                    "paired": { "type": "boolean" }
                },
                "required": ["dataset", "x"]
            }),
        },
        ToolDefinition {
            name: "cor_test".to_string(),
            description: "Test for correlation between two variables (Pearson, Spearman, or Kendall).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "x": { "type": "string" },
                    "y": { "type": "string" },
                    "method": { "type": "string", "description": "pearson, spearman, or kendall" }
                },
                "required": ["dataset", "x", "y"]
            }),
        },
        // Database Tools
        ToolDefinition {
            name: "db_sqlite_query".to_string(),
            description: "Execute SQL query on a SQLite database and load result as dataset.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to SQLite database file" },
                    "query": { "type": "string", "description": "SQL query to execute" },
                    "name": { "type": "string", "description": "Name for result dataset" }
                },
                "required": ["path", "query", "name"]
            }),
        },
        ToolDefinition {
            name: "db_duckdb_query".to_string(),
            description: "Execute SQL query using DuckDB (can query Parquet/CSV files directly).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to database or data file" },
                    "query": { "type": "string", "description": "SQL query" },
                    "name": { "type": "string", "description": "Name for result dataset" }
                },
                "required": ["path", "query", "name"]
            }),
        },
        // Additional Time Series
        ToolDefinition {
            name: "ts_mstl".to_string(),
            description: "Run MSTL decomposition (Multiple Seasonal-Trend decomposition using LOESS).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" },
                    "periods": { "type": "array", "items": { "type": "integer" } }
                },
                "required": ["dataset", "column", "periods"]
            }),
        },
        ToolDefinition {
            name: "ts_var_irf".to_string(),
            description: "Compute Impulse Response Functions for VAR model.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "lags": { "type": "integer" },
                    "horizon": { "type": "integer" }
                },
                "required": ["dataset", "columns", "lags", "horizon"]
            }),
        },
        ToolDefinition {
            name: "acf".to_string(),
            description: "Compute autocorrelation or partial autocorrelation function.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "column": { "type": "string" },
                    "lag_max": { "type": "integer" },
                    "type": { "type": "string", "description": "correlation, covariance, or partial" }
                },
                "required": ["dataset", "column"]
            }),
        },
        // Additional ML
        ToolDefinition {
            name: "ml_hierarchical".to_string(),
            description: "Run hierarchical clustering (Ward, single, complete, average linkage).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "method": { "type": "string", "description": "ward, single, complete, or average" },
                    "k": { "type": "integer", "description": "Number of clusters to cut" }
                },
                "required": ["dataset", "columns"]
            }),
        },
        ToolDefinition {
            name: "ml_tsne".to_string(),
            description: "Run t-SNE for dimensionality reduction and visualization.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "columns": { "type": "array", "items": { "type": "string" } },
                    "n_components": { "type": "integer" },
                    "perplexity": { "type": "number" }
                },
                "required": ["dataset", "columns"]
            }),
        },
        ToolDefinition {
            name: "ml_random_forest".to_string(),
            description: "Train Random Forest for classification or regression.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "target": { "type": "string" },
                    "features": { "type": "array", "items": { "type": "string" } },
                    "n_trees": { "type": "integer" },
                    "max_depth": { "type": "integer" }
                },
                "required": ["dataset", "target", "features"]
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
        ToolDefinition {
            name: "viz_coefficient".to_string(),
            description: "Generate a coefficient plot with confidence intervals from regression results.".to_string(),
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
            name: "viz_residual_diagnostics".to_string(),
            description: "Generate residual diagnostic plots (Q-Q, residuals vs fitted, etc.).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "dataset": { "type": "string" },
                    "formula": { "type": "string" }
                },
                "required": ["dataset", "formula"]
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
        assert!(
            tools.len() >= 20,
            "Expected at least 20 tools, got {}",
            tools.len()
        );

        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"load_dataset"));
        assert!(tool_names.contains(&"regression_ols"));
        assert!(tool_names.contains(&"viz_histogram"));
    }
}
