//! Request types for tool search functionality.
//!
//! These tools allow LLMs to discover tools from the full library dynamically,
//! enabling scaling beyond the 128 tool limit of most LLM APIs.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for searching tools by natural language query.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchToolsRequest {
    /// Natural language description of the analysis you want to perform.
    /// Examples: "regression with robust standard errors", "test for unit root",
    /// "compare treatment and control groups"
    #[schemars(description = "Natural language description of the analysis task")]
    pub query: String,

    /// Optional category filter. If specified, only returns tools from this category.
    /// Valid categories: data, cleaning, munging, descriptive, statistics, regression,
    /// panel, iv, did, rd, matching, treatment, mediation, discrete, timeseries,
    /// spatial, survival, ml, visualization, database, utility
    #[schemars(description = "Optional category to filter results")]
    pub category: Option<String>,

    /// Maximum number of tools to return (default: 10, max: 25)
    #[schemars(description = "Maximum tools to return (default: 10)")]
    pub limit: Option<usize>,
}

/// Request for executing a tool by name with dynamic dispatch.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExecuteToolRequest {
    /// Name of the tool to execute (from search_tools results)
    #[schemars(description = "Name of the tool to execute")]
    pub tool_name: String,

    /// Tool arguments as a JSON object. Structure depends on the specific tool.
    #[schemars(description = "Tool arguments as JSON object")]
    pub arguments: serde_json::Value,
}

/// Request for listing tool categories.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListToolCategoriesRequest {
    /// If true, include tool counts per category (default: true)
    #[schemars(description = "Include tool counts (default: true)")]
    pub include_counts: Option<bool>,
}
