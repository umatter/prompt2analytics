//! Request types for tool search functionality.
//!
//! These tools allow LLMs to discover tools from the full library dynamically,
//! enabling scaling beyond the 128 tool limit of most LLM APIs.

use schemars::JsonSchema;
use serde::Deserialize;

/// Request for searching tools by natural language query.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
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

    /// Maximum number of tools to return (1 to 25, default: 10)
    #[schemars(
        description = "Maximum tools to return, 1 to 25 (default: 10); larger values return 25"
    )]
    pub limit: Option<usize>,
}

/// Request for listing tool categories.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListToolCategoriesRequest {
    /// If true, include tool counts per category (default: true)
    #[schemars(description = "Include tool counts (default: true)")]
    pub include_counts: Option<bool>,
}
