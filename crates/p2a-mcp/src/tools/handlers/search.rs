//! Tool search handlers for dynamic tool discovery.
//!
//! These meta-tools enable LLMs to discover and use tools from the full library,
//! solving the 128 tool limit imposed by most LLM APIs.
//!
//! # Workflow
//!
//! 1. LLM calls `search_tools` with a natural language query
//! 2. Server returns matching tools with descriptions and parameters
//! 3. LLM selects the appropriate tool and calls it directly (or via `execute_tool`)
//!
//! # Example
//!
//! User: "Test for serial correlation in my panel data"
//! LLM: search_tools(query="serial correlation test panel")
//! Server: Returns [regression_bgtest, timeseries_box_test, ...]
//! LLM: regression_bgtest(dataset="mydata", ...)

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::registry::{ToolCategory, category_counts, parse_category, search_tools_scored};
use crate::tools::requests::search::{
    ExecuteToolRequest, ListToolCategoriesRequest, SearchToolsRequest,
};

#[tool_router(router = search_router, vis = "pub")]
impl AnalyticsServer {
    // ========================================================================
    // Tool Discovery
    // ========================================================================

    /// Search for analytics tools by natural language description.
    #[tool(
        description = "Search for analytics tools by natural language description. Use this when you need to find the right tool for an analysis task. Returns matching tools with descriptions and relevance scores. Example queries: 'regression with robust standard errors', 'test for serial correlation', 'panel data fixed effects'."
    )]
    pub async fn search_tools(
        &self,
        Parameters(request): Parameters<SearchToolsRequest>,
    ) -> Result<CallToolResult, McpError> {
        let limit = request.limit.unwrap_or(10).min(25);

        // Parse category filter
        let category = request.category.as_ref().and_then(|s| parse_category(s));

        // Search with scoring
        let results = search_tools_scored(&request.query, category, limit);

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No tools found matching '{}'. Try:\n\
                 - Using different keywords\n\
                 - Removing the category filter\n\
                 - Describing what you want to achieve",
                request.query
            ))]));
        }

        // Format results
        let mut output = format!(
            "Found {} tools matching '{}':\n\n",
            results.len(),
            request.query
        );

        for (i, result) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. **{}** (score: {:.1})\n",
                i + 1,
                result.tool.name,
                result.score
            ));
            output.push_str(&format!("   Category: {:?}\n", result.tool.category));
            output.push_str(&format!("   {}\n", result.tool.description));
            if let Some(r_equiv) = result.tool.r_equivalent {
                output.push_str(&format!("   R equivalent: {}\n", r_equiv));
            }
            if !result.tool.related.is_empty() {
                output.push_str(&format!("   Related: {}\n", result.tool.related.join(", ")));
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "To use a tool, call it directly with the appropriate parameters.\n\
             Example: {}(dataset=\"your_data\", ...)",
            results[0].tool.name
        ));

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// List all available tool categories with descriptions.
    #[tool(
        description = "List all available tool categories with descriptions and tool counts. Use this to understand what types of analyses are available before searching for specific tools."
    )]
    pub async fn list_tool_categories(
        &self,
        Parameters(request): Parameters<ListToolCategoriesRequest>,
    ) -> Result<CallToolResult, McpError> {
        let include_counts = request.include_counts.unwrap_or(true);
        let counts = if include_counts {
            Some(category_counts())
        } else {
            None
        };

        let mut output = String::from("Available Tool Categories\n");
        output.push_str(&"=".repeat(50));
        output.push_str("\n\n");

        for category in ToolCategory::all() {
            let count_str = if let Some(ref c) = counts {
                format!(" ({} tools)", c.get(category).unwrap_or(&0))
            } else {
                String::new()
            };

            output.push_str(&format!(
                "**{:?}**{}\n  {}\n\n",
                category,
                count_str,
                category.description()
            ));
        }

        output.push_str("\nTo search within a category, use:\n");
        output.push_str("  search_tools(query=\"your query\", category=\"category_name\")\n\n");
        output.push_str("Category aliases:\n");
        output.push_str("  - stats, statistics\n");
        output.push_str("  - ts, timeseries, time_series\n");
        output.push_str("  - ml, machine_learning\n");
        output.push_str("  - viz, visualization\n");
        output.push_str("  - db, database\n");

        Ok(CallToolResult::success(vec![Content::text(output)]))
    }

    /// Get detailed information about a specific tool.
    #[tool(
        description = "Get detailed information about a specific tool including its full description, parameters, related tools, and R equivalent. Use this after search_tools to learn more about a specific tool before using it."
    )]
    pub async fn tool_info(
        &self,
        Parameters(request): Parameters<ToolInfoRequest>,
    ) -> Result<CallToolResult, McpError> {
        use crate::tools::registry::get_registry;

        let tool_name = request.tool_name.to_lowercase();
        let registry = get_registry();

        let tool = registry.iter().find(|t| t.name.to_lowercase() == tool_name);

        match tool {
            Some(t) => {
                let mut output = format!("Tool: {}\n", t.name);
                output.push_str(&"=".repeat(50));
                output.push_str("\n\n");
                output.push_str(&format!("Category: {:?}\n", t.category));
                output.push_str(&format!("Description: {}\n\n", t.description));

                if let Some(r_equiv) = t.r_equivalent {
                    output.push_str(&format!("R Equivalent: {}\n\n", r_equiv));
                }

                if !t.related.is_empty() {
                    output.push_str("Related Tools:\n");
                    for related in t.related {
                        output.push_str(&format!("  - {}\n", related));
                    }
                }

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            None => Ok(CallToolResult::error(vec![Content::text(format!(
                "Tool '{}' not found. Use search_tools to find available tools.",
                request.tool_name
            ))])),
        }
    }
}

/// Request for getting detailed tool information.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ToolInfoRequest {
    /// Name of the tool to get information about
    #[schemars(description = "Name of the tool")]
    pub tool_name: String,
}
