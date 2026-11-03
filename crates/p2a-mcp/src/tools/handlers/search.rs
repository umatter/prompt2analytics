//! Tool search handlers for dynamic tool discovery.
//!
//! These meta-tools let MCP clients (stdio clients and `/api/tools` callers) search all
//! routed tools. They are not offered in the LLM chat tool list (`llm::INTERNAL_TOOLS`,
//! design D3). The registry they read is generated from the tool router
//! (`tools::registry`), so a new tool is searchable once routed and must get a
//! `CATEGORY_TABLE` row (tests enforce it).
//!
//! # Workflow
//!
//! 1. The client calls `search_tools` with a natural language query
//! 2. The server returns matching tools with descriptions and relevance scores
//! 3. `tool_info` gives one tool's parameters and JSON input schema
//! 4. The client calls the selected tool directly
//!
//! # Example
//!
//! User: "Test for serial correlation in my panel data"
//! Client: search_tools(query="serial correlation test panel")
//! Server: Returns [regression_bgtest, ...]
//! Client: regression_bgtest(dataset="mydata", ...)

use rmcp::{
    ErrorData as McpError,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content},
    tool, tool_router,
};

use crate::server::AnalyticsServer;
use crate::tools::registry::{
    ToolCategory, category_counts, category_name, parse_category, registry, search_tools_scored,
};
use crate::tools::requests::search::{ListToolCategoriesRequest, SearchToolsRequest};

/// The most results `search_tools` returns.
const MAX_SEARCH_LIMIT: usize = 25;

/// At most `max` characters of a caller-supplied string, for an error message.
fn cut(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// A JSON schema's type, for the `Parameters:` list of `tool_info`: a `$ref`'s last path
/// segment, `array of <items type>`, a type name, the non-null members of a type list or an
/// `anyOf` joined with " | ", else `any`.
fn schema_type(schema: &serde_json::Value) -> String {
    use serde_json::Value;
    if let Some(r) = schema.get("$ref").and_then(Value::as_str) {
        return r.rsplit('/').next().unwrap_or(r).to_string();
    }
    match schema.get("type") {
        Some(Value::String(t)) if t == "array" => {
            let items = schema
                .get("items")
                .map(schema_type)
                .unwrap_or_else(|| "any".to_string());
            return format!("array of {items}");
        }
        Some(Value::String(t)) => return t.clone(),
        Some(Value::Array(ts)) => {
            let names: Vec<String> = ts
                .iter()
                .filter_map(Value::as_str)
                .filter(|t| *t != "null")
                .map(|t| {
                    if t == "array" {
                        let items = schema
                            .get("items")
                            .map(schema_type)
                            .unwrap_or_else(|| "any".to_string());
                        format!("array of {items}")
                    } else {
                        t.to_string()
                    }
                })
                .collect();
            if !names.is_empty() {
                return names.join(" | ");
            }
        }
        _ => {}
    }
    if let Some(any) = schema.get("anyOf").and_then(Value::as_array) {
        let names: Vec<String> = any
            .iter()
            .filter(|s| s.get("type").and_then(Value::as_str) != Some("null"))
            .map(schema_type)
            .collect();
        if !names.is_empty() {
            return names.join(" | ");
        }
    }
    "any".to_string()
}

/// The `Parameters:` section of `tool_info`: one line per property of `schema`, in map
/// order, with its type, whether it is required, and its description.
fn parameter_lines(schema: &serde_json::Value) -> String {
    let required: Vec<&str> = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .map(|r| r.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    let props = schema
        .get("properties")
        .and_then(serde_json::Value::as_object);
    let Some(props) = props.filter(|p| !p.is_empty()) else {
        return "Parameters: none\n".to_string();
    };
    let mut out = String::from("Parameters:\n");
    for (name, prop) in props {
        let need = if required.contains(&name.as_str()) {
            "required"
        } else {
            "optional"
        };
        out.push_str(&format!("  - {name} ({}, {need})", schema_type(prop)));
        if let Some(d) = prop.get("description").and_then(serde_json::Value::as_str) {
            out.push_str(&format!(": {d}"));
        }
        out.push('\n');
    }
    out
}

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
        let limit = match request.limit {
            None => 10,
            Some(0) => {
                return Ok(CallToolResult::error(vec![Content::text(format!(
                    "limit must be between 1 and {MAX_SEARCH_LIMIT} (got 0)"
                ))]));
            }
            Some(n) => n.min(MAX_SEARCH_LIMIT),
        };

        // Parse category filter
        let category = match request.category.as_deref() {
            None => None,
            Some(s) => match parse_category(s) {
                Some(c) => Some(c),
                None => {
                    let valid: Vec<&str> = ToolCategory::all()
                        .iter()
                        .map(|c| category_name(*c))
                        .collect();
                    return Ok(CallToolResult::error(vec![Content::text(format!(
                        "Unknown category '{}'. Valid categories: {}.",
                        cut(s, 64),
                        valid.join(", ")
                    ))]));
                }
            },
        };

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
            if let Some(r_fn) = result.tool.r_equivalent {
                output.push_str(&format!("   R counterpart: {}\n", r_fn));
            }
            if !result.tool.related.is_empty() {
                output.push_str(&format!("   Related: {}\n", result.tool.related.join(", ")));
            }
            output.push('\n');
        }

        output.push_str(&format!(
            "To use a tool, call it directly with its parameters; \
             tool_info(tool_name=\"{}\") lists them.",
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

        if let Some(c) = &counts {
            output.push_str(&format!("Total: {} tools\n", c.values().sum::<usize>()));
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
        description = "Get detailed information about a specific tool including its full description, parameters, JSON input schema, related tools, and R counterpart. Use this after search_tools to learn more about a specific tool before using it."
    )]
    pub async fn tool_info(
        &self,
        Parameters(request): Parameters<ToolInfoRequest>,
    ) -> Result<CallToolResult, McpError> {
        let tool_name = request.tool_name.to_lowercase();
        let tool = registry()
            .iter()
            .find(|t| t.name.to_lowercase() == tool_name);

        match tool {
            Some(t) => {
                let mut output = format!("Tool: {}\n", t.name);
                output.push_str(&"=".repeat(50));
                output.push_str("\n\n");
                output.push_str(&format!("Category: {:?}\n", t.category));
                output.push_str(&format!("Description: {}\n\n", t.description));

                if let Some(r_fn) = t.r_equivalent {
                    output.push_str(&format!("R counterpart: {}\n\n", r_fn));
                }

                if !t.related.is_empty() {
                    output.push_str("Related Tools:\n");
                    for related in t.related {
                        output.push_str(&format!("  - {}\n", related));
                    }
                    output.push('\n');
                }

                output.push_str(&parameter_lines(&t.input_schema));
                output.push_str("\nInput schema (JSON):\n");
                output.push_str(
                    &serde_json::to_string(&t.input_schema).unwrap_or_else(|_| "{}".to_string()),
                );
                output.push('\n');

                Ok(CallToolResult::success(vec![Content::text(output)]))
            }
            None => Ok(CallToolResult::error(vec![Content::text(format!(
                "Tool '{}' not found. Use search_tools to find available tools.",
                cut(&request.tool_name, 64)
            ))])),
        }
    }
}

/// Request for getting detailed tool information.
#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ToolInfoRequest {
    /// Name of the tool to get information about
    #[schemars(description = "Name of the tool")]
    pub tool_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::requests::search::{ListToolCategoriesRequest, SearchToolsRequest};

    fn text(r: &CallToolResult) -> String {
        r.content
            .iter()
            .filter_map(|c| c.raw.as_text())
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[tokio::test]
    async fn test_tool_info_returns_the_input_schema() {
        let server = AnalyticsServer::new();
        let r = server
            .tool_info(Parameters(ToolInfoRequest {
                tool_name: "Regression_OLS".into(),
            }))
            .await
            .unwrap();
        let t = text(&r);
        assert!(
            t.contains("Parameters:\n")
                && t.contains("  - dataset (string, required)")
                && t.contains("  - x (array of string, required)"),
            "{t}"
        );
        let json = t
            .split("Input schema (JSON):\n")
            .nth(1)
            .expect("schema section");
        let got: serde_json::Value = serde_json::from_str(json.trim()).unwrap();
        let want = server
            .router_tools()
            .into_iter()
            .find(|x| x.name == "regression_ols")
            .unwrap();
        assert_eq!(got, serde_json::Value::Object((*want.input_schema).clone()));
    }

    #[tokio::test]
    async fn test_tool_info_unknown_name_is_an_error() {
        let r = AnalyticsServer::new()
            .tool_info(Parameters(ToolInfoRequest {
                tool_name: "survival_cox".into(),
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("Tool 'survival_cox' not found"));
    }

    #[tokio::test]
    async fn test_search_tools_refuses_unknown_category_and_zero_limit() {
        let s = AnalyticsServer::new();
        let r = s
            .search_tools(Parameters(SearchToolsRequest {
                query: "test".into(),
                category: Some("survival_analysis".into()),
                limit: None,
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).starts_with(
            "Unknown category 'survival_analysis'. Valid categories: data, cleaning, munging,"
        ));
        let r = s
            .search_tools(Parameters(SearchToolsRequest {
                query: "test".into(),
                category: None,
                limit: Some(0),
            }))
            .await
            .unwrap();
        assert_eq!(r.is_error, Some(true));
        assert!(text(&r).contains("limit must be between 1 and 25 (got 0)"));
        let r = s
            .search_tools(Parameters(SearchToolsRequest {
                query: "kaplan".into(),
                category: Some("survival".into()),
                limit: None,
            }))
            .await
            .unwrap();
        assert!(text(&r).contains("**kaplan_meier**") && text(&r).contains("tool_info"));
    }

    #[tokio::test]
    async fn test_list_tool_categories_counts_the_router() {
        let t = text(
            &AnalyticsServer::new()
                .list_tool_categories(Parameters(ListToolCategoriesRequest {
                    include_counts: None,
                }))
                .await
                .unwrap(),
        );
        for c in ToolCategory::all() {
            let n = registry().iter().filter(|r| r.category == *c).count();
            assert!(
                t.contains(&format!("**{c:?}** ({n} tools)")),
                "{c:?} {n}\n{t}"
            );
        }
        // categories without optional tools: fixed counts (before: Spatial 5, MachineLearning 7)
        for (c, n) in [("Survival", 5), ("Spatial", 10), ("MachineLearning", 17)] {
            assert!(t.contains(&format!("**{c}** ({n} tools)")), "{c}");
        }
        assert!(t.contains(&format!("Total: {} tools", registry().len())));
    }
}
