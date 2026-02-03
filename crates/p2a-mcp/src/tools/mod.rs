//! MCP Tool definitions and registry for prompt2analytics.
//!
//! This module provides:
//! - Tool metadata and registry for programmatic discovery
//! - Category-based organization of tools
//! - Search functionality for finding relevant tools
//! - Documentation generation
//! - Common utilities for tool handlers
//! - Request types organized by category
//! - Handler implementations organized by category
//!
//! # Module Organization
//!
//! - `registry`: Tool metadata for programmatic discovery
//! - `common`: Shared utilities (success_text, error_text, macros)
//! - `requests`: Request structs organized by category
//! - `handlers`: Tool implementations organized by category
//!
//! # Example
//!
//! ```
//! use p2a_mcp::tools::registry::{get_registry, search_tools, ToolCategory};
//!
//! // Get all tools
//! let all_tools = get_registry();
//! println!("Total tools: {}", all_tools.len());
//!
//! // Search for regression tools
//! let regression_tools = search_tools("regression");
//! for tool in regression_tools {
//!     println!("{}: {}", tool.name, tool.description);
//! }
//! ```

pub mod common;
pub mod registry;

// Request types organized by category
pub mod requests;

// Handler implementations organized by category
pub mod handlers;

pub use registry::{
    ToolCategory, ToolInfo, category_counts, generate_markdown_docs, get_registry, search_tools,
    tool_count, tools_by_category,
};
