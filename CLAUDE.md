# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build (debug)
cargo build

# Build (release)
cargo build --release

# Run tests
cargo test

# Check compilation without building
cargo check

# Run the MCP server (debug)
cargo run --bin p2a-mcp

# Run the MCP server (release, faster startup)
./target/release/p2a-mcp
```

## Architecture

prompt2analytics is an MCP (Model Context Protocol) server that exposes data analytics capabilities to LLMs. It's a Cargo workspace with two crates:

### Crate Structure

**p2a-core** (`crates/p2a-core/`) - Analytics engine library
- `data/` - Data loading (CSV, Parquet via Polars) and Dataset wrapper
- `stats/` - Descriptive statistics and correlation matrix
- `regression/` - OLS regression with linfa-linear
- `ml/` - Placeholder for future ML algorithms

**p2a-mcp** (`crates/p2a-mcp/`) - MCP server binary
- `server.rs` - Tool definitions using rmcp macros (`#[tool_router]`, `#[tool]`, `#[tool_handler]`)
- `main.rs` - Entry point, stdio transport setup

### Key Dependencies

- **polars 0.46** - DataFrame operations. Note: API differs from older versions (no `is_numeric()` method, use custom `is_numeric_dtype()`)
- **rmcp 0.8** - MCP SDK. Tool parameters use `Parameters<T>` wrapper from `rmcp::handler::server::wrapper`
- **linfa 0.7 / linfa-linear 0.7** - ML framework for regression
- **ndarray 0.15** - Must match linfa's version exactly

### MCP Tools Exposed

1. `list_datasets` - Show loaded datasets
2. `load_dataset` - Load CSV/Parquet files
3. `describe_dataset` - Summary statistics
4. `head_dataset` - Preview rows
5. `compute_correlation` - Pearson correlation matrix
6. `regression_ols` - OLS regression with full output (coefficients, SE, t-values, p-values, R², F-stat)

### Adding New Tools

In `server.rs`, tools are defined with the `#[tool]` attribute inside the `#[tool_router]` impl block:

```rust
#[tool(description = "Tool description for LLM")]
async fn my_tool(
    &self,
    Parameters(request): Parameters<MyRequest>,
) -> Result<CallToolResult, McpError> {
    // Implementation
    Ok(CallToolResult::success(vec![Content::text("result")]))
}
```

Request structs need `Deserialize` and `JsonSchema` derives. Use `#[schemars(description = "...")]` for parameter documentation.

## MCP Configuration

The `.mcp.json` in the repo root configures Claude Code to use this server when working in this directory. Requires `cargo build --release` first.
