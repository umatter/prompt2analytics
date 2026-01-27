# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build all crates
cargo build

# Build individual crates
cargo build -p p2a-core
cargo build -p p2a-cli
cargo build -p p2a-mcp
cargo build -p p2a-dioxus

# Release builds
cargo build --release -p p2a-cli        # CLI binary at target/release/p2a
cargo build --release -p p2a-mcp        # MCP server at target/release/p2a-mcp

# Run tests
cargo test                              # All tests
cargo test -p p2a-core                  # Core library tests only
cargo test -p p2a-mcp                   # MCP server tests only
cargo test test_name                    # Run specific test by name
cargo test -p p2a-core -- test_validate # Run validation tests only

# Linting
cargo clippy --all-targets --all-features
cargo fmt --check                        # Check formatting
cargo fmt                                # Apply formatting

# Run CLI
cargo run -p p2a-cli -- <args>

# Run MCP server (HTTP mode for development)
cargo run -p p2a-mcp --features full -- --transport http --host 127.0.0.1 --port 8080 --cors-permissive

# Dioxus app (web and desktop)
cd crates/p2a-dioxus && dx serve                      # Web dev server with hot reload
cd crates/p2a-dioxus && dx serve --platform desktop   # Desktop app
cd crates/p2a-dioxus && dx build --release            # Production web build

# Build documentation
cargo doc --no-deps --open
```

### Prerequisites

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopenblas-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
```

**macOS:**
```bash
brew install openblas
```

**Dioxus CLI** (for p2a-dioxus):
```bash
cargo install dioxus-cli
rustup target add wasm32-unknown-unknown
```

## Project Overview

prompt2analytics is a Rust workspace (edition 2024, requires Rust 1.85+) exposing econometrics, ML, and visualization through multiple interfaces:

- **p2a-core**: Core analytics library (all algorithms)
- **p2a-cli**: Command-line interface (`p2a` binary)
- **p2a-mcp**: MCP server exposing 60+ tools with LLM integration
- **p2a-dioxus**: Cross-platform GUI (web via WASM, desktop via native)

## Architecture Principles

### Pure Rust Econometrics

All econometrics are implemented in pure Rust without external econometrics libraries. This provides:
1. No dependency version conflicts (especially with ndarray)
2. Full control over API design
3. Column-based API instead of R-style formula parsing

Key dependencies for econometrics:
- `ndarray` 0.16 - Matrix operations
- `faer` 0.22 - Linear algebra (Cholesky, matrix inverse)
- `statrs` 0.18 - Statistical distributions
- `polars` 0.52 - DataFrame operations

### Module Organization (p2a-core)

```
src/
├── errors.rs           # EconError, EconResult types
├── linalg/
│   ├── matrix_ops.rs   # xtx, xty, safe_inverse, cholesky (via faer)
│   └── design.rs       # DesignMatrix, demeaning functions
├── traits/
│   └── estimator.rs    # LinearEstimator trait, SignificanceLevel, p-value helpers
├── regression/
│   ├── ols.rs          # OLS with HC0-HC3 robust SEs, clustered SEs
│   └── diagnostics.rs  # JB, BP, DW, VIF, condition number
├── econometrics/
│   ├── panel.rs        # Fixed Effects, Random Effects, Hausman
│   ├── iv.rs           # 2SLS with first-stage diagnostics
│   ├── did.rs          # Difference-in-Differences
│   ├── discrete.rs     # Logit, Probit (Newton-Raphson MLE with line search)
│   ├── feglm.rs        # GLM with HDFE (IRLS + weighted MAP)
│   ├── rd.rs           # Regression Discontinuity (Sharp/Fuzzy RD)
│   └── timeseries.rs   # VAR, VARMA, VECM, IRF
├── ml/
│   ├── clustering.rs   # K-means, DBSCAN, Hierarchical
│   ├── reduction.rs    # PCA (via SVD), t-SNE
│   ├── trees.rs        # Random Forest (CART)
│   └── svm.rs          # Linear SVM (SMO)
├── visualization/
│   ├── charts.rs       # Static charts (plotters) - PNG output
│   ├── heatmap.rs      # Correlation heatmaps
│   └── interactive.rs  # Interactive charts (plotlars/Plotly) - HTML output
├── export/
│   ├── latex.rs        # LaTeX tables (OLS, Panel, Discrete)
│   ├── markdown.rs     # Markdown tables for documentation
│   ├── html.rs         # Self-contained HTML tables
│   └── csv.rs          # CsvExport trait for all result types
└── data/
    ├── quality.rs      # DataQualityProfile for LLM-assisted cleaning
    ├── verification.rs # Cleaning verification and preview
    └── cleaning_session.rs # Rollback-enabled cleaning sessions
```

### API Design

**Column-based API**: All regression functions use explicit column names:
```rust
pub fn run_ols(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    intercept: bool,
    cov_type: CovarianceType,
) -> Result<OlsResult, EconError>
```

**NOT** formula-based like R (`run_ols("y ~ x1 + x2")`)

### LinearEstimator Trait

All estimators implement `LinearEstimator` for consistent output:
```rust
pub trait LinearEstimator {
    fn coefficients(&self) -> &Array1<f64>;
    fn std_errors(&self) -> &Array1<f64>;
    fn t_values(&self) -> Array1<f64>;
    fn p_values(&self) -> Array1<f64>;
    fn residuals(&self) -> Array1<f64>;
    fn n_obs(&self) -> usize;
    fn df(&self) -> usize;
}
```

### Error Handling

Use `EconError` and `EconResult<T>` from `src/errors.rs`:
```rust
use crate::errors::{EconError, EconResult};

fn my_function() -> EconResult<MyResult> {
    Err(EconError::InvalidInput("message".to_string()))
}
```

## Common Patterns

### Matrix Operations

Use functions from `linalg/matrix_ops.rs`:
```rust
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse};

let xtx = xtx(&x);           // X'X
let xty = xty(&x, &y);       // X'y
let inv = safe_inverse(&m)?;  // Safe matrix inverse via Cholesky
```

### P-Value Calculation

Use helpers from `traits/estimator.rs`:
```rust
use crate::traits::estimator::{t_test_p_value, f_test_p_value, chi_squared_p_value};

let p = t_test_p_value(t_stat, df);  // handles NaN, Inf gracefully
```

### Robust Standard Errors

```rust
pub enum CovarianceType {
    Standard,  // Homoskedastic
    HC0,       // White's heteroskedasticity-consistent
    HC1,       // HC0 with small-sample correction (default)
    HC2,       // HC1 with leverage adjustment
    HC3,       // HC2 with more aggressive correction
}
```

### MLE Settings (Discrete Models)

Logit/Probit use Newton-Raphson with optional backtracking line search:
```rust
pub struct MleSettings {
    pub max_iter: usize,        // Default: 100
    pub tolerance: f64,         // Default: 1e-8
    pub step_size: f64,         // Default: 1.0
    pub use_line_search: bool,  // Default: true (Armijo backtracking)
    pub armijo_c: f64,          // Default: 1e-4 (sufficient decrease)
    pub step_reduction: f64,    // Default: 0.5
    pub max_line_search: usize, // Default: 20
}
```

The line search improves convergence for difficult problems (near-separation).
Multivariate separation is detected via coefficient explosion monitoring.

## MCP Server (p2a-mcp)

### Adding a New Tool

1. Define the request struct with `#[derive(Deserialize, JsonSchema)]`
2. Add the tool handler in `server.rs`
3. Register with the `#[tool]` attribute

```rust
#[derive(Deserialize, JsonSchema)]
pub struct MyToolRequest {
    pub dataset: String,
    pub param: String,
}

#[tool(description = "My tool description")]
async fn my_tool(&self, #[tool(aggr)] request: MyToolRequest) -> Result<String, McpError> {
    // Implementation
}
```

### Database Layer (SurrealDB)

Persistent storage via embedded SurrealDB (RocksDB backend):

**Important Notes:**
- Use `surrealdb::sql::Datetime` for timestamps (not chrono types)
- Use `surrealdb::RecordId` for IDs (not String)
- For datetime updates, use raw SurrealQL with `time::now()`

## Dioxus App (p2a-dioxus)

Cross-platform GUI using Dioxus 0.7, compiling to WebAssembly (web) or native (desktop).

### Running

```bash
# Terminal 1: Backend
cargo run -p p2a-mcp --features full -- --transport http --port 8080 --cors-permissive

# Terminal 2: Dioxus dev server
cd crates/p2a-dioxus && dx serve
```

### State Management

Uses Dioxus signals and context providers:
- `SessionState` - Backend session ID, loaded datasets, refresh counter
- `ChatState` - Current messages, streaming state, prompt history, tool calls
- `ConversationState` - Conversation list and current selection
- `Settings` - LLM provider configuration (with env var detection)

### Tool Call Display

Tool calls tracked during streaming via SSE events (`ToolStart`, `ToolEnd`). Frontend shows:
- "Rust Analytics" indicator for messages with tool calls
- Expandable cards showing arguments and results

## Validation & Benchmarking

### Validation Framework (`validation/`)

All methods validated against R/Python reference implementations. Run validation tests:
```bash
cargo test -p p2a-core -- test_validate
```

### Performance Benchmarks (`performance/`)

Criterion benchmarks with R comparison scripts:
```bash
# Rust benchmarks
cargo bench -p p2a-core --bench comprehensive_benchmarks

# R benchmarks
cd performance/comparisons/r_comparison && Rscript benchmark_comprehensive.R
```

## Agentic Engineering Setup

### Slash Commands (`.claude/commands/`)

- `/implement_metrics <url|file>` - Implement new econometric method from documentation
- `/discover_methods` - Find unimplemented methods from package indices
- `/implement_next` - Implement next highest-priority method from queue
- `/validate-method` - Run R vs Rust validation for a method

### Skills (`.claude/skills/`)

Auto-discovered guidance:
- `econometrics-research` - Finding reference implementations, extracting formulas
- `rust-econometrics-patterns` - p2a-core API patterns, LinearEstimator trait
- `validation-benchmarking` - Validation and benchmarking workflow

## Testing Guidelines

### Test Data

Test datasets should have noise to avoid zero residuals:
```rust
// Good: y has noise
let df = df! {
    "y" => [1.1, 1.9, 3.2, 3.8, 5.1],  // y ≈ x + noise
    "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
}

// Bad: perfect fit causes zero std errors
let df = df! {
    "y" => [1.0, 2.0, 3.0, 4.0, 5.0],  // y = x exactly
    "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
}
```

## Important Notes

1. **ndarray version**: Pinned to 0.16 for compatibility with faer
2. **polars version**: Using 0.52; `is_numeric()` was removed, use custom dtype checking
3. **Serde serialization**: Use `#[serde(skip)]` for large internal matrices in result structs
4. **Visualization**: Two types:
   - Static (plotters): `histogram()`, `scatter_plot()` - returns base64 PNG
   - Interactive (plotlars/Plotly): `scatter_interactive()` - returns HTML
5. **Export formats**: Four export types available via `export/` module:
   - LaTeX tables (publication-ready, OLS/Panel/Discrete)
   - Markdown tables (documentation, GitHub)
   - HTML tables (self-contained with CSS)
   - CSV via `CsvExport` trait (all result types)

## Key Files

**Core Implementation:**
- `crates/p2a-core/src/regression/ols.rs` - OLS with robust SEs
- `crates/p2a-core/src/linalg/matrix_ops.rs` - Linear algebra primitives
- `crates/p2a-core/src/traits/estimator.rs` - LinearEstimator trait

**MCP Server:**
- `crates/p2a-mcp/src/server.rs` - All MCP tool definitions
- `crates/p2a-mcp/src/transport/http.rs` - HTTP transport with SSE streaming
- `crates/p2a-mcp/src/db/` - SurrealDB persistence layer

**Dioxus App:**
- `crates/p2a-dioxus/src/components/chat_panel.rs` - Main chat interface
- `crates/p2a-dioxus/src/state/` - State management (chat, session, settings)
- `crates/p2a-dioxus/src/api/sse.rs` - SSE streaming for chat

**Documentation:**
- `DEVELOPMENT_REPORT.md` - Detailed development history and current status
- `validation/` - Validation against R/Python reference implementations
- `performance/` - Benchmark results and methodology
- `docs/guides/TESTING.md` - Test runtime expectations, validation framework
- `docs/guides/DATA_SECURITY.md` - Data write locations, offline capability
- `docs/security/PROMPT_INJECTION.md` - MCP security considerations

**Export Module:**
- `crates/p2a-core/src/export/latex.rs` - LaTeX table builders
- `crates/p2a-core/src/export/csv.rs` - CsvExport trait implementations
- `crates/p2a-core/src/export/html.rs` - Self-contained HTML export
- `crates/p2a-core/src/export/markdown.rs` - Markdown table builders
