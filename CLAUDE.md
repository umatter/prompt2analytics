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

# Run CLI
cargo run -p p2a-cli -- <args>

# Run MCP server (HTTP mode for development)
cargo run -p p2a-mcp --features full -- --transport http --host 127.0.0.1 --port 8080

# Dioxus app (web and desktop)
cd crates/p2a-dioxus && dx serve                      # Web dev server with hot reload
cd crates/p2a-dioxus && dx serve --platform desktop   # Desktop app
cd crates/p2a-dioxus && dx build --release            # Production web build

# Build documentation
cargo doc --no-deps --open
```

## Project Overview

## Project Overview

prompt2analytics is a Rust workspace (edition 2024, requires Rust 1.85+) exposing econometrics, ML, and visualization through multiple interfaces:

- **p2a-core**: Core analytics library (all algorithms)
- **p2a-cli**: Command-line interface (`p2a` binary) for direct analytics execution
- **p2a-mcp**: MCP server exposing 55+ tools
- **p2a-dioxus**: Cross-platform GUI (web via WASM, desktop via native)

## Architecture Principles

### Pure Rust Econometrics

All econometrics are implemented in pure Rust without external econometrics libraries. This design choice was made to:
1. Avoid dependency version conflicts (especially with ndarray)
2. Have full control over the API design
3. Use a column-based API instead of R-style formula parsing

Key dependencies for econometrics:
- `ndarray` 0.16 - Matrix operations
- `faer` 0.22 - Linear algebra (Cholesky, matrix inverse)
- `statrs` 0.18 - Statistical distributions

### Module Organization (p2a-core)

```
src/
├── errors.rs           # EconError, EconResult types
├── linalg/
│   ├── matrix_ops.rs   # xtx, xty, safe_inverse, cholesky (via faer)
│   └── design.rs       # DesignMatrix, demeaning functions
├── traits/
│   └── estimator.rs    # LinearEstimator trait, p-value helpers
├── regression/
│   ├── ols.rs          # OLS with HC0-HC3 robust SEs, clustered SEs
│   └── diagnostics.rs  # JB, BP, DW, VIF, condition number
├── econometrics/
│   ├── panel.rs        # Fixed Effects, Random Effects, Hausman
│   ├── iv.rs           # 2SLS with first-stage diagnostics
│   ├── did.rs          # Difference-in-Differences
│   ├── discrete.rs     # Logit, Probit (Newton-Raphson MLE)
│   ├── feglm.rs        # GLM with HDFE (IRLS + weighted MAP)
│   └── timeseries.rs   # VAR, VARMA, VECM, IRF
├── visualization/
│   ├── charts.rs       # Static charts (plotters) - PNG output
│   ├── heatmap.rs      # Correlation heatmaps
│   └── interactive.rs  # Interactive charts (plotlars/Plotly) - HTML output
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

**NOT** formula-based like R:
```rust
// This is NOT how it works
run_ols("y ~ x1 + x2")
```

### LinearEstimator Trait

All estimators implement the `LinearEstimator` trait for consistent output:
```rust
pub trait LinearEstimator {
    fn coefficients(&self) -> &Array1<f64>;
    fn std_errors(&self) -> &Array1<f64>;
    fn t_values(&self) -> Array1<f64>;
    fn p_values(&self) -> Array1<f64>;
    fn residuals(&self) -> Array1<f64>;
    fn n_obs(&self) -> usize;
    fn df(&self) -> usize;
    // ... and more
}
```

### Error Handling

Use `EconError` and `EconResult<T>` from `src/errors.rs`:
```rust
use crate::errors::{EconError, EconResult};

fn my_function() -> EconResult<MyResult> {
    // ...
    Err(EconError::InvalidInput("message".to_string()))
}
```

## Common Patterns

### Matrix Operations

Use functions from `linalg/matrix_ops.rs`:
```rust
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse, cholesky};

let xtx = xtx(&x);           // X'X
let xty = xty(&x, &y);       // X'y
let inv = safe_inverse(&m)?;  // Safe matrix inverse via Cholesky
```

### P-Value Calculation

Use helpers from `traits/estimator.rs`:
```rust
use crate::traits::estimator::{t_test_p_value, f_test_p_value, chi_squared_p_value};

let p = t_test_p_value(t_stat, df);
let p = f_test_p_value(f_stat, df1, df2);
```

These functions handle edge cases (NaN, Inf) gracefully.

### Design Matrix

Use `DesignMatrix` from `linalg/design.rs` for building regression matrices:
```rust
use crate::linalg::design::DesignMatrix;

let dm = DesignMatrix::from_dataset(dataset, x_cols, intercept)?;
let x = dm.view();
```

### Robust Standard Errors

The `CovarianceType` enum controls variance estimation:
```rust
pub enum CovarianceType {
    Standard,  // Homoskedastic
    HC0,       // White's heteroskedasticity-consistent
    HC1,       // HC0 with small-sample correction (default)
    HC2,       // HC1 with leverage adjustment
    HC3,       // HC2 with more aggressive correction
}
```

## CLI (p2a-cli)

### Command Structure

The CLI uses clap with hierarchical subcommands:

```
p2a [OPTIONS] <COMMAND>
  data       Data loading and inspection
  reg        Regression analysis
  panel      Panel data econometrics
  causal     Causal inference (IV, DiD)
  discrete   Discrete choice models
  ts         Time series analysis
  ml         Machine learning
  viz        Visualization
  script     Session/script management
```

### Adding a New Command

1. Create or modify the appropriate command module in `commands/`
2. Add the subcommand enum variant with clap attributes
3. Implement the execute function

Example in `commands/regression.rs`:
```rust
#[derive(Subcommand)]
pub enum RegressionCommands {
    /// Ordinary Least Squares regression
    Ols {
        dataset: String,
        #[arg(short = 'y', long)]
        dep_var: String,
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
        #[arg(long, default_value = "true")]
        intercept: bool,
        #[arg(short, long, default_value = "hc1")]
        robust: RobustSE,
    },
}
```

### Session Management

The CLI maintains session state for dataset persistence across commands:
```rust
pub struct SessionManager {
    session_path: PathBuf,
    session: Session,
    datasets: HashMap<String, Dataset>,
}
```

Use `--session <file>` to record commands for reproducibility.

### Data Extraction Patterns

**For regression-type functions** (use Dataset directly):
```rust
let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
run_ols(ds, dep_var, &x_cols, intercept, cov_type)
```

**For ML functions** (need ArrayView2):
```rust
fn extract_columns_as_array(dataset: &Dataset, cols: &[String]) -> Result<Array2<f64>, String> {
    let df = dataset.df();
    let mut data = Vec::new();
    for row_idx in 0..df.height() {
        for col_name in cols {
            let value = df.column(col_name)?.f64()?.get(row_idx)?;
            data.push(value);
        }
    }
    Array2::from_shape_vec((n_rows, n_cols), data)
}

// Then use: kmeans(data.view(), k, ...)
```

**For static visualization** (need raw Vec<f64>):
```rust
fn extract_column(dataset: &Dataset, col: &str) -> Result<Vec<f64>, String> {
    let column = dataset.df().column(col)?;
    Ok(column.f64()?.into_no_null_iter().collect())
}

// Then use: histogram(&data, bins, config)
```

**For interactive visualization** (pass DataFrame directly):
```rust
use p2a_core::visualization::{scatter_interactive, InteractiveConfig};

let config = InteractiveConfig {
    title: Some("My Plot".to_string()),
    x_label: Some("X Axis".to_string()),
    y_label: Some("Y Axis".to_string()),
    ..Default::default()
};

// Pass DataFrame directly - plotlars handles extraction
let result = scatter_interactive(dataset.df(), "x_col", "y_col", Some("group_col"), config)?;
// result.html contains full HTML page with embedded Plotly.js
```

### Output Formatting

Results support multiple output formats via `OutputFormat`:
```rust
match format {
    OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&json)?),
    OutputFormat::Table => /* use tabled */,
    OutputFormat::Text => /* formatted text */,
}
```

## MCP Server (p2a-mcp)

### Adding a New Tool

1. Define the request struct with `#[derive(Deserialize, JsonSchema)]`
2. Add the tool handler in `server.rs`
3. Register in the `#[tool]` attribute

Example:
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

### Key Tools

**`create_dataset`** - Creates a dataset from inline CSV content (for generated/test data):
```rust
#[derive(Deserialize, JsonSchema)]
pub struct CreateDatasetRequest {
    pub name: String,
    pub csv_content: String,  // e.g., "x,y\n1,2\n3,4"
}
```
Uses `DataLoader::from_csv_string()` internally. The LLM system prompt instructs it to use this tool when asked to generate pseudo/test data.

### Session State

The MCP server maintains a `DatasetStore` for loaded datasets:
```rust
let datasets = self.datasets.lock().await;
let dataset = datasets.get(&request.dataset)
    .ok_or_else(|| McpError::invalid_request("Dataset not found", None))?;
```

### Database Layer (SurrealDB)

The `db/` module provides persistent storage via embedded SurrealDB (RocksDB backend):

```
src/db/
├── mod.rs           # Module exports
├── connection.rs    # Database connection management
├── models.rs        # Data models (Session, Conversation, Message, Settings)
├── conversations.rs # Conversation CRUD operations
└── sessions.rs      # Session persistence
```

**Key Models:**
```rust
// Uses SurrealDB native types for proper serialization
pub struct Conversation {
    pub id: surrealdb::RecordId,
    pub session_id: surrealdb::RecordId,
    pub title: String,
    pub created_at: surrealdb::sql::Datetime,
    pub updated_at: surrealdb::sql::Datetime,
    pub is_archived: bool,
}

pub struct ConversationMessage {
    pub id: surrealdb::RecordId,
    pub conversation_id: surrealdb::RecordId,
    pub role: String,  // "user" or "assistant"
    pub content: String,
    pub created_at: surrealdb::sql::Datetime,
}
```

**Database Operations:**
```rust
// Initialize database (creates tables if not exist)
let db = Database::new("~/.p2a/data").await?;

// Conversation CRUD
db.create_conversation(session_id, title).await?;
db.list_conversations(session_id).await?;
db.update_conversation_title(id, title).await?;
db.delete_conversation(id).await?;

// Message operations
db.add_message(conversation_id, role, content).await?;
db.get_messages(conversation_id).await?;
db.clear_messages(conversation_id).await?;
```

**Important Notes:**
- Use `surrealdb::sql::Datetime` for timestamps (not chrono types)
- Use `surrealdb::RecordId` for IDs (not String)
- For datetime updates, use raw SurrealQL with `time::now()`:
```rust
self.db()
    .query("UPDATE conversations SET updated_at = time::now() WHERE id = $id")
    .bind(("id", RecordId::from(("conversations", id))))
    .await?;
```
- The `id_string()` helper strips Unicode angle brackets from RecordId keys

## Dioxus App (p2a-dioxus)

### Overview

Cross-platform GUI using Dioxus 0.7, compiling to WebAssembly (web) or native (desktop). Communicates with p2a-mcp HTTP backend.

### Architecture

```
p2a-dioxus/
├── src/
│   ├── main.rs           # Entry point
│   ├── app.rs            # Root App component (sidebar + chat layout)
│   ├── api/              # Backend communication
│   │   ├── client.rs     # HTTP client with conversation endpoints
│   │   ├── sse.rs        # SSE streaming for LLM chat
│   │   └── types.rs      # Request/response types (including Conversation)
│   ├── state/            # State management (Dioxus signals)
│   │   ├── chat.rs       # Messages, history navigation
│   │   ├── conversation.rs # Conversation list and selection
│   │   ├── session.rs    # Session lifecycle
│   │   └── settings.rs   # Provider config (localStorage)
│   ├── components/       # UI components
│   │   ├── chat_panel.rs # Main chat interface (integrated with conversations)
│   │   ├── conversation_sidebar.rs # Conversation list with CRUD
│   │   ├── chat_input.rs # Input with keyboard shortcuts
│   │   ├── message.rs    # Message with markdown
│   │   ├── message_list.rs # Auto-scrolling list
│   │   ├── tool_call.rs  # Expandable tool display
│   │   └── settings_modal.rs # Provider configuration
│   └── utils/
│       └── markdown.rs   # Markdown to RSX (pulldown-cmark)
└── assets/
    └── styles.css        # Tailwind-like CSS
```

### Key Dependencies

- `dioxus` 0.7 - UI framework with web feature
- `reqwest` - HTTP client (WASM-compatible)
- `pulldown-cmark` - Markdown parsing
- `gloo-storage` - localStorage for settings persistence
- `chrono` + `uuid` - Timestamps and IDs

### Running

```bash
# Terminal 1: Backend
cargo run -p p2a-mcp --features full -- --transport http --port 8080 --cors-permissive

# Terminal 2: Dioxus dev server
cd crates/p2a-dioxus && dx serve
```

### State Management

Uses Dioxus signals and context providers:
```rust
let chat_state = use_context::<Signal<ChatState>>();
chat_state.write().add_user_message(&message);
```

**Global State (provided in App):**
- `SessionState` - Backend session ID, loaded datasets, refresh counter
- `ChatState` - Current messages, streaming state, prompt history, tool calls
- `ConversationState` - Conversation list and current selection
- `Settings` - LLM provider configuration (with env var detection)

### Tool Call Display

The frontend displays tool calls with full transparency:
- **"Rust Analytics" indicator**: Shows at the top of assistant messages when tools were called
- **Tool chips**: List of tool names in a compact horizontal format
- **Expandable cards**: Click to see arguments and results

Tool calls are tracked during streaming via SSE events:
```rust
// Backend sends these events:
ProgressEvent::ToolStart { tool: String, arguments: serde_json::Value }
ProgressEvent::ToolEnd { tool: String, elapsed_ms: u64, result: Option<String> }

// Frontend tracks them in ChatMessage:
pub struct ToolCallInfo {
    pub name: String,
    pub arguments: serde_json::Value,
    pub result: Option<String>,
    pub success: Option<bool>,  // None = running, Some(true) = success
}
```

### Dataset Sidebar Refresh

The dataset sidebar auto-refreshes after tool execution:
```rust
// In SessionState:
pub datasets_refresh_counter: u32,

pub fn trigger_datasets_refresh(&mut self) {
    self.datasets_refresh_counter = self.datasets_refresh_counter.wrapping_add(1);
}

// Called after chat completes in ChatPanel:
session.write().trigger_datasets_refresh();

// DatasetSidebar watches this counter via use_effect
```

### Environment Variable Detection

On desktop, Settings automatically detects API keys from environment:
```rust
#[cfg(not(target_arch = "wasm32"))]
fn populate_from_env(&mut self) {
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        self.openai_api_key = key;
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        self.anthropic_api_key = key;
    }
}
```

**ConversationState:**
```rust
pub struct ConversationState {
    pub conversations: Vec<Conversation>,
    pub current_conversation_id: Option<String>,
    pub current_messages: Vec<ConversationMessage>,
    pub is_loading: bool,
    pub is_operating: bool,
    pub error: Option<String>,
}

// Methods for API operations
state.load_conversations(session_id).await?;
state.create_conversation(session_id, title).await?;
state.update_conversation_title(id, title).await?;
state.delete_conversation(id).await?;
state.load_messages(conversation_id).await?;
state.add_message(conversation_id, role, content).await?;
```

**ChatMessage Conversion:**
```rust
// Convert persisted message to UI message
let chat_msg = ChatMessage::from_conversation_message(&conversation_message);
```

Settings persist to localStorage via `gloo-storage`.

## Testing

### Running Tests

```bash
cargo test -p p2a-core        # Core library tests
cargo test -p p2a-mcp         # MCP server tests
```

### Test Data

Test datasets should have noise (not perfect linear relationships) to avoid zero residuals:
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
3. **No formula parsing**: Use column names directly, not R-style formulas
4. **Serde serialization**: Use `#[serde(skip)]` for large internal matrices in result structs
5. **Error handling**: Use `EconError` for econometric errors, `McpError` for MCP errors
6. **Visualization**: Two types available:
   - Static (plotters): `histogram()`, `scatter_plot()`, etc. - returns base64 PNG
   - Interactive (plotlars/Plotly): `scatter_interactive()`, etc. - returns HTML

## Key Files

- `crates/p2a-core/src/regression/ols.rs` - Main OLS implementation
- `crates/p2a-core/src/linalg/matrix_ops.rs` - Core linear algebra (xtx, xty, safe_inverse)
- `crates/p2a-core/src/traits/estimator.rs` - LinearEstimator trait and p-value helpers
- `crates/p2a-core/src/visualization/interactive.rs` - Interactive charts (plotlars/Plotly)
- `crates/p2a-mcp/src/server.rs` - All 55+ MCP tool definitions (including create_dataset)
- `crates/p2a-mcp/src/transport/http.rs` - HTTP transport with SSE streaming events
- `crates/p2a-mcp/src/llm/tools.rs` - LLM tool definitions and system prompt
- `crates/p2a-mcp/src/db/` - SurrealDB persistence layer
- `crates/p2a-mcp/src/db/conversations.rs` - Conversation CRUD operations
- `crates/p2a-mcp/src/db/models.rs` - Database models with SurrealDB types
- `crates/p2a-cli/src/commands/` - CLI subcommand implementations
- `crates/p2a-dioxus/src/components/chat_panel.rs` - Main Dioxus chat interface
- `crates/p2a-dioxus/src/components/message.rs` - Message display with tool call indicator
- `crates/p2a-dioxus/src/components/tool_call.rs` - Expandable tool call card
- `crates/p2a-dioxus/src/components/dataset_sidebar.rs` - Dataset list with refresh
- `crates/p2a-dioxus/src/components/conversation_sidebar.rs` - Conversation list UI
- `crates/p2a-dioxus/src/state/chat.rs` - Chat state with tool call tracking
- `crates/p2a-dioxus/src/state/session.rs` - Session state with dataset refresh counter
- `crates/p2a-dioxus/src/state/settings.rs` - Settings with env var detection
- `crates/p2a-dioxus/src/api/client.rs` - API client with conversation endpoints
- `crates/p2a-dioxus/src/api/sse.rs` - SSE streaming for Dioxus
- `crates/p2a-dioxus/src/api/types.rs` - API types including StreamEvent
- `DEVELOPMENT_REPORT.md` - Detailed development history and current status
