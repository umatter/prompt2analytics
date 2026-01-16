# prompt2analytics

A comprehensive analytics toolkit exposing econometrics, machine learning, and visualization capabilities through multiple interfaces:
- **CLI (`p2a`)**: Direct command-line execution for scripted workflows
- **MCP Server**: Model Context Protocol integration for AI assistants
- **Desktop App**: Tauri application with LLM-powered natural language analysis
- **Web App (Dioxus)**: Pure Rust web frontend (experimental prototype)

## Features

### Econometrics (Pure Rust)
- **OLS Regression** with robust standard errors (HC0-HC3) and clustered SEs
- **Panel Data**: Fixed Effects, Random Effects, Hausman specification test
- **Instrumental Variables**: 2SLS with first-stage diagnostics
- **Causal Inference**: Difference-in-Differences estimation
- **Discrete Choice**: Logit and Probit via Newton-Raphson MLE
- **Regression Diagnostics**: Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF

### Time Series
- **Univariate**: ARIMA modeling and forecasting, MSTL decomposition
- **Multivariate**: VAR, VARMA, VECM with Impulse Response Functions
- **Changepoint Detection**: PELT and Binary Segmentation algorithms

### Machine Learning (Pure Rust)
- **Clustering**: K-means (k-means++ init), DBSCAN, Hierarchical (Ward, single, complete, average)
- **Dimensionality Reduction**: PCA, t-SNE
- **Supervised Learning**: Random Forest, Linear SVM

### Visualization
- **Static charts** (PNG): Histograms, scatter plots, line charts, box plots
- Correlation heatmaps, coefficient plots, IRF plots
- Event study plots, residual diagnostics, dendrograms
- **Interactive charts** (HTML/Plotly.js): Scatter, histogram, line with zoom/pan/hover
- Static charts rendered as base64-encoded PNG, interactive as self-contained HTML

### Data Sources
- **Files**: CSV, Parquet, Excel (.xlsx/.xls), Stata (.dta), SAS (.sas7bdat)
- **Databases**: SQLite, DuckDB (with direct file querying)

### Command-Line Interface
- Full access to all analytics via `p2a` binary
- Session recording for reproducibility
- Script export for sharing and automation
- JSON output format for programmatic use

### Desktop Application
- Tauri 2.0 + SvelteKit interface
- Multi-provider LLM integration (Ollama, Anthropic, OpenAI)
- Conversation history with SQLite persistence
- Natural language data analysis

## Installation

### Prerequisites

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libopenblas-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
```

**macOS:**
```bash
brew install openblas
```

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt2analytics.git
cd prompt2analytics

# Build the CLI
cargo build --release -p p2a-cli

# Build the MCP server
cargo build --release -p p2a-mcp

# Build the desktop application (optional)
cd crates/p2a-desktop/ui && npm install && cd ../../..
cargo build --release -p p2a-desktop
```

The CLI binary will be at `target/release/p2a`.

## Usage

### CLI (`p2a`)

The `p2a` command provides direct access to all analytics functions:

```bash
# Load a dataset
p2a --session analysis.json data load /path/to/data.csv --name mydata

# View data summary
p2a --session analysis.json data describe mydata

# Run OLS regression with robust standard errors
p2a --session analysis.json reg ols mydata -y price -x sqft bedrooms bathrooms --robust hc1

# Run clustered standard errors regression
p2a --session analysis.json reg clustered mydata -y outcome -x treatment control --cluster firm_id

# Panel fixed effects
p2a --session analysis.json panel fe mydata -y revenue -x employees --entity firm_id

# Time series ARIMA with forecasting
p2a --session analysis.json ts arima mydata --col sales -p 1 -d 1 -q 1 --horizon 12

# K-means clustering
p2a --session analysis.json ml kmeans mydata --cols x1 x2 x3 -k 3

# Create static visualizations (PNG)
p2a --session analysis.json viz scatter mydata -x income -y spending -f scatter.png
p2a --session analysis.json viz histogram mydata --col price -f hist.png

# Create interactive visualizations (HTML with Plotly.js)
p2a --session analysis.json viz scatter-interactive mydata -x income -y spending -f scatter.html
p2a --session analysis.json viz histogram-interactive mydata --col price -f hist.html
p2a --session analysis.json viz line-interactive mydata -x date -y sales -f timeseries.html

# Export session to reproducible bash script
p2a script export analysis.json -o analysis.sh
```

**Command categories:**
- `data` - Load, list, describe, preview datasets
- `reg` - OLS, clustered SEs, diagnostics
- `panel` - Fixed effects, random effects, Hausman test, HDFE
- `causal` - IV/2SLS, difference-in-differences
- `discrete` - Logit, probit
- `ts` - ARIMA, MSTL, VAR
- `ml` - K-means, PCA
- `viz` - Static (PNG) and interactive (HTML) charts: histograms, scatter plots, line charts, box plots, heatmaps, coefficient plots, residual diagnostics, dendrograms, event study plots, IRF plots
- `script` - Export/run reproducible scripts

**Output formats:** `--output text` (default), `--output json`, `--output table`

### MCP Server

The MCP server exposes 55 analytics tools via the Model Context Protocol. Configure it in your MCP client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "prompt2analytics": {
      "command": "/path/to/target/release/p2a-mcp"
    }
  }
}
```

### Desktop Application

```bash
./target/release/p2a-desktop
```

The desktop app provides:
- **Chat Panel**: Natural language interface to analytics tools
- **Data Panel**: Load and preview datasets
- **Results Panel**: View analysis results and visualizations

### Dioxus Web App (Experimental)

A pure Rust web frontend built with Dioxus 0.7, compiling to WebAssembly:

```bash
# Install Dioxus CLI (if not already installed)
cargo install dioxus-cli

# Install WASM target
rustup target add wasm32-unknown-unknown

# Start backend in one terminal
cargo run -p p2a-mcp --features full -- --transport http --port 8080 --cors-permissive

# Start Dioxus dev server in another terminal
cd crates/p2a-dioxus
dx serve
```

Open http://localhost:8080 in your browser. Features:
- Chat interface with streaming LLM responses
- Support for Ollama, Anthropic, and OpenAI providers
- **Conversation management**: Create, rename, archive, and delete conversations
- **Persistent history**: Messages are saved to SurrealDB backend
- **Sidebar UI**: Collapsible conversation list with search
- Settings persistence via localStorage
- Tool call display with expandable details
- Markdown rendering for assistant messages

### Persistence Layer (SurrealDB)

The MCP server includes an optional SurrealDB persistence layer for conversation history:

```bash
# Run with database persistence (embedded RocksDB)
cargo run -p p2a-mcp --features full -- --transport http --port 8080 --cors-permissive

# Database stored at: ~/.p2a/data/
```

Features:
- **Embedded database**: No separate server required (RocksDB backend)
- **Conversation persistence**: Messages survive server restarts
- **Session management**: User sessions with timestamps and metadata
- **Settings storage**: Per-session LLM provider configuration

### Example Commands

Load a dataset:
```
/load_dataset path:/path/to/data.csv
```

Run OLS regression:
```
/regression_ols dataset:mydata y:price x:sqft,bedrooms,bathrooms
```

Generate a scatter plot (static PNG):
```
/viz_scatter dataset:mydata x_column:sqft y_column:price
```

Generate an interactive scatter plot (HTML with Plotly.js):
```
/viz_scatter_interactive dataset:mydata x_column:sqft y_column:price group_column:neighborhood
```

Run panel fixed effects:
```
/panel_fixed_effects dataset:panel y:outcome x:treatment entity_col:firm time_col:year
```

## Architecture

```
prompt2analytics/
├── crates/
│   ├── p2a-core/          # Core analytics library
│   │   ├── data/          # Data loading and manipulation
│   │   ├── stats/         # Descriptive statistics, correlation
│   │   ├── regression/    # OLS, diagnostics
│   │   ├── econometrics/  # Panel, IV, DiD, discrete choice, time series
│   │   ├── forecasting/   # ARIMA, MSTL, changepoint
│   │   ├── ml/            # Clustering, PCA, t-SNE, Random Forest, SVM
│   │   ├── visualization/ # Static (plotters) and interactive (plotlars) charts
│   │   ├── linalg/        # Matrix operations (via faer)
│   │   ├── traits/        # LinearEstimator trait
│   │   └── errors.rs      # Error types
│   ├── p2a-cli/           # CLI binary (`p2a`)
│   │   ├── commands/      # Subcommand implementations
│   │   ├── session.rs     # Session recording
│   │   ├── script.rs      # Bash script generation
│   │   └── output.rs      # Output formatting
│   ├── p2a-mcp/           # MCP server (55+ tools)
│   │   ├── src/
│   │   │   ├── server.rs  # Tool definitions
│   │   │   ├── session.rs # In-memory session management
│   │   │   └── db/        # SurrealDB persistence layer
│   │   │       ├── connection.rs  # Database connection
│   │   │       ├── models.rs      # Data models
│   │   │       ├── conversations.rs # Conversation CRUD
│   │   │       └── sessions.rs    # Session persistence
│   ├── p2a-desktop/       # Tauri desktop application
│   │   ├── src/           # Rust backend
│   │   └── ui/            # SvelteKit frontend
│   └── p2a-dioxus/        # Dioxus web app (experimental)
│       ├── src/           # Pure Rust frontend (WASM)
│       │   ├── api/       # HTTP client and SSE streaming
│       │   │   ├── client.rs  # API client with conversation endpoints
│       │   │   ├── sse.rs     # Server-sent events for streaming
│       │   │   └── types.rs   # Request/response types
│       │   ├── components/# UI components
│       │   │   ├── chat_panel.rs    # Main chat interface
│       │   │   ├── conversation_sidebar.rs # Conversation list
│       │   │   ├── chat_input.rs    # Message input
│       │   │   ├── message.rs       # Message display
│       │   │   └── settings_modal.rs # Provider config
│       │   ├── state/     # State management (Dioxus signals)
│       │   │   ├── chat.rs        # Message state
│       │   │   ├── conversation.rs # Conversation management
│       │   │   ├── session.rs     # Backend session
│       │   │   └── settings.rs    # User preferences
│       │   └── utils/     # Markdown rendering
│       └── assets/        # CSS styles
```

## MCP Tools

| Category | Tools |
|----------|-------|
| Data | `load_dataset`, `list_datasets`, `describe_dataset`, `head_dataset` |
| Statistics | `compute_correlation` |
| Regression | `regression_ols`, `regression_diagnostics`, `regression_clustered` |
| Panel | `panel_fixed_effects`, `panel_random_effects`, `hausman_test` |
| IV | `iv_2sls`, `iv_first_stage` |
| Causal | `diff_in_diff` |
| Discrete | `logit`, `probit` |
| Time Series | `ts_var`, `ts_varma`, `ts_vecm`, `ts_var_irf`, `ts_arima_fit`, `ts_arima_forecast`, `ts_mstl`, `ts_changepoint` |
| ML | `ml_kmeans`, `ml_dbscan`, `ml_hierarchical`, `ml_pca`, `ml_tsne`, `ml_random_forest`, `ml_svm` |
| Database | `db_sqlite_query`, `db_sqlite_tables`, `db_sqlite_schema`, `db_duckdb_query`, `db_duckdb_tables`, `db_duckdb_schema` |
| Visualization | `viz_histogram`, `viz_scatter`, `viz_line`, `viz_boxplot`, `viz_heatmap`, `viz_event_study`, `viz_coefficient`, `viz_irf`, `viz_residual_diagnostics`, `viz_dendrogram` (static PNG) |
| Interactive Viz | `viz_scatter_interactive`, `viz_histogram_interactive`, `viz_line_interactive` (HTML/Plotly.js) |
| Utilities | `generate_report`, `batch_process`, `compare_datasets`, `export_session`, `import_session`, `set_seed`, `get_seed` |
| Conversations | Session and conversation management via REST API (`/api/sessions`, `/api/conversations`) |

## Development

### Running Tests

```bash
cargo test -p p2a-core
```

### Building Documentation

```bash
cargo doc --no-deps --open
```

### Managing Disk Space

The Rust build cache (`target/`) can grow to 100GB+ during active development due to debug builds, incremental compilation, and benchmark artifacts. To reclaim disk space:

```bash
# Remove all build artifacts (will be regenerated on next build)
cargo clean

# Check current target size
du -sh target/
```

**Tip for contributors**: Run `cargo clean` periodically, especially after switching branches or completing major features. Debug builds in `target/debug/` are the largest consumers.

## Docker Deployment

Docker is provided for **deployment** rather than development. For active development, use native tools (`cargo run`, `npm run dev`) for faster iteration.

### Quick Start

```bash
# Build and run all services
docker compose up --build

# Or run in detached mode
docker compose up --build -d
```

This starts:
- **Backend** (p2a-mcp): http://localhost:8080
- **Frontend** (p2a-web): http://localhost:3000

### With Local LLM (Ollama)

```bash
# Include Ollama for local LLM support
docker compose --profile with-ollama up --build
```

### Health Check

```bash
curl http://localhost:8080/health   # Backend
curl http://localhost:3000          # Frontend
```

### Development Recommendation

For development, run services natively:

```bash
# Terminal 1: Backend
cargo run -p p2a-mcp --features full -- --transport http --host 127.0.0.1 --port 8080

# Terminal 2: Frontend
cd p2a-web && npm run dev
```

This provides faster rebuilds and hot module replacement.

## Technical Details

- **Matrix Operations**: Uses `faer` 0.22 for high-performance linear algebra (Cholesky decomposition, matrix inverse)
- **Statistical Distributions**: Uses `statrs` for t, F, chi-squared, and normal distributions
- **DataFrames**: Uses `polars` 0.52 for efficient data manipulation
- **Static Visualization**: Uses `plotters` for PNG chart generation
- **Interactive Visualization**: Uses `plotlars` (Plotly.js wrapper) for HTML charts with zoom/pan/hover
- **MCP Protocol**: Uses `rmcp` SDK for Model Context Protocol implementation
- **Database**: Uses `surrealdb` with embedded RocksDB for persistent conversation storage
- **Web Frontend**: Uses `dioxus` 0.7 compiling to WebAssembly with `web-sys` for browser APIs

## Paper

The `paper/` directory contains materials for a Journal of Statistical Software (JSS) article:

```
paper/
├── article.tex       # Main manuscript (LaTeX)
├── article.pdf       # Compiled paper
├── references.bib    # Bibliography (66+ entries)
├── Makefile          # Build: make, make clean
├── code/             # Benchmark analysis scripts
│   └── analyze_benchmarks.sh  # Generate benchmark figures
├── figures/          # Figures and logo
│   └── jsslogo.jpg
└── style/            # JSS LaTeX style files
    ├── jss.cls       # JSS document class
    ├── jss.bst       # JSS BibTeX style
    └── jss.pdf       # JSS style manual (author guidelines)
```

Build the paper with `make` in the `paper/` directory (requires pdfLaTeX and BibTeX).

To generate benchmark figures (requires p2a CLI and jq):
```bash
cd paper/code
./analyze_benchmarks.sh
```

## License

MIT

## Contributing

Contributions are welcome! Please see the development report (`DEVELOPMENT_REPORT.md`) for architecture details and the current state of the project.
