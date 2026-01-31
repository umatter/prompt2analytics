# prompt2analytics

[![CI](https://github.com/umatter/prompt2analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/umatter/prompt2analytics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/umatter/prompt2analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/umatter/prompt2analytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-blue.svg)](https://www.rust-lang.org/)

A comprehensive analytics toolkit exposing econometrics, machine learning, and visualization capabilities through multiple interfaces:
- **CLI (`p2a`)**: Direct command-line execution for scripted workflows
- **MCP Server**: Model Context Protocol integration for AI assistants (250+ tools)
- **Dioxus App**: Cross-platform frontend (web, desktop) with LLM-powered natural language analysis

**Requirements**: Rust 1.85+ (edition 2024)

## Features

### Econometrics (Pure Rust)
- **OLS Regression** with robust standard errors (HC0-HC3) and clustered SEs
- **Panel Data**: Fixed Effects, Random Effects, Hausman specification test, HDFE
- **Instrumental Variables**: 2SLS with first-stage diagnostics
- **Causal Inference**: Difference-in-Differences, Regression Discontinuity (Sharp/Fuzzy RD)
- **Discrete Choice**: Logit, Probit, FEGLM (GLM with high-dimensional fixed effects)
- **Regression Diagnostics**: Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF
- **Identification Diagnostics**: Automated checks for IV, DiD, RD, and matching estimators with severity-ranked warnings and remediation suggestions

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

### Data Management
- **File formats**: CSV, Parquet, Excel (.xlsx/.xls), Stata (.dta), SAS (.sas7bdat)
- **Databases**: SQLite, DuckDB (with direct file querying)
- **LLM-assisted cleaning**: Quality profiling, preview/verify operations, rollback-enabled sessions

### Export Formats
- **LaTeX**: Publication-ready regression tables (OLS, Panel, Discrete)
- **Markdown**: GitHub-compatible tables for documentation
- **HTML**: Self-contained tables with embedded CSS
- **CSV**: Generic export via `CsvExport` trait for all result types

### Command-Line Interface
- Full access to all analytics via `p2a` binary
- Session recording for reproducibility
- Script export for sharing and automation
- JSON output format for programmatic use

### Dioxus Cross-Platform App
- Pure Rust frontend compiled to WASM (web) or native (desktop)
- Multi-provider LLM integration (Ollama, Anthropic, OpenAI)
- Conversation history with SurrealDB persistence
- Tool call transparency with expandable details

## Installation

### Prerequisites

**Linux (Ubuntu/Debian):**
```bash
# Core dependencies
sudo apt-get install libopenblas-dev

# For Dioxus desktop app
sudo apt-get install libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev libgtk-3-dev libxdo-dev
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

# Build the Dioxus app (web - requires dioxus-cli)
cargo install dioxus-cli
rustup target add wasm32-unknown-unknown
cd crates/p2a-dioxus && dx build --release && cd ../..

# Build the Dioxus app (desktop)
cd crates/p2a-dioxus && dx build --release --platform desktop && cd ../..
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

# Create interactive visualizations (HTML with Plotly.js)
p2a --session analysis.json viz scatter-interactive mydata -x income -y spending -f scatter.html

# Export session to reproducible bash script
p2a script export analysis.json -o analysis.sh
```

**Command categories:**
- `data` - Load, list, describe, preview datasets
- `reg` - OLS, clustered SEs, diagnostics
- `panel` - Fixed effects, random effects, Hausman test, HDFE
- `causal` - IV/2SLS, difference-in-differences, regression discontinuity
- `discrete` - Logit, probit
- `ts` - ARIMA, MSTL, VAR
- `ml` - K-means, PCA, t-SNE, Random Forest
- `viz` - Static (PNG) and interactive (HTML) charts
- `script` - Export/run reproducible scripts

**Output formats:** `--output text` (default), `--output json`, `--output table`

### MCP Server

The MCP server exposes 60+ analytics tools via the Model Context Protocol. Configure it in your MCP client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "prompt2analytics": {
      "command": "/path/to/target/release/p2a-mcp"
    }
  }
}
```

### Dioxus App (Web, Desktop)

A cross-platform frontend built with Dioxus 0.7:

```bash
# Install Dioxus CLI (if not already installed)
cargo install dioxus-cli

# Install WASM target for web builds
rustup target add wasm32-unknown-unknown

# Start backend in one terminal
cargo run -p p2a-mcp --features full -- --transport http --port 8080 --cors-permissive

# Start Dioxus dev server (web) in another terminal
cd crates/p2a-dioxus
dx serve

# Or run as desktop app
dx serve --platform desktop
```

Open http://localhost:8080 in your browser (web) or the native window (desktop). Features:
- Chat interface with streaming LLM responses
- Support for Ollama, Anthropic, and OpenAI providers
- Conversation management with persistent history (SurrealDB)
- Dataset sidebar with live metadata view
- Tool call transparency ("Rust Analytics" indicator)
- Environment variable detection for API keys
- Markdown rendering for assistant messages

### Example MCP Tool Calls

```
# Load a dataset
load_dataset path:/path/to/data.csv

# Create an inline dataset
create_dataset name:test_data csv_content:"x,y\n1,2\n2,4\n3,6"

# Run OLS regression
regression_ols dataset:mydata y:price x:sqft,bedrooms,bathrooms

# Generate a scatter plot
viz_scatter dataset:mydata x_column:sqft y_column:price

# Run panel fixed effects
panel_fixed_effects dataset:panel y:outcome x:treatment entity_col:firm time_col:year

# Data quality profiling (for LLM-assisted cleaning)
data_quality_profile dataset:mydata
```

## Architecture

```
prompt2analytics/
├── crates/
│   ├── p2a-core/          # Core analytics library (all algorithms)
│   │   ├── data/          # Data loading, quality profiling, cleaning
│   │   ├── stats/         # Descriptive statistics, correlation
│   │   ├── regression/    # OLS, diagnostics
│   │   ├── econometrics/  # Panel, IV, DiD, RD, discrete choice, time series
│   │   ├── diagnostics/   # Identification diagnostics (IV, DiD, RD, matching)
│   │   ├── forecasting/   # ARIMA, MSTL, changepoint
│   │   ├── ml/            # Clustering, PCA, t-SNE, Random Forest, SVM
│   │   ├── visualization/ # Static (plotters) and interactive (plotlars) charts
│   │   ├── export/        # LaTeX, Markdown, HTML, CSV export
│   │   ├── linalg/        # Matrix operations (via faer)
│   │   └── traits/        # LinearEstimator trait
│   ├── p2a-cli/           # CLI binary (`p2a`)
│   ├── p2a-mcp/           # MCP server (60+ tools)
│   │   └── db/            # SurrealDB persistence layer
│   └── p2a-dioxus/        # Cross-platform Dioxus app
│       ├── api/           # HTTP client and SSE streaming
│       ├── components/    # UI components
│       └── state/         # State management (Dioxus signals)
├── validation/            # Validation against R/Python references
├── performance/           # Benchmark framework and results
└── paper/                 # JSS article materials
```

## MCP Tools (256 total)

The MCP server exposes 256 analytics tools. Key categories include:

| Category | Example Tools |
|----------|---------------|
| Data Management | `load_dataset`, `create_dataset`, `list_datasets`, `describe_dataset`, `head_dataset`, `export_dataset` |
| Data Quality | `data_quality_profile`, `preview_cleaning`, `verify_cleaning`, `suggest_cleaning` |
| Cleaning Sessions | `cleaning_session_start`, `cleaning_session_apply`, `cleaning_rollback` |
| Statistics | `compute_correlation`, `hypothesis_t_test`, `hypothesis_chisq_*`, `hypothesis_wilcoxon`, `anova_*` |
| Regression | `regression_ols`, `regression_gls`, `regression_clustered`, `regression_quantile`, `regression_nls` |
| Panel | `panel_fixed_effects`, `panel_random_effects`, `hausman_test`, `panel_gmm`, `feglm` |
| IV/2SLS | `iv_2sls`, `iv_first_stage`, `iv_diagnostics` |
| Causal Inference | `diff_in_diff`, `staggered_did`, `rd_estimate`, `rd_fuzzy`, `synth_control`, `matching_*`, `ipw_*` |
| Discrete Choice | `logit`, `probit`, `multinomial_logit`, `ordered_probit`, `poisson`, `negbin`, `zip`, `zinb` |
| Time Series | `ts_arima_*`, `ts_var*`, `ts_vecm`, `ts_mstl`, `ts_changepoint`, `ts_garch`, `ts_kalman_*` |
| Spatial | `spatial_neighbors`, `spatial_sar`, `spatial_sem`, `spatial_sac`, `moran_test` |
| ML | `ml_kmeans`, `ml_dbscan`, `ml_hierarchical`, `ml_pca`, `ml_tsne`, `ml_random_forest`, `ml_svm` |
| Database | `db_sqlite_query`, `db_duckdb_query`, `db_*_tables`, `db_*_schema` |
| Visualization | `viz_histogram`, `viz_scatter`, `viz_line`, `viz_boxplot`, `viz_heatmap`, `viz_*_interactive` |
| Power Analysis | `power_t_test`, `power_prop_test`, `power_anova_test` |
| Utilities | `generate_random_data`, `set_seed`, `generate_report` |

For a complete list, see the tool definitions in `crates/p2a-mcp/src/server.rs`.

## Development

### Running Tests

```bash
cargo test                              # All tests
cargo test -p p2a-core                  # Core library only
cargo test -p p2a-core -- test_validate # Validation tests
```

### Linting

```bash
cargo clippy --all-targets --all-features
cargo fmt --check
```

### Building Documentation

```bash
cargo doc --no-deps --open
```

### Managing Disk Space

The Rust build cache (`target/`) can grow large. To reclaim disk space:

```bash
cargo clean
du -sh target/
```

## Docker Deployment

Docker is provided for **deployment** rather than development.

```bash
# Build and run the backend
docker compose up --build

# With local LLM (Ollama)
docker compose --profile with-ollama up --build

# Health check
curl http://localhost:8080/health
```

For development, run services natively for faster iteration:

```bash
# Terminal 1: Backend
cargo run -p p2a-mcp --features full -- --transport http --host 127.0.0.1 --port 8080 --cors-permissive

# Terminal 2: Frontend
cd crates/p2a-dioxus && dx serve
```

## Technical Details

| Component | Library | Version |
|-----------|---------|---------|
| Matrix Operations | `faer` | 0.22 |
| Statistical Distributions | `statrs` | 0.18 |
| DataFrames | `polars` | 0.52 |
| Static Visualization | `plotters` | 0.3 |
| Interactive Visualization | `plotlars` | 0.11 |
| MCP Protocol | `rmcp` | 0.8 |
| Database | `surrealdb` | embedded RocksDB |
| Web Frontend | `dioxus` | 0.7 |

## Paper

The `paper/` directory contains materials for a Journal of Statistical Software (JSS) article:

```bash
# Build the paper (requires pdfLaTeX and BibTeX)
cd paper && make

# Generate benchmark figures (requires p2a CLI and jq)
cd paper/code && ./analyze_benchmarks.sh
```

## License

MIT

## Documentation

- `docs/guides/TESTING.md` - Test runtime expectations, validation framework
- `docs/guides/DATA_SECURITY.md` - Data write locations, privacy
- `docs/security/PROMPT_INJECTION.md` - MCP security considerations

## Contributing

Contributions are welcome! See:
- `CLAUDE.md` - Development guidance for Claude Code
- `DEVELOPMENT_REPORT.md` - Architecture details and current status
- `validation/` - How to add validation tests
