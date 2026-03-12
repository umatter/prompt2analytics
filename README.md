# prompt2analytics

[![CI](https://github.com/umatter/prompt2analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/umatter/prompt2analytics/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/umatter/prompt2analytics/branch/main/graph/badge.svg)](https://codecov.io/gh/umatter/prompt2analytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-blue.svg)](https://www.rust-lang.org/)

A comprehensive analytics toolkit exposing econometrics, machine learning, and visualization capabilities through multiple interfaces:
- **CLI (`p2a`)**: Direct command-line execution for scripted workflows
- **MCP Server**: Model Context Protocol integration for AI assistants (257 tools)
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

**Linux (Ubuntu/Debian) - x86_64:**
```bash
# Core dependencies
sudo apt-get install libopenblas-dev

# For Dioxus desktop app
sudo apt-get install libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev libgtk-3-dev libxdo-dev
```

**Linux (Ubuntu/Debian) - ARM64 (aarch64):**
```bash
# Core build dependencies (all required for p2a-mcp)
sudo apt-get install \
  libopenblas-dev \
  libssl-dev \
  pkg-config \
  build-essential \
  clang \
  libclang-dev

# For Dioxus desktop app
sudo apt-get install \
  libwebkit2gtk-4.1-dev \
  libsoup-3.0-dev \
  libjavascriptcoregtk-4.1-dev \
  libgtk-3-dev \
  libxdo-dev
```

**macOS:**
```bash
brew install openblas
```

**ARM64 Note:** On ARM64, debug builds use `opt-level = 1` (configured in `.cargo/config.toml`) to avoid linker relocation errors with large binaries. Release builds work without modification.

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

The `p2a` command provides direct access to all analytics functions. Use `--session` to enable reproducibility.

#### Data Management

```bash
# Load datasets (CSV, Parquet, Excel, Stata, SAS)
p2a --session s.json data load sales.csv --name sales
p2a --session s.json data load quarterly.parquet --name quarterly
p2a --session s.json data load survey.dta --name survey

# View data
p2a --session s.json data list
p2a --session s.json data describe sales
p2a --session s.json data head sales -n 20

# Generate random data for testing
p2a --session s.json data generate -n 1000 -d simdata \
    --columns '[{"name":"x","distribution":{"type":"normal","mean":0,"std":1}}]'

# Export data
p2a --session s.json data save sales --output results.parquet
```

#### Regression

```bash
# OLS with robust standard errors
p2a --session s.json reg ols mydata -y price -x sqft bedrooms --robust hc1

# Clustered standard errors
p2a --session s.json reg clustered mydata -y revenue -x employees --cluster firm_id

# HAC (Newey-West) standard errors for time series
p2a --session s.json reg hac mydata -y returns -x mkt_rf smb hml --lag 4

# Bootstrap standard errors
p2a --session s.json reg bootstrap mydata -y outcome -x treatment -B 1000

# Quantile regression
p2a --session s.json reg quantile mydata -y wage -x education experience --tau 0.5

# Diagnostics (VIF, Breusch-Pagan, Durbin-Watson)
p2a --session s.json reg diagnostics mydata -y price -x sqft bedrooms bathrooms
```

#### Panel Data

```bash
# Fixed effects
p2a --session s.json panel fe mydata -y revenue -x employees capital --entity firm_id

# Random effects
p2a --session s.json panel re mydata -y gdp -x investment --entity country --time year

# Hausman specification test
p2a --session s.json panel hausman mydata -y outcome -x treatment --entity id --time period

# High-dimensional fixed effects
p2a --session s.json panel hdfe mydata -y sales -x price --fe firm_id year

# FEGLM (GLM with HDFE)
p2a --session s.json panel feglm mydata -y count -x treatment --fe firm_id --family poisson

# Arellano-Bond GMM
p2a --session s.json panel gmm mydata -y growth -x investment --entity country --time year
```

#### Causal Inference

```bash
# Instrumental Variables (2SLS)
p2a --session s.json causal iv mydata -y wage -x education --instruments distance

# Difference-in-Differences
p2a --session s.json causal did mydata -y outcome --treat treatment --post post_period

# Staggered DiD (Callaway-Sant'Anna)
p2a --session s.json causal staggered-did mydata -y outcome --unit id --time period --treat first_treat

# Regression Discontinuity (Sharp)
p2a --session s.json causal rd mydata -y outcome --running score --cutoff 0

# Fuzzy RD
p2a --session s.json causal fuzzy-rd mydata -y outcome --running score --treat treatment --cutoff 0

# Propensity Score Matching
p2a --session s.json causal matching mydata -y outcome --treat treatment -x age income education

# IPW and Doubly Robust
p2a --session s.json causal ipw mydata -y outcome --treat treatment -x age income
p2a --session s.json causal doubly-robust mydata -y outcome --treat treatment -x age income

# Synthetic Control
p2a --session s.json causal synth mydata -y gdp --unit state --time year --treat california --treat-time 2000
```

#### Discrete Choice

```bash
# Logit and Probit
p2a --session s.json discrete logit mydata -y hired -x education experience age
p2a --session s.json discrete probit mydata -y default -x income debt_ratio

# Ordered models
p2a --session s.json discrete ologit mydata -y satisfaction -x service_quality price

# Multinomial logit
p2a --session s.json discrete mlogit mydata -y transport_mode -x income distance

# Count models
p2a --session s.json discrete negbin mydata -y accidents -x age speed
p2a --session s.json discrete zip mydata -y doctor_visits -x age income
```

#### Time Series

```bash
# ARIMA modeling and forecasting
p2a --session s.json ts arima mydata --col sales -p 1 -d 1 -q 1 --horizon 12

# VAR (Vector Autoregression)
p2a --session s.json ts var mydata --cols gdp inflation unemployment --lags 2

# GARCH volatility modeling
p2a --session s.json ts garch mydata --col returns -p 1 -q 1

# Holt-Winters forecasting
p2a --session s.json ts holt-winters mydata --col sales --seasonal 12

# STL decomposition
p2a --session s.json ts stl mydata --col sales --period 12

# Granger causality
p2a --session s.json ts granger mydata --cause money_supply --effect inflation --lags 4
```

#### Statistics

```bash
# T-tests
p2a --session s.json stats t-test-one mydata --col score --mu 0
p2a --session s.json stats t-test-two mydata --col1 treatment --col2 control
p2a --session s.json stats t-test-paired mydata --col1 before --col2 after

# ANOVA
p2a --session s.json stats anova mydata --response score --factor treatment
p2a --session s.json stats tukey mydata --response score --factor treatment

# Non-parametric tests
p2a --session s.json stats wilcoxon mydata --col1 group_a --col2 group_b
p2a --session s.json stats kruskal mydata --col score --group treatment

# Normality and independence
p2a --session s.json stats shapiro mydata --col residuals
p2a --session s.json stats box-test mydata --col residuals --lag 10

# Time series diagnostics
p2a --session s.json stats acf mydata --col returns --lag-max 20
p2a --session s.json stats pacf mydata --col returns --lag-max 15
```

#### Spatial Econometrics

```bash
# SAR (Spatial Lag Model)
p2a --session s.json spatial sar mydata -y price -x sqft bedrooms \
    --coord-x longitude --coord-y latitude -k 5

# SEM (Spatial Error Model)
p2a --session s.json spatial sem mydata -y price -x sqft \
    --coord-x lon --coord-y lat -k 5

# Moran's I test
p2a --session s.json spatial moran mydata -y residuals \
    --coord-x longitude --coord-y latitude -k 5
```

#### Survival Analysis

```bash
# Kaplan-Meier curves
p2a --session s.json survival km mydata -t time -e status -g treatment

# Log-rank test
p2a --session s.json survival log-rank mydata -t time -e status -g treatment

# Cox Proportional Hazards
p2a --session s.json survival cox mydata -t time -e status -x age treatment --robust
```

#### Machine Learning

```bash
# Clustering
p2a --session s.json ml kmeans mydata --cols x1 x2 x3 -k 5 --seed 42
p2a --session s.json ml dbscan mydata --cols x y --eps 0.5 --min-samples 5
p2a --session s.json ml hierarchical mydata --cols x1 x2 x3 -n 4 --linkage ward

# Dimensionality reduction
p2a --session s.json ml pca mydata --cols x1 x2 x3 x4 x5 -n 3
p2a --session s.json ml tsne mydata --cols x1 x2 x3 x4 --perplexity 30

# Supervised learning
p2a --session s.json ml random-forest mydata --cols x1 x2 x3 -y target --n-trees 100
p2a --session s.json ml svm mydata --cols x1 x2 x3 -y label -c 1.0
```

#### Data Munging

```bash
# Filter and select
p2a --session s.json munge filter mydata --column age --op gt --value 30
p2a --session s.json munge select mydata --columns id name income

# Transform
p2a --session s.json munge mutate mydata --new-col total --expr 'add:price:tax'
p2a --session s.json munge standardize mydata --columns x1 x2 x3

# Reshape
p2a --session s.json munge pivot mydata --index id --on year --values sales
p2a --session s.json munge melt mydata --id-vars id --value-vars jan feb mar

# Join and aggregate
p2a --session s.json munge join orders customers --on customer_id -t left
p2a --session s.json munge group-by mydata --by region --aggs revenue:sum units:mean

# Clean
p2a --session s.json munge drop-na mydata
p2a --session s.json munge fill-na mydata --method mean
p2a --session s.json munge deduplicate mydata --subset id
```

#### Visualization

```bash
# Static charts (PNG)
p2a --session s.json viz histogram mydata --col income -f hist.png --bins 50
p2a --session s.json viz scatter mydata -x age -y income -f scatter.png
p2a --session s.json viz line mydata -x date -y price -f timeseries.png
p2a --session s.json viz box mydata -y score -g treatment -f boxplot.png
p2a --session s.json viz heatmap mydata --cols x1 x2 x3 x4 -f corr.png
p2a --session s.json viz coefplot mydata -y outcome -x x1 x2 x3 -f coef.png
p2a --session s.json viz residuals mydata -y outcome -x x1 x2 -f resid.png

# Interactive charts (HTML with Plotly.js)
p2a --session s.json viz scatter-interactive mydata -x age -y income -f scatter.html
p2a --session s.json viz histogram-interactive mydata --col income -f hist.html
p2a --session s.json viz line-interactive mydata -x date -y price -f line.html
```

#### Session and Scripting

```bash
# Export session to reproducible script
p2a script export analysis.json -o analysis.sh

# Run a script
p2a script run analysis.sh
```

**Command categories:**
- `data` - Load, list, describe, preview, generate, export datasets
- `reg` - OLS, clustered SEs, HAC, bootstrap, quantile, diagnostics
- `panel` - Fixed effects, random effects, Hausman, HDFE, FEGLM, GMM
- `causal` - IV/2SLS, DiD, staggered DiD, RD, matching, IPW, synthetic control, TMLE
- `discrete` - Logit, probit, ordered, multinomial, count models (Poisson, NegBin, ZIP)
- `ts` - ARIMA, VAR, GARCH, Holt-Winters, STL, Granger causality
- `stats` - T-tests, ANOVA, non-parametric tests, ACF/PACF, normality tests
- `spatial` - SAR, SEM, SAC, Moran's I
- `survival` - Kaplan-Meier, Cox PH, AFT, log-rank test
- `ml` - K-means, DBSCAN, hierarchical, PCA, t-SNE, Random Forest, SVM
- `munge` - Filter, select, join, reshape, aggregate, clean
- `viz` - Static (PNG) and interactive (HTML) charts
- `script` - Export/run reproducible scripts

**Output formats:** `--format text` (default), `--format json`, `--format table`

### MCP Server

The MCP server exposes 257 analytics tools via the Model Context Protocol. Configure it in your MCP client (e.g., Claude Desktop):

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
│   ├── p2a-mcp/           # MCP server (257 tools)
│   │   └── db/            # SurrealDB persistence layer
│   └── p2a-dioxus/        # Cross-platform Dioxus app
│       ├── api/           # HTTP client and SSE streaming
│       ├── components/    # UI components
│       └── state/         # State management (Dioxus signals)
├── validation/            # Validation against R references
│   ├── [category]/        # Per-method validation docs (R code + expected values)
│   ├── datasets/          # Reference datasets (Grunfeld, Longley, Iris)
│   ├── scripts/           # R validation scripts
│   └── run_validation.sh  # Master validation runner
├── performance/           # Benchmark framework and results
│   └── comparisons/
│       ├── run_all.sh     # Full pipeline orchestration
│       └── r_comparison/  # R benchmark scripts + merged results
│           ├── benchmark_*.R     # 67 R benchmark scripts
│           ├── merge_results.R   # Merge Rust JSON + R CSV
│           └── results/          # comparison_speed.csv, comparison_memory.csv
└── paper/                 # JSS article materials
    ├── code/              # Figure/table generation scripts (R)
    ├── figures/           # Generated benchmark figures (PDF + PNG)
    └── tables/            # Generated LaTeX tables
```

## MCP Tools (257 total)

The MCP server exposes 257 analytics tools. Key categories include:

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

For a complete list, see the tool definitions in `crates/p2a-mcp/src/tools/handlers/`.

## Validation

All methods are validated against reference R implementations to ensure numerical correctness. Validation uses a two-layer approach:

1. **Rust unit tests** (`test_validate_*` prefix) compare outputs to known R results at tight tolerances
2. **Validation documents** (`validation/`) record the R code, expected values, and precision analysis for each method

### Running Validation Tests

```bash
# All validation tests
cargo test -p p2a-core -- test_validate

# Specific method
cargo test -p p2a-core -- test_validate_ols
cargo test -p p2a-core -- test_validate_hdfe

# Full validation (Rust + R scripts)
./validation/run_validation.sh

# Rust only (faster)
./validation/run_validation.sh --rust-only

# Filter by category
./validation/run_validation.sh --category stats
```

### Tolerance Guidelines

| Sample Size | Coefficient Tolerance | SE Tolerance |
|-------------|----------------------|--------------|
| n < 100     | 1e-6                 | 1e-5         |
| n = 100-1000| 1e-8                 | 1e-6         |
| n > 1000    | 1e-10                | 1e-8         |

For iterative methods (HDFE, MLE), slightly larger differences are expected due to convergence criteria.

### Validation Directory Structure

```
validation/
├── README.md                    # Validation framework overview
├── VALIDATION_STATUS.md         # Current coverage report
├── run_validation.sh            # Master validation runner
├── reference_implementations.md # Catalog of R/Python reference packages
├── regression/                  # OLS, robust SEs, GLS, LOESS, sensemakr, E-value
├── econometrics/                # Panel, IV, DiD, discrete choice, spatial, survival
│   └── timeseries/              # VAR, VARMA, VECM, IRF
├── forecasting/                 # ARIMA, MSTL, Holt-Winters, changepoint
├── stats/                       # 50+ statistical tests (t-test, ANOVA, Fisher, etc.)
├── diagnostics/                 # JB, BP, DW, VIF, Breusch-Godfrey, RESET
├── ml/                          # K-means, DBSCAN, PCA, t-SNE, Random Forest
├── multivariate/                # MANOVA, CCA, factor analysis
├── linalg/                      # Matrix operation validation
├── spatial/                     # Moran's I, SAR, SEM
├── datasets/                    # Reference datasets (Grunfeld, Longley, Iris)
├── scripts/                     # R validation scripts
└── reports/                     # Generated validation reports
```

Each validation document records: method overview, reference R/Python code, test cases with expected values, numerical precision analysis, and known differences.

## Performance Benchmarking

Performance is measured with a custom benchmarking framework that captures distribution statistics (min, p25, median, p75, max, mean, std) and memory usage, matching the output format of R's `bench::mark()` package for direct comparison.

### Pipeline Overview

```
Rust benchmarks (19 files)          R benchmarks (67 scripts)
         │                                    │
         ▼                                    ▼
  rust_comprehensive_*.json             r_*_*.csv files
         │                                    │
         └──────────┬─────────────────────────┘
                    ▼
            merge_results.R
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
  comparison_   comparison_   validation_
  speed.csv     memory.csv    coverage.csv
         │          │
         ▼          ▼
  generate_paper_figures.R    generate_paper_tables.R
         │                              │
         ▼                              ▼
  paper/figures/*.pdf           paper/tables/*.tex
```

### Running Benchmarks

```bash
# Full pipeline (Rust validation + benchmarks + R benchmarks + merge)
./performance/comparisons/run_all.sh

# Rust benchmarks only
cargo bench -p p2a-core --bench comprehensive_benchmarks

# R benchmarks only
cd performance/comparisons/r_comparison && Rscript r_benchmark_runner.R

# Merge existing results (no new benchmark runs)
./performance/comparisons/run_all.sh --merge-only

# Quick mode (validation + comprehensive R benchmarks only)
./performance/comparisons/run_all.sh --quick
```

### Rust Benchmarks

**Location**: `crates/p2a-core/benches/` (19 files)

The primary benchmark file is `comprehensive_benchmarks.rs`, which covers all method categories. Specialized benchmark files exist for deeper per-module coverage (regression, econometrics, forecasting, hypothesis tests, ML, spatial, causal, etc.).

Key design choices:
- Custom `bench_utils.rs` runner (not Criterion) to produce distribution statistics compatible with R's `bench::mark()`
- Reproducible data generation via seeded ChaCha8Rng (seed=42)
- Memory tracking via `memory_stats` crate (physical memory before/after/peak)
- JSON output for automated merging with R results

### R Benchmarks

**Location**: `performance/comparisons/r_comparison/` (67 `benchmark_*.R` scripts)

Each R script benchmarks the equivalent R function(s) using `bench::mark()` with identical data-generating processes (same seeds, same sample sizes). Scripts cover: regression (`lm`, `sandwich`, `plm`, `lfe`), discrete choice (`glm`, `MASS`), time series (`forecast`, `stats::stl`), spatial (`spdep`), survival (`survival`), ML (`kmeans`, `dbscan`, `prcomp`), and 50+ statistical tests.

### Merge and Comparison

**Script**: `performance/comparisons/r_comparison/merge_results.R`

This script:
1. Loads all timestamped R CSV results and Rust JSON results
2. Normalizes method names across languages (e.g., `lagsarlm` → `SAR`, `ols_lm` → `OLS`)
3. Matches on method + sample size (exact match, then fuzzy within 2x)
4. Computes speedup factors: `speedup = R_median_time / Rust_median_time`
5. Assigns methods to modules (Regression, Panel, Stats, ML, etc.)
6. Outputs: `comparison_speed.csv`, `comparison_memory.csv`, `validation_coverage.csv`

### Paper Figures and Tables

**Location**: `paper/code/` (18 R scripts)

```bash
# Generate all figures (reads comparison CSVs, writes to paper/figures/)
cd paper/code && Rscript generate_paper_figures.R

# Generate all tables (reads comparison CSVs, writes to paper/tables/)
cd paper/code && Rscript generate_paper_tables.R
```

**Generated figures** (`paper/figures/`):
- `benchmark_speedup_violin.pdf` — Violin plots of speedup distribution by module
- `benchmark_boxplots.pdf` — Box plots of speedup by module
- `benchmark_histogram.pdf` — Histogram of speedup factors with median line
- `benchmark_memory.pdf` — Memory usage ratio comparison
- `benchmark_speedup.pdf` — Log-scale speedup by method

**Generated tables** (`paper/tables/`):
- `tab_speedup_by_module.tex` — Per-module speedup summary (min, median, mean, max)
- `tab_benchmark_summary.tex` — Representative method benchmarks with execution times

### R Prerequisites for Benchmarks

```bash
# Core packages (required)
install.packages(c("bench", "sandwich", "plm", "lfe"))

# Extended packages (for full coverage)
install.packages(c("forecast", "changepoint", "survival", "MatchIt",
                   "WeightIt", "randomForest", "e1071", "dbscan",
                   "Rtsne", "spdep", "rugarch", "dlm", "tseries"))
```

### Results Directory

```
performance/comparisons/r_comparison/results/
├── comparison_speed.csv          # Merged Rust vs R speed comparison
├── comparison_memory.csv         # Merged memory comparison
├── validation_coverage.csv       # Method coverage matrix
└── *.log                         # Execution logs from run_all.sh
```

Raw timestamped results (`r_*_2026*.csv`, `rust_comprehensive_2026*.json`) are gitignored. Only the merged comparison CSVs are tracked.

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

The `paper/` directory contains materials for a Journal of Statistical Software (JSS) article describing the chat-first data analytics approach.

### Building the Paper

```bash
# Build PDF (requires pdfLaTeX and BibTeX)
cd paper && pdflatex article-jss && bibtex article-jss && pdflatex article-jss && pdflatex article-jss
```

### Reproducing Benchmark Exhibits

All figures and tables in the paper are generated from benchmark data via R scripts. To reproduce from scratch:

```bash
# Step 1: Run Rust benchmarks
cargo bench -p p2a-core --bench comprehensive_benchmarks

# Step 2: Run R benchmarks (requires R + benchmark packages)
cd performance/comparisons/r_comparison && Rscript r_benchmark_runner.R

# Step 3: Merge results
Rscript merge_results.R

# Step 4: Generate figures and tables
cd ../../paper/code
Rscript generate_paper_figures.R    # → paper/figures/*.pdf
Rscript generate_paper_tables.R     # → paper/tables/*.tex

# Or run the full pipeline in one command:
./performance/comparisons/run_all.sh
```

### Reproducing End-to-End Evaluation

The paper reports two evaluations:
1. **96-prompt single-turn evaluation** across 6 models (GPT-4.1 Mini, Sonnet 4.6, Haiku 4.5, Gemini 2.5 Flash, Ministral 3B, Llama 4 Scout), scored on tool selection, parameter extraction, numerical correctness, and interpretation quality.
2. **Chrome-based multi-turn evaluation** with Claude Sonnet 4.6 (8 conversations, 30 turns, 96.7% adequate).

```bash
# Run single-turn evaluation (requires API keys for each provider)
cd paper/code/e2e_eval
python3 run_evaluation.py --models gpt-4.1-mini claude-sonnet-4.6 claude-haiku-4.5

# Re-score existing results
python3 rescore.py

# Generate evaluation figures and tables
cd paper/code
Rscript generate_e2e_figures.R
```

### Paper Directory Structure

```
paper/
├── article-jss.tex                # Main JSS wrapper (title, abstract, bibliography)
├── paper.tex                      # Section includes
├── sections/                      # Paper sections
│   ├── introduction_new.tex       # Introduction
│   ├── defining.tex               # Background and design principles
│   ├── tools.tex                  # Software implementation
│   ├── examples.tex               # Worked examples
│   ├── evaluation_new.tex         # Evaluation (96-prompt e2e + multi-turn)
│   ├── deployment.tex             # Performance benchmarks and local deployment
│   ├── discussion.tex             # Discussion and appendix TOC
│   ├── appendices.tex             # Appendices A–G
│   └── e2e_eval_appendix.tex      # Appendix H (e2e evaluation protocol)
├── code/
│   ├── generate_paper_figures.R   # Benchmark figure generator
│   ├── generate_paper_tables.R    # Benchmark table generator
│   ├── generate_e2e_figures.R     # Evaluation figure generator
│   └── e2e_eval/                  # End-to-end evaluation framework
│       ├── test_cases.json        # 96 evaluation prompts with expected tools
│       ├── run_evaluation.py      # Evaluation runner (API calls + scoring)
│       ├── rescore.py             # Re-score existing results
│       ├── generate_datasets.R    # Generate evaluation datasets
│       ├── PROTOCOL.md            # Evaluation protocol documentation
│       └── results/               # Result JSONs per model (gitignored)
├── figures/                       # Generated PDF/PNG figures
├── tables/                        # Generated LaTeX tables
└── references.bib                 # Bibliography
```

## License

MIT

## Documentation

- `validation/README.md` - Validation framework and method index
- `validation/VALIDATION_STATUS.md` - Current validation coverage report
- `docs/guides/TESTING.md` - Test runtime expectations
- `docs/guides/DATA_SECURITY.md` - Data write locations, privacy
- `docs/security/PROMPT_INJECTION.md` - MCP security considerations

## Contributing

Contributions are welcome! See:
- `CLAUDE.md` - Development guidance for Claude Code
- `DEVELOPMENT_REPORT.md` - Architecture details and current status
- `validation/README.md` - How to add validation tests
