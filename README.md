# prompt2analytics

[![CI](https://github.com/umatter/prompt2analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/umatter/prompt2analytics/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-yellow.svg)](#license)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-blue.svg)](https://www.rust-lang.org/)

A comprehensive analytics toolkit exposing econometrics, machine learning, and visualization capabilities through multiple interfaces:
- **CLI (`p2a`)**: Direct command-line execution for scripted workflows
- **MCP Server**: Model Context Protocol integration for AI assistants (270 tools)
- **Dioxus App**: Cross-platform frontend (web, desktop) with LLM-powered natural language analysis

**🌐 Live demo (bring your own key):** [p2a.qamelab.org](https://p2a.qamelab.org) — the hosted web app. You supply your own LLM API key in the browser; it is never stored server-side. This is a small **demo deployment**: it runs on a modest single instance (limited performance) and accepts uploaded files of up to about 1.5 MB (larger ones are refused with HTTP 413), so expect slower responses and use small datasets. Each tool call stops after 60 seconds, each session holds up to about 200 MB of data and runs one request at a time, and other large or costly requests are refused with a message that names the limit and what to make smaller. For full performance and large data, run the desktop app or self-host the backend container `ghcr.io/umatter/prompt2analytics/backend` (see [docs/guides/DEPLOYMENT.md](docs/guides/DEPLOYMENT.md)).

The demo runs the latest tagged release (check [`/health`](https://p2a-api.qamelab.org/health)) in open public mode with the hosted resource profile: there are no user accounts, per-visitor isolation relies on unguessable session IDs, the session-list and file-browser endpoints are disabled, and idle sessions end after 60 minutes (sooner when the server is full) (threat model and limits in [docs/guides/DEPLOYMENT.md](docs/guides/DEPLOYMENT.md)). Two caveats follow from it being a **public, shared instance**: do **not** upload confidential or personal data — for sensitive work use the desktop app or a self-hosted backend, the deployment mode the project is designed around — and treat sessions as **ephemeral**: the instance has no persistent volume, so uploaded data and results disappear whenever it restarts or is redeployed.

**Requirements**: Rust 1.95+ (edition 2024)

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
git clone https://github.com/umatter/prompt2analytics.git
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
p2a --session s.json reg hac mydata -y returns -x mkt_rf smb hml --lags 4

# Quantile regression
p2a --session s.json reg quantreg mydata -y wage -x education experience --tau 0.5

# Diagnostics (VIF, Breusch-Pagan, Durbin-Watson)
p2a --session s.json reg diagnostics mydata -y price -x sqft bedrooms bathrooms
```

Bootstrap covariances are an MCP tool (`regression_bootstrap_cov`), not a CLI command.

#### Panel Data

```bash
# Fixed effects
p2a --session s.json panel fe mydata -y revenue -x employees capital --entity firm_id

# Random effects
p2a --session s.json panel re mydata -y gdp -x investment --entity country

# Hausman specification test
p2a --session s.json panel hausman mydata -y outcome -x treatment --entity id

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
p2a --session s.json causal iv mydata -y wage --endog education --instruments distance

# Difference-in-Differences
p2a --session s.json causal did mydata -y outcome --treat treatment --post post_period

# Staggered DiD (Callaway-Sant'Anna)
p2a --session s.json causal staggered-did mydata -y outcome --unit id --time period --treatment-time first_treat

# Regression Discontinuity (Sharp)
p2a --session s.json causal rd mydata -y outcome --running score --cutoff 0

# Fuzzy RD
p2a --session s.json causal fuzzy-rd mydata -y outcome --running score --treatment treatment --cutoff 0

# Propensity Score Matching
p2a --session s.json causal matching mydata --treatment treatment -x age income education

# IPW and Doubly Robust
p2a --session s.json causal ipw mydata -y outcome --treatment treatment -x age income
p2a --session s.json causal doubly-robust mydata -y outcome --treatment treatment -x age income

# Synthetic Control
p2a --session s.json causal synth mydata -y gdp --unit state --time year --treated california --treatment-time 2000
```

#### Discrete Choice

```bash
# Logit and Probit
p2a --session s.json discrete logit mydata -y hired -x education experience age
p2a --session s.json discrete probit mydata -y default -x income debt_ratio

# Ordered models
p2a --session s.json discrete o-logit mydata -y satisfaction -x service_quality price

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
p2a --session s.json ts holt-winters mydata --col sales --period 12

# STL decomposition
p2a --session s.json ts stl mydata --col sales --period 12

# Granger causality
p2a --session s.json ts granger mydata --x money_supply --y inflation --lags 4
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

**Commands** (`p2a <command> -h` for each; every option is in the generated reference at the end of
[docs/guides/CLI_REFERENCE.md](docs/guides/CLI_REFERENCE.md#command-reference)):

<!-- BEGIN GENERATED: p2a command list (cargo run -p p2a-cli --example generate_cli_reference; do not edit by hand) -->
- `data`: Data loading and inspection. Subcommands: `load`, `list`, `describe`, `head`, `generate`, `save`
- `munge`: Data munging (filter, join, reshape, clean, aggregate). Subcommands: `filter`, `select`, `drop`, `rename`, `sort`, `mutate`, `sample`, `join`, `concat`, `group-by`, `value-counts`, `pivot`, `melt`, `drop-na`, `fill-na`, `deduplicate`, `lag`, `lead`, `diff`, `standardize`, `bin`, `cast`, `one-hot`
- `regression` (alias `reg`): Regression analysis. Subcommands: `ols`, `clustered`, `diagnostics`, `quantreg`, `loess`, `nls`, `gls`, `step`, `smooth-spline`, `supsmu`, `line`, `hac`
- `panel`: Panel data estimation. Subcommands: `fe`, `re`, `hausman`, `hdfe`, `feglm`, `gmm`, `pvcm`, `pmg`, `gls`, `unit-root`
- `causal`: Causal inference methods. Subcommands: `iv`, `did`, `synth`, `ipw`, `doubly-robust`, `mediation`, `rd`, `fuzzy-rd`, `staggered-did`, `tmle`, `gsynth`, `matching`, `stdreg`
- `discrete`: Discrete choice models. Subcommands: `logit`, `probit`, `o-logit`, `o-probit`, `mlogit`, `negbin`, `poisson`, `zip`, `zinb`, `hurdle`, `clogit`, `mixed-logit`
- `stats`: Statistical tests and descriptive statistics. Subcommands: `t-test-one`, `t-test-two`, `t-test-paired`, `chisq-gof`, `chisq-indep`, `fisher`, `shapiro`, `wilcoxon`, `bartlett`, `kruskal`, `fivenum`, `iqr`, `mad`, `acf`, `anova`, `anova-two`, `tukey`, `friedman`, `pacf`, `ccf`, `box-test`, `ks-test`, `pp-test`, `power-t`
- `timeseries` (alias `ts`): Time series analysis. Subcommands: `arima`, `mstl`, `var`, `varma`, `vecm`, `irf`, `changepoint`, `garch`, `holt-winters`, `granger`, `ar`, `stl`, `decompose`, `struct-ts`, `causal-impact`, `spectrum`
- `survival`: Survival analysis. Subcommands: `km`, `log-rank`, `cox`, `aft`, `competing-risks`
- `spatial`: Spatial econometrics. Subcommands: `sar`, `sem`, `sac`, `moran`
- `machine-learning` (alias `ml`): Machine learning. Subcommands: `kmeans`, `pca`, `tsne`, `random-forest`, `dbscan`, `hierarchical`, `svm`
- `visualize` (alias `viz`): Visualization (output to file). Subcommands: `histogram`, `scatter`, `line`, `box`, `heatmap`, `coefplot`, `residuals`, `dendrogram`, `event-study`, `irf`, `scatter-interactive`, `histogram-interactive`, `line-interactive`
- `script`: Script generation and session management. Subcommands: `export`, `history`, `run`
- `chat`: Interactive chat with a p2a-mcp server (natural language analytics)
- `smoke-test`: Run a smoke test to verify the CLI works correctly
<!-- END GENERATED: p2a command list -->

**Output formats:** `-F text` (default), `-F json`, `-F table` (printed like `text` in this version)

### MCP Server

The MCP server exposes 270 analytics tools via the Model Context Protocol. Configure it in your MCP client (e.g., Claude Desktop):

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

Open the URL printed by `dx serve` in your browser (web), or use the native window (desktop). The backend MCP server runs separately on port 8080. Features:
- Chat interface with streaming LLM responses
- Support for Ollama, Anthropic, and OpenAI providers
- Conversation management with persistent history (SurrealDB)
- Dataset sidebar with live metadata view
- Tool call transparency ("Rust Analytics" indicator)
- Environment variable detection for API keys
- Markdown rendering for assistant messages

### Example MCP Tool Calls

Chat shorthand (see `docs/guides/MCP_TOOL_EXAMPLES.md`):

```text
/load_dataset path:data.csv
/create_dataset name:test_data csv_content:"x,y\n1,2.1\n2,3.9\n3,6.2"
/regression_ols dataset:mydata y:price x:sqft,bedrooms,bathrooms
/viz_scatter dataset:mydata x_column:sqft y_column:price
/panel_fixed_effects dataset:panel y:outcome x:treatment entity_var:firm
/data_quality_profile dataset:mydata
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
│   ├── p2a-mcp/           # MCP server (270 tools)
│   │   └── db/            # SurrealDB persistence layer
│   └── p2a-dioxus/        # Cross-platform Dioxus app
│       ├── api/           # HTTP client and SSE streaming
│       ├── components/    # UI components
│       └── state/         # State management (Dioxus signals)
├── validation/            # Validation against R references
│   ├── [category]/        # Per-method validation docs (R code + expected values)
│   ├── r/                 # R reference harness and cases (run.R --check | --write)
│   ├── expected/          # R reference cases (JSON) and legacy CSV outputs
│   ├── methods/           # Registry: which tests pin which R cases
│   ├── datasets/          # Reference data (Grunfeld, ...); generated/ written by R cases
│   ├── scripts/           # Legacy R validation scripts
│   └── run_validation.sh  # Master validation runner
└── docs/                  # Guides, design notes, security docs
```

## MCP Tools

<!-- BEGIN GENERATED: MCP tool table (cargo run -p p2a-mcp --example generate_tool_docs; do not edit by hand) -->
The server registers 270 tools in a default build (263 without the `database` feature, which CI's feature set leaves out). [docs/mcp/tools-reference.md](docs/mcp/tools-reference.md) describes each one and names its R equivalent.

| Category | Tools | Names |
|---|---|---|
| Data | 8 | `compare_datasets`, `create_dataset`, `describe_dataset`, `export_dataset`, `head_dataset`, `list_datasets`, `load_dataset`, `upload_dataset` |
| Cleaning | 10 | `cleaning_rollback`, `cleaning_session_apply`, `cleaning_session_checkpoints`, `cleaning_session_start`, `cleaning_session_status`, `data_quality_profile`, `list_cleaning_sessions`, `preview_cleaning`, `suggest_cleaning`, `verify_cleaning` |
| Munging | 34 | `batch_process`, `munge_bin`, `munge_cast`, `munge_concat`, `munge_deduplicate`, `munge_diff`, `munge_drop_columns`, `munge_drop_na`, `munge_fill_na`, `munge_filter`, `munge_group_by`, `munge_join`, `munge_lag_lead`, `munge_melt`, `munge_mutate`, `munge_one_hot_encode`, `munge_pivot`, `munge_rename`, `munge_sample`, `munge_select`, `munge_sort`, `munge_standardize`, `munge_value_counts`, `str_concat`, `str_length`, `str_regex_count`, `str_regex_extract`, `str_regex_replace`, `str_replace_value`, `str_split`, `str_substring`, `str_to_lowercase`, `str_to_uppercase`, `str_trim` |
| Descriptive | 12 | `compute_correlation`, `descriptive_isoreg`, `descriptive_medpolish`, `stats_approx`, `stats_cov_wt`, `stats_density`, `stats_ecdf`, `stats_fivenum`, `stats_iqr`, `stats_mad`, `stats_spline`, `stats_weighted_mean` |
| Statistics | 34 | `anova_manova`, `anova_one_way`, `anova_tukey_hsd`, `anova_two_way`, `hypothesis_bartlett_test`, `hypothesis_chisq_gof`, `hypothesis_chisq_independence`, `hypothesis_cor_test`, `hypothesis_fisher_exact`, `hypothesis_friedman`, `hypothesis_kruskal_wallis`, `hypothesis_ks_test`, `hypothesis_mantelhaen`, `hypothesis_mcnemar`, `hypothesis_mood_test`, `hypothesis_oneway`, `hypothesis_pairwise_t_test`, `hypothesis_pairwise_wilcox`, `hypothesis_poisson`, `hypothesis_prop_trend_test`, `hypothesis_quade`, `hypothesis_shapiro_wilk`, `hypothesis_t_test`, `hypothesis_wilcoxon`, `multivariate_cancor`, `multivariate_factanal`, `multivariate_mahalanobis`, `power_anova_test`, `power_prop_test`, `power_t_test`, `stats_loglin`, `stats_mauchly_test`, `stats_model_tables`, `stats_se_contrast` |
| Regression | 25 | `evalue`, `marginal_effects`, `regression_bgtest`, `regression_bootstrap_cov`, `regression_clustered`, `regression_cv_glmnet`, `regression_diagnostics`, `regression_driscoll_kraay`, `regression_glmnet`, `regression_gls`, `regression_hac`, `regression_harvtest`, `regression_lasso`, `regression_line`, `regression_loess`, `regression_nls`, `regression_ols`, `regression_quantreg`, `regression_resettest`, `regression_ridge`, `regression_smooth_spline`, `regression_step`, `regression_supsmu`, `regression_waldtest`, `sensemakr` |
| Panel | 9 | `hausman_test`, `panel_fixed_effects`, `panel_gls`, `panel_gmm`, `panel_hdfe`, `panel_pmg`, `panel_pvcm`, `panel_random_effects`, `panel_unit_root` |
| IV | 6 | `bp_bounds`, `gmm_iv`, `iv_2sls`, `iv_first_stage`, `iv_mte`, `iv_sargan_test` |
| DiD | 7 | `bacon_decomp`, `diff_in_diff`, `etwfe`, `gsynth`, `scpi`, `staggered_did`, `synthetic_control` |
| RD | 4 | `rd_bw`, `rd_estimate`, `rd_fuzzy`, `rd_multi` |
| Matching | 1 | `propensity_matching` |
| Treatment | 14 | `collaborative_tmle`, `gformula`, `heterogeneity_test`, `ltmle`, `regression_standardization`, `treatment_cbps`, `treatment_double_ml`, `treatment_doubly_robust`, `treatment_entropy_balance`, `treatment_ipw`, `treatment_sbw`, `treatment_tmle`, `treatment_twang`, `treatment_weightit` |
| Mediation | 2 | `mediation_analysis`, `natural_effects_mediation` |
| Discrete | 10 | `feglm`, `hurdle_model`, `logit`, `mixed_logit`, `mlogit`, `multinom`, `negbin`, `ordered_model`, `probit`, `zeroinfl` |
| TimeSeries | 32 | `causal_impact_analysis`, `linalg_toeplitz`, `timeseries_acf`, `timeseries_acf_to_ar`, `timeseries_ar`, `timeseries_arima_sim`, `timeseries_arma_acf`, `timeseries_arma_to_ma`, `timeseries_box_test`, `timeseries_ccf`, `timeseries_cpgram`, `timeseries_decompose`, `timeseries_diffinv`, `timeseries_embed`, `timeseries_filter`, `timeseries_lag`, `timeseries_pp_test`, `timeseries_runmed`, `timeseries_spectrum`, `timeseries_structts`, `timeseries_window`, `ts_arima_fit`, `ts_arima_forecast`, `ts_changepoint`, `ts_garch_fit`, `ts_granger`, `ts_holt_winters`, `ts_mstl`, `ts_var`, `ts_var_irf`, `ts_varma`, `ts_vecm` |
| Spatial | 10 | `moran_test`, `sar_model`, `sar_probit_model`, `sem_model`, `sem_probit_model`, `spatial_lm_tests_tool`, `spatial_neighbors`, `spatial_panel_gmm`, `spatial_panel_ml`, `sphet_model` |
| Survival | 5 | `aft`, `competing_risks`, `cox_ph`, `kaplan_meier`, `log_rank` |
| MachineLearning | 17 | `ml_bart_causal`, `ml_c50`, `ml_causal_forest`, `ml_cmdscale`, `ml_ctree`, `ml_cubist`, `ml_cutree`, `ml_dbscan`, `ml_hierarchical`, `ml_kmeans`, `ml_mboost`, `ml_pca`, `ml_ppr`, `ml_random_forest`, `ml_shap_values`, `ml_svm`, `ml_tsne` |
| Visualization | 13 | `viz_boxplot`, `viz_coefficient`, `viz_dendrogram`, `viz_event_study`, `viz_heatmap`, `viz_histogram`, `viz_histogram_interactive`, `viz_irf`, `viz_line`, `viz_line_interactive`, `viz_residual_diagnostics`, `viz_scatter`, `viz_scatter_interactive` |
| Database | 7 | `db_duckdb_query`, `db_duckdb_schema`, `db_duckdb_tables`, `db_query_file`, `db_sqlite_query`, `db_sqlite_schema`, `db_sqlite_tables` |
| Utility | 10 | `export_session`, `generate_random_data`, `generate_report`, `get_seed`, `import_session`, `list_tool_categories`, `search_tools`, `server_stats`, `set_seed`, `tool_info` |
| **Total** | **270** | |
<!-- END GENERATED: MCP tool table -->

[docs/mcp/tools-reference.md](docs/mcp/tools-reference.md) describes every tool, and
[docs/guides/MCP_TOOL_EXAMPLES.md](docs/guides/MCP_TOOL_EXAMPLES.md) shows how to call them. Where a tool's default
differs from R's, [Where p2a's defaults differ from R](docs/guides/CLI_REFERENCE.md#where-p2as-defaults-differ-from-r)
lists it.

## Validation

Numerical results are checked against R. *R-pinned* tests load a committed R reference case (`validation/expected/<case>.json`, produced by `validation/r/cases/*.R`) and compare every reported number at the tolerances in `validation/README.md`; they are registered in `validation/methods/*.json`. Other `test_validate_*` tests check properties, known data-generating processes or quoted values. `validation/VALIDATION_STATUS.md` is generated from a full run and lists which methods are R-pinned.

### Running Validation Tests

```bash
validation/run_validation.sh                         # all test_validate tests (CI feature sets) + R drift check
validation/run_validation.sh --rust-only --category econometrics   # stats | regression | econometrics | forecasting | ml | cli | mcp
validation/run_validation.sh --write-status          # also regenerate VALIDATION_STATUS.md from this run
Rscript validation/r/run.R --check                   # R reference values still reproduce (needs R + packages)
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::hdfe::tests::test_validate
python3 validation/tools/validation_status.py lint --strict --tests target/*-tests.txt
```

`run_validation.sh` exits 0 (PASS), 1 (FAIL, including a compile error or zero selected tests), 2 (usage error) or 3 (INCOMPLETE: `cargo` or `Rscript` missing for a requested step).

### Tolerances (`p2a_validation::Tol`, R-pinned tests)

| Quantity | n < 100 | 100 ≤ n ≤ 1000 | n > 1000 |
|---|---|---|---|
| coefficients (`Tol::coef`) | 1e-6 | 1e-8 | 1e-10 |
| standard errors (`Tol::se`) | 1e-5 | 1e-6 | 1e-8 |
| test statistics, R², CI bounds (`Tol::stat`) | 1e-5 | 1e-6 | 1e-8 |

Relative tolerances; iterative methods use `Tol::iterative(rel)` with a documented reason. See `validation/README.md` for the full table.

### Validation Directory Structure

```
validation/
├── README.md                 # what "validated" means, running, adding an R-pinned validation, method index
├── VALIDATION_STATUS.md      # generated by tools/validation_status.py render; never edited by hand
├── reference_implementations.md
├── run_validation.sh         # test_validate tests in the CI feature sets + R drift check
├── r/harness.R               # case(), named(), vec(), dataset CSV and case JSON writers, comparator
├── r/run.R                   # --check [ids] | --write [ids] | --list
├── r/cases/<area>.R          # R reference cases, one file per area
├── expected/<case>.json      # one R reference case per file (schema 1)
├── expected/*.csv            # legacy outputs of scripts/*.R; asserted only where a registry entry supersedes them
├── methods/<area>.json       # registry: method -> cases, tests, quantities, tolerance, deviations
├── datasets/                 # committed reference data; datasets/generated/ is written by R cases
├── tools/                    # validation_status.py (select | collect | lint | render), lint_ignore_names.txt, tests/
├── scripts/*.R               # legacy R scripts; run only with run_validation.sh --legacy-r
├── r_comparison/             # legacy one-off comparison scripts (feglm, survival, synth)
├── reports/                  # run output (gitignored)
└── <category>/<method>.md    # method documents: regression/, diagnostics/, econometrics/ (+ timeseries/),
                              #   forecasting/, timeseries/, stats/, multivariate/, ml/, linalg/
```

Each validation document records: method overview, reference R code, test cases with R's values, the Rust tests and what they check, its status (R-pinned or not) and known differences with their finding ids.

## Performance Benchmarking

Performance is measured with a custom Rust benchmarking framework that captures distribution statistics (min, p25, median, p75, max, mean, std) and memory usage, matching the output format of R's `bench::mark()` package for direct comparison.

**Location**: `crates/p2a-core/benches/` (19 files)

```bash
# Rust benchmarks (outputs JSON)
cargo bench -p p2a-core --bench comprehensive_benchmarks
```

The primary benchmark file is `comprehensive_benchmarks.rs`, which covers all method categories. Specialized benchmark files exist for deeper per-module coverage (regression, econometrics, forecasting, hypothesis tests, ML, spatial, causal, etc.).

Key design choices:
- Custom `bench_utils.rs` runner (not Criterion) to produce distribution statistics compatible with R's `bench::mark()`
- Reproducible data generation via seeded ChaCha8Rng (seed=42)
- Memory tracking via `memory_stats` crate (physical memory before/after/peak)
- JSON output for programmatic consumption and cross-language comparison

The harness writes timestamped JSON (`rust_comprehensive_*.json`) to a local
`performance/results/` directory, falling back to the current directory if it
cannot be created.

## Development

### Running Tests

```bash
cargo test                              # All tests (benchmark-only tests report as ignored)
cargo test -p p2a-core                  # Core library only
cargo test -p p2a-core -- test_validate # Validation tests

# Benchmark-only tests, in a release build (docs/guides/TESTING.md lists them)
cargo test -p p2a-core --release --no-default-features --features ci --test benchmark_ml_rust -- --ignored --nocapture
```

### Linting

The clippy lines CI runs (`--all-features` would turn on `cuda`):

```bash
cargo clippy -p p2a-core --no-default-features --features ci
cargo clippy -p p2a-mcp --no-default-features --features "http,llm,auth,websocket"
cargo clippy -p p2a-cli --no-default-features --features spectral-analysis
cargo clippy -p p2a-chat
cargo clippy -p p2a-text
cargo clippy -p p2a-validation --all-targets
cargo fmt --all -- --check
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

The container binds all interfaces, so it starts only with `P2A_ACCESS_TOKEN` set (every `/api` route then needs the token), or with `P2A_ALLOW_UNAUTHENTICATED=1` for an open demo, which always runs with the hosted limits; see [docs/guides/DEPLOYMENT.md](docs/guides/DEPLOYMENT.md). The backend accepts a plain-http Ollama endpoint only on its own loopback address and refuses private network addresses, so it cannot reach the `with-ollama` profile's container over the compose network.

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
| Database | `surrealdb` | embedded SurrealKV (on disk) or in-memory store |
| Web Frontend | `dioxus` | 0.7 |

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT) or https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or https://www.apache.org/licenses/LICENSE-2.0)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

## Documentation

- `validation/README.md` - Validation framework and method index
- `validation/VALIDATION_STATUS.md` - Current validation coverage report
- `docs/guides/DEPLOYMENT.md` - Hosting the backend: deployment modes, limits, configuration, release
- `API_ENDPOINTS.md` - HTTP API reference
- `docs/guides/TESTING.md` - Test runtime expectations, integration tests, what CI runs
- `docs/guides/DATA_SECURITY.md` - Data write locations, privacy
- `docs/security/PROMPT_INJECTION.md` - MCP security considerations

## Contributing

Contributions are welcome! See:
- `CLAUDE.md` - Development guidance for Claude Code
- `validation/README.md` - How to add validation tests
