# prompt2analytics Development Report

**Date:** January 22, 2026
**Status:** Phase 7 (Dioxus Cross-Platform App) ✅ COMPLETE

---

## Executive Summary

Phases 1, 2, 2b, 3a, 3b, 4, and part of Phase 5 of the prompt2analytics development plan are now complete. The analytics engine includes:
- Panel data estimators (Fixed Effects, Random Effects)
- Hausman specification test
- Instrumental variables (2SLS) with first-stage diagnostics
- Difference-in-differences
- Regression Discontinuity Design (Sharp and Fuzzy RD with robust bias-corrected inference)
- Regression diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
- Clustered standard errors (one-way and two-way)
- Discrete choice models (Logit, Probit)
- Time series: VAR, VARMA, VECM models with Impulse Response Functions
- Univariate forecasting: ARIMA and MSTL decomposition
- File formats: CSV, Parquet, Excel, Stata, SAS
- ML algorithms: K-means, DBSCAN, Hierarchical clustering, PCA, t-SNE, Random Forest, Linear SVM
- Database connectivity: SQLite and DuckDB (query, list tables, schema)
- **Visualization: Histograms, scatter plots, line charts, box plots, correlation heatmaps, event study plots, coefficient plots, IRF plots, residual diagnostics**
- **Desktop Application: Tauri 2.0 + SvelteKit with MCP subprocess integration**
- **LLM Integration: Multi-provider support (Ollama, Anthropic, OpenAI) with streaming**
- **Conversation History: SQLite persistence with search, rename, and export**
- **Dataset Context: Automatic injection of loaded dataset info into LLM prompts**
- **Dioxus Cross-Platform App: Pure Rust web/desktop/mobile frontend with tool call transparency**

The codebase uses **pure Rust implementations** for all econometrics (OLS, panel data, IV, DiD, discrete choice, time series), ML algorithms, native database drivers for SQLite/DuckDB, `plotters` for in-memory chart generation with base64-encoded PNG output, Tauri 2.0 for the desktop application, and multi-provider LLM integration with streaming responses and tool execution loop.

---

## Phase 1: Foundation (MVP Core) — ✅ COMPLETE

### Planned vs Implemented

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Cargo workspace scaffold | ✅ Complete | `p2a-core` and `p2a-mcp` crates |
| MCP server with rmcp SDK | ✅ Complete | Using rmcp 0.8 with tool macros |
| Data loading (CSV) | ✅ Complete | Via Polars |
| Data loading (Parquet) | ✅ Complete | Via Polars |
| `data_load` tool | ✅ Complete | Implemented as `load_dataset` |
| `data_describe` tool | ✅ Complete | Implemented as `describe_dataset` |
| `data_head` tool | ✅ Complete | Implemented as `head_dataset` |
| `regression_ols` tool | ✅ Complete | Full output: coefficients, SE, t-values, p-values, R², F-stat |
| `correlation_matrix` tool | ✅ Complete | Implemented as `compute_correlation` |
| `list_datasets` tool | ✅ Complete | Additional tool for session management |
| Integration tests | ⚠️ Partial | Server runs, basic unit test exists |

### MVP Success Criteria Checklist

From the original plan:

- [x] MCP server starts and registers tools correctly
- [x] `data_load` handles CSV files (100MB+ not tested but architecture supports it)
- [x] `data_describe` returns accurate summary stats
- [x] `regression_ols` produces correct coefficients, R², p-values
- [x] Results are formatted for LLM consumption

---

## Phase 2: Econometrics & Time Series — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| Fixed Effects (FE) estimation | ✅ Complete | Pure Rust (within-group demeaning) |
| Random Effects (RE) estimation | ✅ Complete | Pure Rust (GLS/quasi-demeaning) |
| Hausman test | ✅ Complete | Pure Rust (FE vs RE specification) |
| Two-way clustering | ✅ Complete | Pure Rust (Cameron-Gelbach-Miller) |
| One-way clustering | ✅ Complete | Pure Rust |
| 2SLS (Instrumental Variables) | ✅ Complete | Pure Rust |
| First-stage diagnostics | ✅ Complete | Pure Rust (F-stat, partial R²) |
| Difference-in-Differences | ✅ Complete | Pure Rust |
| Regression Discontinuity (Sharp) | ✅ Complete | Pure Rust (CCT 2014 robust inference) |
| Regression Discontinuity (Fuzzy) | ✅ Complete | Pure Rust (LATE via Wald estimator) |
| RD Bandwidth Selection | ✅ Complete | Pure Rust (MSE/CER optimal, IK 2012) |
| Regression diagnostics | ✅ Complete | Pure Rust (JB, BP, DW, VIF) |
| Logit (logistic regression) | ✅ Complete | Pure Rust (Newton-Raphson MLE) |
| Probit regression | ✅ Complete | Pure Rust (Newton-Raphson MLE) |
| FEGLM (GLM + HDFE) | ✅ Complete | Pure Rust (IRLS + weighted MAP, Stammann 2018) |
| Event study plots | ❌ Deferred | Phase 5 |
| ARIMA modeling | ✅ Complete | arima crate |
| MSTL decomposition | ✅ Complete | augurs-mstl |
| Changepoint detection | ❌ Deferred | Phase 5 |
| VAR model | ✅ Complete | Pure Rust (OLS per equation) |
| VARMA model | ✅ Complete | Pure Rust (Hannan-Rissanen) |
| VECM (Johansen cointegration) | ✅ Complete | Pure Rust (Johansen ML) |
| Impulse Response Functions | ✅ Complete | Pure Rust (Cholesky orthogonalization) |
| Robust Standard Errors (HC0-HC3) | ✅ Complete | Pure Rust |
| Excel file support | ✅ Complete | calamine |
| Stata (.dta) support | ✅ Complete | Pure Rust (v117-119) |
| SAS (.sas7bdat) support | ✅ Complete | Pure Rust |
| SQLite connections | ✅ Complete | rusqlite 0.33 (Phase 2b) |
| DuckDB connections | ✅ Complete | duckdb 1.2 (Phase 2b) |

---

## Phase 3a: Visualization — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| Histograms | ✅ Complete | plotters (in-memory PNG) |
| Scatter plots | ✅ Complete | plotters (with correlation) |
| Line charts | ✅ Complete | plotters (multi-series) |
| Box plots | ✅ Complete | plotters (with quartile stats) |
| Correlation heatmaps | ✅ Complete | plotters (blue-white-red colormap) |
| Event study plots | ✅ Complete | plotters (CI bands, treatment line) |
| Coefficient plots | ✅ Complete | plotters (horizontal/vertical error bars) |
| IRF plots | ✅ Complete | plotters (optional CI bands) |
| Residual diagnostics | ✅ Complete | 4 plots: Residuals vs Fitted, Q-Q, Scale-Location, Leverage |

### Visualization Implementation Details

**Chart Generation (plotters 0.3):**
- In-memory bitmap rendering (no file system writes)
- Base64-encoded PNG output for MCP tool responses
- Configurable dimensions (default: 800x600)
- Support for custom titles, axis labels
- Automatic axis range calculation

**Histogram:**
- Configurable bin count (default: Sturges' rule)
- Returns bin edges, frequencies, and image

**Scatter Plot:**
- Auto-calculates Pearson correlation
- Displays point count and correlation coefficient

**Line Chart:**
- Multi-series support (multiple Y columns)
- Automatic color cycling for series
- Legend display

**Box Plot:**
- Shows min, Q1, median, Q3, max for each group
- Returns full quartile statistics
- Handles multiple columns

**Correlation Heatmap:**
- Diverging blue-white-red colormap
- Cell value annotations (when cells are large enough)
- Colorbar legend
- Optional column filtering

### Econometrics Implementation Details

All econometrics are implemented in **pure Rust** using `ndarray` for matrix operations, `faer` for linear algebra (Cholesky, matrix inverse), and `statrs` for statistical distributions.

**OLS Regression:**
- Column-based API (y_col, x_cols) instead of formula parsing
- Robust standard errors: HC0, HC1, HC2, HC3 (heteroskedasticity-consistent)
- Clustered standard errors: one-way and two-way (Cameron-Gelbach-Miller)
- Full output: coefficients, SE, t-values, p-values, R², adjusted R², F-statistic

**Panel Data Estimators:**
- Fixed Effects (within estimator) with entity demeaning
- Random Effects (GLS with quasi-demeaning, theta estimation)
- Hausman specification test (choose between FE/RE)
- Automatic entity ID mapping from string/integer columns

**Instrumental Variables:**
- Two-Stage Least Squares (2SLS) with separate exogenous/endogenous regressors
- First-stage diagnostics: F-statistic, partial R², instrument strength
- Robust standard errors option

**Causal Inference:**
- Difference-in-Differences (canonical 2x2)
- Treatment effect (ATT) with standard errors
- Group means for parallel trends assessment
- Optional control variables

**Regression Diagnostics:**
- Jarque-Bera test (normality of residuals)
- Breusch-Pagan test (heteroskedasticity)
- Durbin-Watson test (autocorrelation)
- Variance Inflation Factor (multicollinearity)
- Condition number (multicollinearity)

**Discrete Choice Models:**
- Logit (logistic regression) via Newton-Raphson MLE
- Probit regression via Newton-Raphson MLE
- McFadden's Pseudo R-squared
- Marginal effects at means

**FEGLM (GLM with High-Dimensional Fixed Effects):**
- Generalized Linear Models with multi-way FE absorption (Stammann 2018)
- Supported families: Logit, Probit, Poisson, Gaussian
- Iteratively Reweighted Least Squares (IRLS) outer loop
- Weighted Method of Alternating Projections (MAP) for FE demeaning
- Equivalent to R's alpaca::feglm() / fixest::feglm()
- Applications: Gravity models with exporter/importer FE, conditional logit with individual FE

**Multivariate Time Series:**
- VAR (Vector Autoregression) via OLS equation-by-equation
- VARMA (Vector ARMA) via Hannan-Rissanen two-step estimation
- VECM (Vector Error Correction Model) via Johansen ML
- Impulse Response Functions (IRF) with Cholesky orthogonalization
- Cointegration vectors (beta) and adjustment speeds (alpha)

---

## Phase 2b: ML Toolkit & Database — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| K-means clustering | ✅ Complete | Pure Rust (k-means++ init) |
| DBSCAN | ✅ Complete | Pure Rust |
| Hierarchical clustering | ✅ Complete | Pure Rust (Ward, single, complete, average linkage) |
| Logistic regression | ✅ Complete | Pure Rust (Newton-Raphson MLE) |
| Random Forest | ✅ Complete | Pure Rust (CART algorithm, feature importance) |
| SVM | ✅ Complete | Pure Rust (Linear SVM with SMO) |
| PCA | ✅ Complete | Pure Rust |
| t-SNE | ✅ Complete | Pure Rust (Barnes-Hut approximation) |
| SQLite connectivity | ✅ Complete | rusqlite 0.33 |
| DuckDB connectivity | ✅ Complete | duckdb 1.2 |

### ML Implementation Details

**K-means Clustering:**
- K-means++ initialization for better convergence
- Configurable number of clusters, max iterations, and random initializations
- Returns cluster assignments, centroids, and inertia (within-cluster sum of squares)
- Pure Rust implementation (no linfa dependency to avoid ndarray conflicts)

**DBSCAN Clustering:**
- Density-based spatial clustering
- Identifies outliers as noise points (cluster = -1)
- Does not require specifying number of clusters
- Configurable epsilon (neighborhood radius) and min_samples

**PCA (Principal Component Analysis):**
- Dimensionality reduction via eigendecomposition
- Returns principal components, explained variance ratios, and loadings
- Supports specifying number of components to retain
- Pure Rust implementation using power iteration

### Database Implementation Details

**SQLite Support (rusqlite 0.33):**
- `query_sqlite` — Execute SQL query, return results as DataFrame
- `list_sqlite_tables` — List all tables in database
- `sqlite_table_schema` — Get column names and types for a table
- Automatic type inference (INTEGER, REAL, TEXT, BLOB)

**DuckDB Support (duckdb 1.2):**
- `query_duckdb` — Execute SQL query, return results as DataFrame
- `list_duckdb_tables` — List all tables in database
- `duckdb_table_schema` — Get column names and types for a table
- `query_file_with_duckdb` — Query CSV/Parquet files directly without loading
- Support for in-memory databases (`:memory:`)

---

## Phase 3b: Desktop Application — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| Tauri 2.0 application shell | ✅ Complete | Tauri 2.9, tauri-plugin-dialog, tauri-plugin-shell |
| MCP subprocess integration | ✅ Complete | JSON-RPC over stdio to p2a-mcp |
| Chat interface (SvelteKit) | ✅ Complete | Svelte 5 with runes, command parsing |
| Data viewer | ✅ Complete | Dataset preview, file picker |
| Results panel | ✅ Complete | Collapsible results, base64 image rendering |
| Dataset management | ✅ Complete | Load, list, describe datasets |
| Test scenario | ✅ Complete | docs/testing/ with sample data |

### Desktop Application Architecture

**Backend (Rust/Tauri):**
- Spawns `p2a-mcp` as subprocess on startup
- JSON-RPC 2.0 protocol over stdin/stdout
- Async request/response with oneshot channels
- Graceful shutdown on window close

**Frontend (SvelteKit):**
- Three-panel layout: Chat, Data Viewer, Results
- Svelte 5 runes for reactive state (`$state`, `$derived`)
- Static adapter for Tauri (SSR disabled)
- Native file dialogs via tauri-plugin-dialog

**Tauri Commands:**
- `invoke_tool` — Call any MCP tool by name
- `list_tools` — Get available tools
- `list_datasets` — Get loaded datasets
- `load_dataset` — Load file via native dialog
- `get_dataset_preview` — Preview rows
- `describe_dataset` — Summary statistics
- `pick_file` / `pick_files` / `pick_directory` — File dialogs

**System Requirements (Linux):**
```bash
sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
```

**Build & Run:**
```bash
cargo build --release -p p2a-mcp
cd crates/p2a-dioxus && dx serve --platform desktop
```

---

## Phase 4: LLM Integration — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| Ollama integration | ✅ Complete | OllamaProvider with streaming |
| Anthropic API support | ✅ Complete | AnthropicProvider with tool use |
| OpenAI API support | ✅ Complete | OpenAIProvider with streaming |
| Conversation history | ✅ Complete | SQLite persistence (HistoryStore) |
| Streaming responses | ✅ Complete | Tauri events for real-time UI updates |
| Tool execution loop | ✅ Complete | MCP tool integration with LLM |
| Settings UI | ✅ Complete | Provider selection, API keys, model refresh |
| Conversation management | ✅ Complete | Rename, export (JSON/Markdown), search |
| Dataset context | ✅ Complete | Auto-inject dataset info in system prompt |
| Markdown rendering | ✅ Complete | marked + highlight.js for code blocks |
| UI improvements | ✅ Complete | Loading spinners, auto-scroll, error handling |
| Export (PDF/HTML reports) | ❌ Deferred | Phase 5 |

### LLM Implementation Details

**Multi-Provider Architecture:**
- Abstract `LlmProvider` trait with streaming support
- `OllamaProvider`: Local Ollama server, model listing, streaming
- `AnthropicProvider`: Claude API with tool use support
- `OpenAIProvider`: GPT models with function calling
- Configurable base URLs, API keys, and model selection

**Conversation History (SQLite):**
- `HistoryStore` for persistent conversation storage
- `Conversation` struct with metadata (title, created_at, updated_at)
- `StoredMessage` for role, content, and tool calls
- Full CRUD operations: create, list, load, delete, rename
- Export to JSON and Markdown formats

**Streaming Architecture:**
- `StreamChunk` enum for text, tool calls, and completion events
- Tauri event emission for real-time frontend updates
- Token-by-token display in chat interface
- Error handling with graceful degradation

**Tool Execution Loop:**
- LLM can invoke any of the 100 MCP analytics tools
- Automatic tool call parsing and execution
- Results fed back to LLM for interpretation
- Support for multi-turn tool conversations

**Dataset Context Integration:**
- `get_dataset_context()` retrieves loaded dataset info
- `get_system_prompt_with_context()` injects context into prompt
- LLM automatically aware of available datasets and columns
- Enables natural language queries about loaded data

**Settings Management:**
- Provider type selection (Ollama, Anthropic, OpenAI)
- API key storage (local, not persisted to disk)
- Model refresh: fetches available models from provider
- Test connection: validates API key and connectivity
- Settings validation before saving

**Frontend Enhancements:**
- `marked` library for Markdown → HTML conversion
- `highlight.js` for syntax highlighting in code blocks
- Loading spinner during LLM response generation
- Auto-scroll to bottom on new messages
- Conversation sidebar with search functionality
- Rename and export conversation actions

---

## Phase 5: Advanced Features — ✅ COMPLETE

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Event study plots | ✅ Complete | Dynamic DiD visualization with CI bands |
| Coefficient plots | ✅ Complete | With confidence intervals (horizontal/vertical) |
| IRF plots | ✅ Complete | VAR impulse response with optional CI |
| Residual diagnostics | ✅ Complete | 4 plots: Residuals vs Fitted, Q-Q, Scale-Location, Leverage |
| Hierarchical clustering | ✅ Complete | Ward/single/complete/average linkage |
| Random Forest | ✅ Complete | Pure Rust CART with feature importance |
| SVM | ✅ Complete | Linear SVM with SMO algorithm |
| t-SNE | ✅ Complete | Pure Rust with early exaggeration |
| Changepoint detection | ✅ Complete | PELT and Binary Segmentation algorithms |
| HTML reports | ✅ Complete | Self-contained HTML report generation |
| Dendrogram visualization | ✅ Complete | Tree visualization for hierarchical clustering |
| Batch processing | ✅ Complete | Run same analysis across multiple datasets |
| Dataset comparison | ✅ Complete | Compare columns across datasets (summary, distribution, corr) |
| Session export/import | ✅ Complete | Save/restore analysis sessions to JSON |
| Seed management | ✅ Complete | Global seed for ML reproducibility |
| **Deferred to future releases:** | | |
| Plugin system | ❌ | Custom analytics extensions |
| Community tool registry | ❌ | Shared tool definitions |
| PDF reports | ❌ | PDF export |
| Documentation/tutorials | ❌ | User guides, examples |

---

## Progress Summary

| Phase | Status | Completion |
|-------|--------|------------|
| Phase 1: Foundation (MVP Core) | ✅ Complete | 100% |
| Phase 2: Econometrics & Time Series | ✅ Complete | 100% |
| Phase 2b: ML Toolkit & Database | ✅ Complete | 100% |
| Phase 3a: Visualization | ✅ Complete | 100% |
| Phase 3b: Desktop Application | ✅ Complete | 100% |
| Phase 4: LLM Integration | ✅ Complete | 95% |
| Phase 5: Advanced Features | ✅ Complete | 100% |

**Overall Progress: ~99%** (All planned phases complete; optional features deferred to future releases)

---

## Technical Implementation Details

**Dependencies (current versions):**

*Core Analytics (p2a-core, p2a-mcp):*
- `polars` 0.46 — DataFrame operations
- `rmcp` 0.8 — MCP SDK with tool macros
- `ndarray` 0.16 — Numerical arrays for matrix operations
- `faer` 0.22 — High-performance linear algebra (Cholesky, matrix inverse)
- `statrs` 0.18 — Statistical distributions (t, F, chi-squared, normal)
- `calamine` 0.32 — Excel file reading (xlsx, xls, xlsb, ods)
- `arima` 0.3 — ARIMA model fitting and forecasting
- `augurs-mstl` 0.10 — MSTL seasonal-trend decomposition
- `augurs-core` 0.10 — Augurs common traits
- `rand` 0.8 — Random number generation (for ML/forecasting)
- `rusqlite` 0.33 — SQLite database connectivity (bundled)
- `duckdb` 1.2 — DuckDB database connectivity (bundled)
- `plotters` 0.3 — In-memory chart generation
- `image` 0.24 — PNG encoding
- `base64` 0.22 — Base64 encoding for image output

*Dioxus Cross-Platform App (p2a-dioxus):*
- `dioxus` 0.7 — Cross-platform UI framework (WASM + native)
- `reqwest` — HTTP client for backend API
- `gloo-storage` — Web localStorage for settings
- `pulldown-cmark` — Markdown rendering
- `chrono` + `uuid` — Timestamps and identifiers

*Frontend (SvelteKit):*
- `svelte` 5.x — UI framework with runes
- `@sveltejs/kit` 2.x — SvelteKit framework
- `@sveltejs/adapter-static` 3.x — Static site generation
- `@tauri-apps/api` 2.x — Tauri JavaScript API
- `@tauri-apps/plugin-dialog` 2.x — Dialog plugin bindings
- `vite` 5.x — Build tool
- `typescript` 5.x — Type checking
- `marked` 15.x — Markdown to HTML conversion
- `highlight.js` 11.x — Syntax highlighting for code blocks

**System Requirements:**
- OpenBLAS: `sudo apt-get install libopenblas-dev`

**Major Change:** All econometrics implemented in **pure Rust** (no external econometrics library). This provides:
- Full control over implementation details
- Column-based API (y_col, x_cols) instead of R-style formula parsing
- Robust standard errors (HC0, HC1, HC2, HC3)
- Clustered standard errors (one-way and two-way via Cameron-Gelbach-Miller)
- Panel data estimators (Fixed Effects, Random Effects, Hausman test)
- Instrumental Variables (2SLS) with first-stage diagnostics
- Difference-in-Differences estimation
- Discrete choice models (Logit/Probit via Newton-Raphson MLE)
- Multivariate time series (VAR, VARMA, VECM with IRF)
- Comprehensive regression diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)

**Architecture:** The econometrics implementation uses a modular design:
- `linalg/` — Matrix operations (X'X, X'y, Cholesky, safe inverse) and design matrices
- `traits/` — `LinearEstimator` trait for common regression output interface
- `errors.rs` — Unified error types (`EconError`, `EconResult`)

**MCP Tools Exposed (62 total):**
```
┌─────────────────────────┬──────────────────────────────────────────────────────────────┐
│ Tool                    │ Description                                                  │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ list_datasets           │ Show all loaded datasets                                     │
│ load_dataset            │ Load CSV/Parquet/Excel/Stata/SAS file into session           │
│ describe_dataset        │ Summary statistics (count, mean, std, quartiles)             │
│ head_dataset            │ Preview first N rows                                         │
│ data_quality_profile    │ Comprehensive quality profile for LLM-assisted cleaning      │
│ preview_cleaning        │ Preview cleaning operation before applying (sample changes)  │
│ verify_cleaning         │ Compare before/after datasets to verify cleaning results     │
│ compute_correlation     │ Pearson correlation matrix for numeric columns               │
│ regression_ols          │ OLS regression with robust SEs (HC1)                         │
│ regression_diagnostics  │ Model validation (JB, BP, DW, VIF, condition number)         │
│ regression_clustered    │ OLS with one-way or two-way clustered SEs                    │
│ panel_fixed_effects     │ Fixed Effects panel regression                               │
│ panel_random_effects    │ Random Effects (GLS) panel regression                        │
│ hausman_test            │ Specification test: FE vs RE                                 │
│ iv_2sls                 │ Instrumental Variables / 2SLS regression                     │
│ iv_first_stage          │ First-stage diagnostics (F-stat, instrument strength)        │
│ diff_in_diff            │ Difference-in-Differences causal estimation                  │
│ rd_estimate             │ Sharp RD with robust bias-corrected inference                │
│ rd_bw                   │ RD bandwidth selection (MSE/CER optimal)                     │
│ rd_fuzzy                │ Fuzzy RD / LATE estimation with first-stage diagnostics     │
│ logit                   │ Logistic regression (binary outcomes)                        │
│ probit                  │ Probit regression (binary outcomes)                          │
│ feglm                   │ GLM with high-dimensional fixed effects (IRLS + weighted MAP)│
│ ts_var                  │ Vector Autoregression (VAR) model                            │
│ ts_varma                │ VARMA(p,q) via Hannan-Rissanen                               │
│ ts_vecm                 │ Vector Error Correction Model (Johansen ML)                  │
│ ts_var_irf              │ VAR Impulse Response Functions                               │
│ ts_arima_fit            │ ARIMA(p,d,q) model fitting                                   │
│ ts_arima_forecast       │ ARIMA h-step ahead forecasting                               │
│ ts_mstl                 │ MSTL seasonal-trend decomposition                            │
│ ts_changepoint          │ Changepoint detection (PELT/Binary Segmentation)             │
│ ml_kmeans               │ K-means clustering with k-means++ initialization             │
│ ml_dbscan               │ DBSCAN density-based clustering                              │
│ ml_hierarchical         │ Hierarchical/agglomerative clustering (Ward, single, etc.)   │
│ ml_pca                  │ Principal Component Analysis                                 │
│ ml_tsne                 │ t-SNE dimensionality reduction for visualization             │
│ ml_random_forest        │ Random Forest regression with feature importance             │
│ ml_svm                  │ Linear SVM classification (SMO algorithm)                    │
│ db_sqlite_query         │ Execute SQL query on SQLite database                         │
│ db_sqlite_tables        │ List tables in SQLite database                               │
│ db_sqlite_schema        │ Get schema for SQLite table                                  │
│ db_duckdb_query         │ Execute SQL query on DuckDB database                         │
│ db_duckdb_tables        │ List tables in DuckDB database                               │
│ db_duckdb_schema        │ Get schema for DuckDB table                                  │
│ viz_histogram           │ Histogram for numeric column (base64 PNG)                    │
│ viz_scatter             │ Scatter plot with correlation (base64 PNG)                   │
│ viz_line                │ Line chart for time series (multi-series, base64 PNG)        │
│ viz_boxplot             │ Box plot with quartile statistics (base64 PNG)               │
│ viz_heatmap             │ Correlation heatmap (base64 PNG)                             │
│ viz_event_study         │ Event study plot with confidence bands (base64 PNG)          │
│ viz_coefficient         │ Coefficient plot with error bars (base64 PNG)                │
│ viz_irf                 │ IRF plot for VAR models with optional CI (base64 PNG)        │
│ viz_residual_diagnostics│ 4 diagnostic plots: Residuals vs Fitted, Q-Q, Scale-Loc, Leverage │
│ viz_dendrogram          │ Dendrogram (tree diagram) for hierarchical clustering        │
│ generate_report         │ Generate self-contained HTML report from analysis results    │
│ batch_process           │ Run same analysis across multiple datasets at once           │
│ compare_datasets        │ Compare columns across datasets (summary, distribution, corr) │
│ export_session          │ Export current session (datasets) to JSON file               │
│ import_session          │ Import previously exported session from JSON file            │
│ set_seed                │ Set global random seed for ML reproducibility                │
│ get_seed                │ Get current global seed and list supported ML tools          │
└─────────────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## Files Created

```
prompt2analytics/
├── Cargo.toml                          # Workspace root
├── .gitignore                          # Git ignore rules
├── .mcp.json                           # MCP server config for Claude Code
├── CLAUDE.md                           # Development guidance
├── DEVELOPMENT_REPORT.md               # This file
├── tests/data/sample.csv               # Test dataset
├── tests/data/test.xlsx                # Excel test file
├── docs/
│   └── testing/
│       ├── sample_sales.csv            # Test dataset for desktop app
│       └── DESKTOP_TEST_SCENARIO.md    # Alpha tester guide
├── validation/                          # Validation against reference implementations
│   ├── README.md
│   ├── reference_implementations.md
│   ├── regression/                      # OLS, robust SEs, clustered SEs
│   ├── econometrics/                    # Panel, IV, DiD, discrete choice, timeseries
│   ├── forecasting/                     # ARIMA, MSTL, changepoint
│   ├── ml/                              # K-means, DBSCAN, PCA, t-SNE, RF, SVM
│   ├── diagnostics/                     # Regression diagnostics
│   └── datasets/                        # Grunfeld, Longley, iris datasets
├── performance/                         # Performance benchmarking framework
│   ├── README.md
│   ├── methodology.md
│   ├── benchmarks/                      # Criterion benchmark code
│   ├── results/                         # Raw benchmark data
│   ├── comparisons/r_comparison/        # R microbenchmark scripts
│   └── reports/                         # Performance summary reports
└── crates/
    ├── p2a-core/
    │   ├── Cargo.toml
    │   ├── tests/data/test.xlsx        # Excel test data
    │   └── src/
    │       ├── lib.rs
    │       ├── errors.rs               # EconError, EconResult types
    │       ├── linalg/                 # Linear algebra utilities
    │       │   ├── mod.rs
    │       │   ├── matrix_ops.rs       # X'X, X'y, Cholesky, safe inverse (via faer)
    │       │   └── design.rs           # DesignMatrix, demeaning, quasi-demeaning
    │       ├── traits/                 # Common traits
    │       │   ├── mod.rs
    │       │   └── estimator.rs        # LinearEstimator trait, p-value helpers
    │       ├── data/
    │       │   ├── mod.rs
    │       │   ├── dataset.rs
    │       │   ├── loader.rs           # CSV, Parquet, Excel, Stata, SAS
    │       │   ├── stata.rs            # Pure Rust Stata DTA reader (v117-119)
    │       │   ├── sas.rs              # Pure Rust SAS7BDAT reader
    │       │   └── database.rs         # SQLite + DuckDB connectivity
    │       ├── stats/
    │       │   ├── mod.rs
    │       │   ├── descriptive.rs
    │       │   └── correlation.rs
    │       ├── regression/
    │       │   ├── mod.rs
    │       │   ├── ols.rs              # OLS + robust SEs (HC0-HC3) + clustered SEs
    │       │   └── diagnostics.rs      # JB, BP, DW, VIF, condition number
    │       ├── econometrics/
    │       │   ├── mod.rs
    │       │   ├── panel.rs            # FE/RE + Hausman test
    │       │   ├── iv.rs               # 2SLS/IV + first-stage diagnostics
    │       │   ├── did.rs              # Difference-in-Differences
    │       │   ├── discrete.rs         # Logit/Probit (Newton-Raphson MLE)
    │       │   ├── timeseries.rs       # VAR/VARMA/VECM/IRF
    │       │   └── rd.rs               # Regression Discontinuity (Sharp/Fuzzy RD)
    │       ├── forecasting/
    │       │   ├── mod.rs
    │       │   ├── arima_model.rs      # ARIMA fitting and forecasting
    │       │   ├── mstl.rs             # MSTL decomposition
    │       │   └── changepoint.rs      # Changepoint detection (PELT, Binary Seg)
    │       ├── ml/
    │       │   ├── mod.rs
    │       │   ├── clustering.rs       # K-means, DBSCAN, Hierarchical (pure Rust)
    │       │   ├── reduction.rs        # PCA, t-SNE (pure Rust)
    │       │   ├── trees.rs            # Random Forest with CART (pure Rust)
    │       │   └── svm.rs              # Linear SVM with SMO (pure Rust)
    │       ├── visualization/
    │       │   ├── mod.rs
    │       │   ├── charts.rs           # Histogram, scatter, line, box, event study, coefficient, IRF plots
    │       │   └── heatmap.rs          # Correlation heatmap
    │       └── reports/
    │           ├── mod.rs
    │           └── html.rs             # HTML report generation
    ├── p2a-mcp/
    │   ├── Cargo.toml
    │   └── src/
    │       ├── main.rs
    │       ├── server.rs               # 101 MCP tools
    │       └── tools/
    │           └── mod.rs              # Placeholder
    └── p2a-dioxus/
        ├── Cargo.toml                  # Dioxus dependencies
        ├── Dioxus.toml                 # Dioxus configuration
        ├── src/
        │   ├── main.rs                 # Entry point
        │   ├── app.rs                  # Root App component
        │   ├── api/
        │   │   ├── mod.rs
        │   │   ├── client.rs           # HTTP client for backend
        │   │   ├── sse.rs              # SSE streaming for chat
        │   │   └── types.rs            # Request/response types
        │   ├── state/
        │   │   ├── mod.rs
        │   │   ├── chat.rs             # Message state
        │   │   ├── conversation.rs     # Conversation management
        │   │   ├── session.rs          # Backend session
        │   │   └── settings.rs         # User preferences
        │   ├── components/
        │   │   ├── mod.rs
        │   │   ├── chat_panel.rs       # Main chat interface
        │   │   ├── conversation_sidebar.rs # Conversation list
        │   │   ├── dataset_sidebar.rs  # Dataset list
        │   │   ├── chat_input.rs       # Message input
        │   │   ├── message.rs          # Message display
        │   │   ├── tool_call.rs        # Tool call cards
        │   │   └── settings_modal.rs   # Provider config
        │   └── utils/
        │       └── markdown.rs         # Markdown to RSX
        └── assets/
            └── styles.css              # Tailwind-like CSS
```

---

## Technical Deviations from Plan

1. **Polars version:** Using 0.46 instead of planned 0.50+. The API changed significantly — notably `is_numeric()` method was removed, requiring custom dtype checking.

2. **rmcp version:** Using 0.8 instead of planned 0.12. The SDK uses different versioning than anticipated. Key syntax: `Parameters<T>` wrapper for tool parameters.

3. **Pure Rust econometrics:** Originally planned to use external libraries (linfa, greeners). Implemented all econometrics in pure Rust to avoid dependency conflicts and gain full control over the API design. Uses `faer` for linear algebra operations.

4. **JSON support deferred:** Polars 0.46 removed `JsonReader` — JSON loading not currently supported (CSV and Parquet work).

5. **Column-based API:** Instead of R-style formula parsing (e.g., "y ~ x1 + x2"), uses explicit column names (y_col, x_cols). This is simpler and more explicit for MCP tool integration.

---

## Recommended Next Steps

1. **Future Enhancements (optional):**
   - Plugin system for custom analytics
   - Community tool registry
   - PDF report generation
   - Documentation and tutorials

2. **Desktop App Enhancements:**
   - Theme customization (dark mode)
   - Visual query builder for databases
   - Dataset column selection/filtering
   - Command history and autocomplete
   - Settings persistence to disk

3. **Testing:**
   - ✅ Expanded test coverage for econometrics (86 → 95 tests)
   - ✅ Added integration tests with known econometric results
   - ✅ Stata/SAS file format readers tested with synthetic files
   - Test database tools with larger databases
   - Desktop app end-to-end testing
   - LLM integration tests with mocked providers

4. **Documentation:**
   - ✅ Added usage examples for each MCP tool (`docs/guides/MCP_TOOL_EXAMPLES.md`)
   - ✅ Documented econometric model assumptions and interpretation (`docs/guides/ECONOMETRICS_GUIDE.md`)
   - Document database query patterns
   - ✅ Created desktop app user guide (`docs/guides/DESKTOP_USER_GUIDE.md`)
   - LLM provider configuration guide

---

## Agentic Engineering Setup — ✅ COMPLETE

The repository is now configured for "agentic engineering" — systematic implementation of new econometric methods using Claude Code's advanced features.

### `/implement_metrics` Slash Command

A custom slash command for implementing new econometric methods:

```bash
/implement_metrics https://en.wikipedia.org/wiki/Generalized_least_squares
/implement_metrics ./_resources/methods/fgls.md
```

**Workflow:**
1. **Research** — Fetch URL/file, extract formulas, find reference implementations
2. **Planning** — Design API following p2a-core patterns, plan tests
3. **Implementation** — Implement in Rust following existing patterns
4. **Testing** — Write tests, validate against reference implementations
5. **Documentation** — Update guides and reports

### Skills

Auto-discovered guidance files in `.claude/skills/`:

| Skill | Purpose |
|-------|---------|
| `econometrics-research` | Finding reference implementations, extracting mathematical formulations |
| `rust-econometrics-patterns` | p2a-core API patterns, LinearEstimator trait, error handling |

### Subagent

`.claude/agents/econometrics-implementer.md` — Expert Rust econometrics implementer for complex implementations.

### Settings

`.claude/settings.json` includes:
- Permissions for cargo, web fetch (arxiv, CRAN, Stata docs), DeepWiki
- Post-tool hook to run `cargo check` after editing `.rs` files in p2a-core

### Directory Structure

```
.claude/
├── commands/
│   └── implement_metrics.md      # Main entry point
├── skills/
│   ├── econometrics-research/
│   │   └── SKILL.md              # Research guidance
│   ├── rust-econometrics-patterns/
│   │   └── SKILL.md              # Implementation patterns
│   └── validation-benchmarking/
│       └── SKILL.md              # Validation & benchmarking workflow
├── agents/
│   └── econometrics-implementer.md  # Specialized subagent
├── settings.json                 # Shared permissions and hooks
└── settings.local.json           # Local/user-specific settings
```

---

## Validation & Performance Framework — ✅ COMPLETE

A comprehensive validation and performance framework has been established to prepare the codebase for publication (e.g., Journal of Statistical Software).

### Validation Framework (`validation/`)

All 29+ methods are validated against reference implementations:

| Category | Methods | R/Python References |
|----------|---------|---------------------|
| Regression | OLS, Robust SEs (HC0-HC3), Clustered SEs | `lm()`, `sandwich::vcovHC`, `sandwich::vcovCL` |
| Panel Data | FE, RE, Hausman, HDFE | `plm::plm()`, `lfe::felm()` |
| IV/Causal | 2SLS, DiD | `AER::ivreg()`, manual DiD |
| Discrete Choice | Logit, Probit, FEGLM | `stats::glm()`, `alpaca::feglm()` |
| Time Series | VAR, VARMA, VECM, IRF | `vars::VAR()`, `vars::irf()` |
| Forecasting | ARIMA, MSTL, Changepoint | `forecast::auto.arima()`, `forecast::mstl()`, `changepoint` |
| ML | K-means, DBSCAN, Hierarchical, PCA, t-SNE, RF, SVM | `stats::kmeans()`, `sklearn`, `stats::prcomp()` |
| Diagnostics | JB, BP, DW, VIF | `lmtest`, `car::vif()` |

**Directory Structure:**
```
validation/
├── README.md                          # Overview of validation framework
├── reference_implementations.md       # Catalog of R/Python references
├── regression/                        # OLS, robust SEs, clustered SEs
├── econometrics/                      # Panel, IV, DiD, discrete choice
│   └── timeseries/                    # VAR, VARMA, VECM, IRF
├── forecasting/                       # ARIMA, MSTL, changepoint
├── ml/                                # Clustering, reduction, supervised
├── diagnostics/                       # Regression diagnostics
└── datasets/                          # Standard datasets (Grunfeld, Longley, iris)
```

### Performance Framework (`performance/`)

Criterion benchmarks for all methods with R comparison scripts:

**Directory Structure:**
```
performance/
├── README.md                          # How to run benchmarks
├── methodology.md                     # Statistical methodology
├── benchmarks/                        # Criterion benchmark code (Rust)
├── results/                           # Raw benchmark data (CSV)
├── comparisons/
│   └── r_comparison/                  # R benchmark scripts (microbenchmark)
└── reports/                           # Markdown summary reports
```

### Comprehensive Benchmark Results (Rust vs R)

Distribution-based benchmarks with 100 measurement iterations after 10 warmup iterations. Uses `bench` package for R and custom distribution tracking for Rust.

#### Regression

| Method | n | p2a (µs) | R (µs) | **Speedup** | p2a Memory | R Memory |
|--------|---|----------|--------|-------------|------------|----------|
| OLS | 100 | 41.6 | 812.4 | **19.5x** | 36 KB | 468 KB |
| OLS + HC1 | 100 | 171.4 | 2,279.4 | **13.3x** | 72 KB | 570 KB |
| OLS | 1,000 | 119.2 | 963.2 | **8.1x** | 108 KB | 362 KB |
| OLS + HC1 | 1,000 | 295.2 | 4,112.7 | **13.9x** | 4 KB | 1.01 MB |
| OLS | 10,000 | 923.8 | 2,180.6 | **2.4x** | 0 B | 3.56 MB |
| OLS + HC1 | 10,000 | 1,571.2 | 25,224.3 | **16.1x** | 0 B | 10.09 MB |

#### Panel Data

| Method | n | p2a (µs) | R Package | R (µs) | **Speedup** |
|--------|---|----------|-----------|--------|-------------|
| Fixed Effects | 100 | 26.7 | plm | 4,901.0 | **183.6x** |
| Fixed Effects | 1,000 | 142.7 | plm | 6,388.0 | **44.8x** |
| Fixed Effects | 5,000 | 613.7 | plm | 11,296.0 | **18.4x** |
| HDFE (2-way) | 100 | 43.5 | lfe | 6,215.0 | **142.9x** |
| HDFE (2-way) | 1,000 | 249.0 | lfe | 6,287.9 | **25.3x** |
| HDFE (2-way) | 5,000 | 1,158.7 | lfe | 26,657.3 | **23.0x** |

#### Time Series

| Method | n | p2a (µs) | R Package | R (µs) | **Speedup** |
|--------|---|----------|-----------|--------|-------------|
| ARIMA(1,1,1) | 100 | 92.7 | forecast | 2,774.2 | **29.9x** |
| ARIMA(1,1,1) | 500 | 520.9 | forecast | 4,782.8 | **9.2x** |
| MSTL | 100 | 48.3 | forecast | 1,451.0 | **30.0x** |
| MSTL | 500 | 219.3 | forecast | 1,853.2 | **8.5x** |

#### Machine Learning

| Method | n | p2a (µs) | R (µs) | **Speedup** |
|--------|---|----------|--------|-------------|
| Logit | 1,000 | 401.3 | 2,175.8 | **5.4x** |
| Probit | 1,000 | 2,346.6 | 2,760.0 | **1.2x** |
| K-Means | 1,000 | 587.2 | 1,274.4 | **2.2x** |
| PCA | 5,000 | 227.1 | 797.1 | **3.5x** |

**Summary**: p2a achieves **1.2x to 183.6x speedup** across all methods compared to R reference implementations (median ~10x). Memory usage is typically orders of magnitude lower.

**Running Comprehensive Benchmarks:**
```bash
# Rust benchmarks with distribution statistics
cargo bench -p p2a-core --bench comprehensive_benchmarks

# R benchmarks (using bench package)
cd performance/comparisons/r_comparison
Rscript benchmark_comprehensive.R
```

**Full comparison report**: `performance/reports/comprehensive_comparison.md`

### Workflow Integration

The `implement_metrics` workflow now includes **Phase 6: Validation & Benchmarking**:

1. Create validation document in `validation/[category]/[method].md`
2. Add Criterion benchmark to `performance/benchmarks/`
3. Add R comparison script (if applicable)
4. Update performance reports

**New Skill:** `.claude/skills/validation-benchmarking/SKILL.md` provides guidance for validation and benchmarking new implementations.

**Updated Files:**
- `.claude/commands/implement_metrics.md` — Added Phase 6
- `.claude/agents/econometrics-implementer.md` — Added validation checklist
- `.claude/skills/validation-benchmarking/SKILL.md` — New skill

---

## Phase 6: LLM-Assisted Data Cleaning — ✅ COMPLETE

A new capability for interactive, verified data cleaning workflows powered by LLM assistance with built-in validation checkpoints.

### Vision

Enable LLM-driven data cleaning where each step is:
- **Inspected** — Automated data quality profiling
- **Diagnosed** — LLM identifies issues from profile
- **Proposed** — LLM suggests cleaning operation
- **Previewed** — Sample of what would change
- **Validated** — Verify assumptions/preconditions
- **Applied** — Execute with rollback point
- **Verified** — Compare before/after metrics

### Phase 6.1: Quality Profiling (Foundation) — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| DataQualityProfile struct | ✅ Complete | Pure Rust |
| ColumnProfile with statistics | ✅ Complete | Per-column analysis |
| DataIssue enum (issue types) | ✅ Complete | Automated detection |
| NumericStats (outliers, bounds) | ✅ Complete | Statistical analysis |
| StringStats (patterns, encoding) | ✅ Complete | String analysis |
| `data_quality_profile` MCP tool | ✅ Complete | MCP integration (tool #56) |

**Key Types:**

```rust
pub struct DataQualityProfile {
    pub columns: Vec<ColumnProfile>,
    pub row_count: usize,
    pub duplicate_rows: usize,
    pub completeness_score: f64,  // % non-null
}

pub struct ColumnProfile {
    pub name: String,
    pub dtype: String,
    pub null_count: usize,
    pub null_pct: f64,
    pub unique_count: usize,
    pub unique_pct: f64,
    pub numeric_stats: Option<NumericStats>,
    pub string_stats: Option<StringStats>,
    pub issues: Vec<DataIssue>,
}

pub enum DataIssue {
    HighNullRate { column: String, pct: f64 },
    PossibleDuplicates { columns: Vec<String>, count: usize },
    MixedTypes { column: String, examples: Vec<String> },
    OutlierValues { column: String, count: usize, bounds: (f64, f64) },
    InconsistentFormat { column: String, patterns: Vec<String> },
    WhitespaceIssues { column: String, count: usize },
    EncodingIssues { column: String, examples: Vec<String> },
}
```

### Phase 6.2: Preview & Verification — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| CleaningResult with verification | ✅ Complete | Wraps cleaning ops |
| VerificationReport struct | ✅ Complete | Before/after metrics |
| ChangeExample (sample changes) | ✅ Complete | Show what changed |
| QualityDelta (metric changes) | ✅ Complete | Quality comparison |
| CleaningPreview struct | ✅ Complete | Pre-execution preview |
| CleaningOperation enum | ✅ Complete | 8 operation types |
| `preview_cleaning` MCP tool | ✅ Complete | MCP integration (tool #57) |
| `verify_cleaning` MCP tool | ✅ Complete | MCP integration (tool #58) |

**Key Types:**

```rust
pub struct CleaningResult {
    pub dataset: Dataset,
    pub operation: String,
    pub verification: VerificationReport,
    pub rollback_id: String,
}

pub struct VerificationReport {
    pub rows_before: usize,
    pub rows_after: usize,
    pub rows_modified: usize,
    pub rows_removed: usize,
    pub sample_changes: Vec<ChangeExample>,
    pub quality_delta: QualityDelta,
    pub warnings: Vec<String>,
}
```

### Phase 6.3: Session Management — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| CleaningSession struct | ✅ Complete | Session state with checkpoints |
| Rollback points | ✅ Complete | Full dataset snapshots at each checkpoint |
| Session persistence | ✅ Complete | In-memory with checkpoint history |
| Audit trail | ✅ Complete | OperationRecord tracks all operations |
| `cleaning_session_start` MCP tool | ✅ Complete | Begin session with initial checkpoint |
| `cleaning_session_status` MCP tool | ✅ Complete | Check session progress |
| `list_cleaning_sessions` MCP tool | ✅ Complete | List all active sessions |
| `cleaning_session_apply` MCP tool | ✅ Complete | Apply operations within session |
| `cleaning_rollback` MCP tool | ✅ Complete | Undo to any checkpoint |
| `cleaning_session_checkpoints` MCP tool | ✅ Complete | List all checkpoints |

**Implementation Details:**

```rust
// Core session management types (cleaning_session.rs)
pub struct CleaningSession {
    pub id: String,
    pub dataset_name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub current_checkpoint: usize,
    checkpoints: Vec<SessionCheckpoint>,
    pub audit_trail: Vec<OperationRecord>,
    pub metadata: HashMap<String, String>,
}

pub struct SessionCheckpoint {
    pub id: String,
    pub index: usize,
    pub created_at: DateTime<Utc>,
    pub description: String,
    dataset: Dataset,  // Full snapshot
    pub quality_profile: DataQualityProfile,
}

pub struct OperationRecord {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub operation_type: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
    pub checkpoint_before: usize,
    pub checkpoint_after: Option<usize>,
    pub success: bool,
    pub error: Option<String>,
    pub verification: Option<VerificationReportSummary>,
}
```

**Key Features:**
- Each session maintains a checkpoint history with full dataset snapshots
- Automatic checkpoint creation on successful operations
- Rollback to any previous checkpoint (not just the last one)
- Quality profile comparison between checkpoints
- Complete audit trail with timestamps and parameters
- 10 unit tests covering all session operations

### Phase 6.4: Smart Suggestions — ✅ COMPLETE

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| Pattern recognition | ✅ Complete | Maps DataIssue types to cleaning operations |
| Suggestion engine | ✅ Complete | Priority-ranked CleaningSuggestion objects |
| Suggestion parameters | ✅ Complete | Operation-specific parameters for each suggestion |
| Impact estimation | ✅ Complete | Rows affected, completeness change |
| `suggest_cleaning` MCP tool | ✅ Complete | Proactive suggestions with filtering |

**Implementation Details:**

```rust
// Core suggestion types (suggestion.rs)
pub struct CleaningSuggestion {
    pub id: String,
    pub title: String,
    pub description: String,
    pub addresses_issue: String,
    pub priority: SuggestionPriority,  // Low, Medium, High, Critical
    pub category: CleaningCategory,     // MissingValues, Deduplication, etc.
    pub columns: Option<Vec<String>>,
    pub operation: String,              // "trim", "fill_na", "deduplicate", etc.
    pub parameters: SuggestionParameters,
    pub estimated_impact: EstimatedImpact,
    pub reasoning: String,
    pub considerations: Vec<String>,
}

pub struct SuggestionReport {
    pub suggestions: Vec<CleaningSuggestion>,
    pub issues_analyzed: usize,
    pub dataset_summary: DatasetSummary,
    pub overall_recommendation: String,
}
```

**Pattern Recognition:**
- HighNullRate → fill_na with median (numeric) or mode (string)
- WhitespaceIssues → trim (high priority)
- EmptyStrings → replace with null
- OutlierValues → filter or review
- DuplicateRows → deduplicate
- InconsistentFormat → standardize
- ConstantColumn → drop_column
- MixedTypes → cast

**Key Features:**
- Automatic operation parameter inference (fill value, strategy)
- Impact estimation (rows affected, completeness change)
- Reasoning explains why each suggestion is made
- Considerations list risks and alternatives
- 9 unit tests covering all suggestion types

### Example Interaction Flow

```
User: "Clean this sales data"

LLM: [calls data_quality_profile]
     "I've analyzed your dataset. Here are the issues I found:
      1. 'email' column: 15% have whitespace issues
      2. 'price' column: 3 outliers > $10,000
      3. 'date' column: mixed formats (MM/DD/YYYY and YYYY-MM-DD)
      4. 12 duplicate rows based on order_id

      I recommend we address these in order. Start with whitespace?"

User: "Yes, fix the emails"

LLM: [calls preview_cleaning with trim operation, sample=5]
     "Here's what the email cleaning would do:

      | Row | Before              | After            |
      |-----|---------------------|------------------|
      | 23  | ' john@test.com '   | 'john@test.com'  |
      | 45  | 'jane@test.com  '   | 'jane@test.com'  |

      This will modify 1,847 rows. Proceed?"

User: "Yes"

LLM: [calls apply_cleaning_verified]
     "✓ Email cleaning complete.

      Verification:
      - Rows modified: 1,847
      - Email uniqueness preserved
      - Quality score: 84% → 87%

      Next issue: Price outliers?"
```

### Files Created

```
crates/p2a-core/src/data/
├── quality.rs          # ✅ DataQualityProfile, ColumnProfile, DataIssue
├── verification.rs     # ✅ VerificationReport, CleaningResult, preview_cleaning
├── cleaning_session.rs # ✅ CleaningSession with rollback and audit trail
└── suggestion.rs       # ✅ CleaningSuggestion, SuggestionReport, generate_suggestions

crates/p2a-mcp/src/server.rs  # ✅ 10 new tools for LLM-assisted cleaning
```

**New MCP Tools (Phase 6):**
1. `data_quality_profile` - Generate comprehensive quality analysis
2. `preview_cleaning` - Preview operation before applying
3. `verify_cleaning` - Verify operation after applying
4. `cleaning_session_start` - Start a new cleaning session
5. `cleaning_session_status` - Check session progress
6. `list_cleaning_sessions` - List all active sessions
7. `cleaning_session_apply` - Apply operation within session
8. `cleaning_rollback` - Rollback to previous checkpoint
9. `cleaning_session_checkpoints` - List all checkpoints
10. `suggest_cleaning` - Generate smart cleaning suggestions

---

## Phase 6.5: Regression Discontinuity Design — ✅ COMPLETE

Implementation of local polynomial Regression Discontinuity (RD) estimation with robust bias-corrected confidence intervals based on Calonico, Cattaneo, Titiunik & Farrell (2014-2020) methodology.

### Features Implemented

| Deliverable | Status | Implementation |
|-------------|--------|----------------|
| Sharp RD estimation | ✅ Complete | Local polynomial with kernel weighting |
| Fuzzy RD estimation | ✅ Complete | IV-style Wald estimator (LATE) |
| MSE-optimal bandwidth | ✅ Complete | Imbens-Kalyanaraman (2012) formula |
| CER-optimal bandwidth | ✅ Complete | Coverage error rate optimal |
| Bias correction | ✅ Complete | Higher-order polynomial bias estimation |
| Robust inference | ✅ Complete | Conventional, bias-corrected, robust CIs |
| Kernel functions | ✅ Complete | Triangular, Epanechnikov, Uniform |
| NN variance | ✅ Complete | Nearest-neighbor variance estimator |
| HC variance | ✅ Complete | HC0-HC3 heteroskedasticity-consistent |
| `rd_estimate` MCP tool | ✅ Complete | Sharp RD estimation |
| `rd_bw` MCP tool | ✅ Complete | Bandwidth selection only |
| `rd_fuzzy` MCP tool | ✅ Complete | Fuzzy RD (LATE) estimation |

### Key Types

```rust
/// Configuration for RD estimation
pub struct RdConfig {
    pub p: usize,               // Polynomial order (default: 1 = local linear)
    pub q: Option<usize>,       // Bias polynomial order (default: p+1)
    pub h: Option<f64>,         // Main bandwidth (auto if None)
    pub b: Option<f64>,         // Bias bandwidth (auto if None)
    pub kernel: KernelType,     // Triangular, Epanechnikov, Uniform
    pub bwselect: BandwidthMethod, // MseRd, MseTwo, CerRd, CerTwo
    pub vce: VceType,           // Nn, Hc0, Hc1, Hc2, Hc3
    pub nnmatch: usize,         // NN neighbors (default: 3)
    pub level: f64,             // Confidence level (default: 0.95)
}

/// Result from Sharp RD estimation
pub struct RdResult {
    // Point estimates
    pub tau_conventional: f64,   // Conventional RD estimate
    pub tau_bc: f64,             // Bias-corrected estimate
    pub tau_robust: f64,         // Robust bias-corrected estimate

    // Standard errors and inference
    pub se_robust: f64,
    pub ci_robust: (f64, f64),
    pub p_robust: f64,
    pub significance: SignificanceLevel,

    // Bandwidth and specification
    pub h: f64,                  // Main bandwidth
    pub b: f64,                  // Bias bandwidth
    pub n_eff_left: usize,       // Effective sample left of cutoff
    pub n_eff_right: usize,      // Effective sample right of cutoff
    // ...
}

/// Result from Fuzzy RD estimation (LATE)
pub struct FuzzyRdResult {
    pub outcome: String,
    pub treatment: String,
    pub running_var: String,
    pub cutoff: f64,

    // Wald estimator: tau_fuzzy = tau_Y / tau_D
    pub tau_conventional: f64,
    pub tau_bc: f64,
    pub tau_robust: f64,
    pub se_robust: f64,
    pub ci_robust: (f64, f64),

    // First stage (treatment discontinuity)
    pub first_stage_tau: f64,
    pub first_stage_se: f64,
    pub first_stage_f: f64,      // Effective F-statistic
    // ...
}
```

### Mathematical Implementation

**Local Polynomial Estimator:**
```
β̂ = (X'WX)⁻¹ X'Wy

where W = diag(K((xᵢ - c)/h))
      K(u) = kernel function

Treatment effect: τ̂ = β̂₀⁺ - β̂₀⁻
```

**MSE-Optimal Bandwidth (Imbens-Kalyanaraman 2012):**
```
h_MSE = C_k × [σ²(c) / (f(c) × (m''(c))²)]^(1/5) × n^(-1/5)
```

**Bias Correction:**
```
τ̂_bc = τ̂ - h^(p+1) × B̂

where B̂ is estimated using order-q polynomial with bandwidth b
```

**Robust Inference:**
The robust standard error accounts for both estimation error and bias estimation error, providing valid confidence intervals.

### Files Created/Modified

```
crates/p2a-core/src/econometrics/
├── mod.rs           # Added mod rd; and exports
└── rd.rs            # NEW: ~900+ lines, full RD implementation

crates/p2a-core/src/lib.rs        # Added RD re-exports
crates/p2a-mcp/src/server.rs      # Added 3 RD tools
docs/guides/ECONOMETRICS_GUIDE.md # Added RD documentation section
```

### References

1. Calonico, Cattaneo & Titiunik (2014). "Robust Nonparametric Confidence Intervals for RD Designs". *Econometrica* 82(6): 2295-2326.
2. Calonico, Cattaneo & Farrell (2020). "Optimal Bandwidth Choice for Robust Bias Corrected Inference in RD Designs". *Econometrics Journal* 23(2): 192-210.
3. Imbens & Kalyanaraman (2012). "Optimal Bandwidth Choice for the RD Estimator". *Review of Economic Studies* 79(3): 933-959.
4. R package `rdrobust` (reference implementation)

---

## Phase 7: Dioxus Cross-Platform App — ✅ COMPLETE

**Date:** January 2026

### Overview

The `p2a-dioxus` crate provides a pure Rust cross-platform frontend (web, desktop, mobile) using Dioxus 0.7. It communicates with the p2a-mcp HTTP backend via REST API and Server-Sent Events (SSE) for streaming.

### Key Features Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Cross-platform build | ✅ | Web (WASM), Desktop (native), Mobile (planned) |
| Chat interface with streaming | ✅ | SSE-based streaming with real-time updates |
| Multi-provider LLM support | ✅ | Ollama, Anthropic, OpenAI |
| Conversation management | ✅ | Create, rename, archive, delete with SurrealDB persistence |
| Dataset sidebar | ✅ | Live view of loaded datasets with metadata and refresh |
| Tool call transparency | ✅ | "Rust Analytics" indicator showing which tools were called |
| Tool call details | ✅ | Expandable cards showing arguments and results |
| Environment variable detection | ✅ | Auto-detects OPENAI_API_KEY, ANTHROPIC_API_KEY on desktop |
| Settings persistence | ✅ | localStorage (web) / file-based (native) |
| Markdown rendering | ✅ | Assistant messages rendered as markdown |
| Prompt history navigation | ✅ | Arrow keys to navigate previous prompts |
| Create inline datasets | ✅ | `create_dataset` tool for generated/test data |

### Architecture

```
p2a-dioxus/
├── src/
│   ├── main.rs              # Entry point (web/desktop conditional)
│   ├── app.rs               # Root App component
│   ├── api/
│   │   ├── client.rs        # HTTP client with conversation endpoints
│   │   ├── sse.rs           # SSE streaming for LLM chat
│   │   └── types.rs         # API types (StreamEvent, DatasetMeta, etc.)
│   ├── state/
│   │   ├── chat.rs          # Messages, tool calls, history navigation
│   │   ├── conversation.rs  # Conversation list and selection
│   │   ├── session.rs       # Backend session, dataset refresh counter
│   │   └── settings.rs      # Provider config with env var detection
│   ├── components/
│   │   ├── chat_panel.rs    # Main chat interface
│   │   ├── message.rs       # Message with "Rust Analytics" indicator
│   │   ├── tool_call.rs     # Expandable tool call card
│   │   ├── dataset_sidebar.rs # Dataset list with auto-refresh
│   │   ├── conversation_sidebar.rs # Conversation CRUD
│   │   ├── chat_input.rs    # Input with keyboard shortcuts
│   │   └── settings_modal.rs # Provider configuration
│   └── utils/
│       └── markdown.rs      # Markdown to RSX (pulldown-cmark)
└── assets/
    └── styles.css           # Tailwind-like CSS
```

### SSE Streaming Protocol

The backend sends progress events during LLM chat:

```rust
enum ProgressEvent {
    Status { message: String },
    ToolStart { tool: String, arguments: serde_json::Value },
    ToolEnd { tool: String, elapsed_ms: u64, result: Option<String> },
    ToolResult { tool: String, images: Vec<ImageData> },
    Content { text: String },
    Done { message: Message },
    Error { error: String },
}
```

The frontend tracks tool calls in real-time:
- `ToolStart`: Adds pending tool call with arguments to current message
- `ToolEnd`: Marks tool call as complete with result
- `Done`: Finalizes message with full tool call data from backend

### Tool Call Display

Assistant messages show tool call transparency:

1. **"Rust Analytics" indicator** (green banner at top of message)
   - Shows immediately which tools were called
   - Displays tool names as compact chips

2. **Expandable tool cards** (below message content)
   - Shows tool name with success/running/error status
   - Expandable to show arguments JSON
   - Shows truncated result (up to 2KB via SSE)

### Dataset Management

- **Dataset Sidebar**: Shows all loaded datasets with metadata (rows, columns, type)
- **Auto-refresh**: Sidebar refreshes after any tool execution via `datasets_refresh_counter`
- **`create_dataset` tool**: Allows LLM to create datasets from inline CSV content
- **Metadata persistence**: Dataset metadata stored in SurrealDB for session restoration

### Running the App

```bash
# Terminal 1: Backend (embedded in desktop, or separate for web)
cargo run -p p2a-mcp --features full -- --transport http --port 8081 --cors-permissive

# Terminal 2: Dioxus dev server
cd crates/p2a-dioxus

# Web
dx serve

# Desktop (includes embedded backend)
dx serve --platform desktop
```

### Files Created/Modified

```
crates/p2a-dioxus/                    # NEW: Entire crate
crates/p2a-mcp/src/server.rs          # Added create_dataset tool
crates/p2a-mcp/src/transport/http.rs  # Added arguments/result to ToolStart/ToolEnd events
crates/p2a-mcp/src/llm/tools.rs       # Updated system prompt for create_dataset
crates/p2a-mcp/src/transport/conversation.rs # Added DatasetMetaResponse DTO
```
