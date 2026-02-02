# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Tool registry for MCP with 256 documented tools across 21 categories (`tools/registry.rs`)
- Documentation generation for MCP tools (`generate_markdown_docs()`)
- Comprehensive cookbook with CLI examples (`docs/cookbook.md`)
- MCP tools reference documentation (`docs/mcp/tools-reference.md`)

### Changed
- **MCP Server Architecture**: Refactored `server.rs` (previously 28,000+ lines) into modular structure:
  - Tool handlers organized into 17 category modules in `tools/handlers/`
  - Request types organized into matching modules in `tools/requests/`
  - Router composition using rmcp's `#[tool_router]` pattern
  - Categories: utils, database, data, viz, ml, stats, hypothesis, regression, panel, discrete, causal, timeseries, spatial, munging, survival, cleaning

### Fixed
- Doc examples now compile properly (changed from `ignore` to proper Result wrappers)
- CI workflow improvements for p2a-dioxus builds

## [0.1.0] - 2026-01-30

### Added

#### Core Analytics (p2a-core)
- **Regression**: OLS with robust SEs (HC0-HC3), clustered SEs, HAC (Newey-West), bootstrap, Driscoll-Kraay
- **Diagnostics**: Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF, RESET, Breusch-Godfrey
- **Panel Data**: Fixed effects, random effects, Hausman test, HDFE, Arellano-Bond GMM, PVCM, PMG
- **Instrumental Variables**: 2SLS, first-stage diagnostics, Sargan test, Balke-Pearl bounds, MTE
- **Causal Inference**:
  - DiD (canonical, Callaway-Sant'Anna staggered, Bacon decomposition, ETWFE)
  - RD (sharp, fuzzy, multi-cutoff with CCT robust inference)
  - Synthetic control (classic, gsynth, SCPI)
  - Matching (propensity score, CEM, nearest neighbor, full matching)
  - Treatment effects (IPW, doubly robust, TMLE, CTMLE, LTMLE, Double ML)
  - Mediation analysis (causal mediation, natural effects)
- **Discrete Choice**: Logit, probit, multinomial, ordered, mixed logit, negative binomial, ZIP, ZINB, hurdle
- **Time Series**: ARIMA, VAR, VARMA, VECM, GARCH, Holt-Winters, STL/MSTL, Kalman filter, changepoint detection
- **Spatial Econometrics**: SAR, SEM, SAC, spatial probit, panel spatial (SPML, SPGM), Moran's I, local Moran (LISA)
- **Survival Analysis**: Kaplan-Meier, Cox PH, AFT, competing risks
- **Machine Learning**: K-means, DBSCAN, HDBSCAN, OPTICS, hierarchical, spectral, GMM, PCA, t-SNE, MDS, Random Forest, SVM, causal forest, BART
- **Statistics**: 50+ hypothesis tests, power analysis, factor analysis, canonical correlation
- **Visualization**: Static (PNG) and interactive (HTML/Plotly) charts
- **Export**: LaTeX, Markdown, HTML tables; CSV export for all result types
- **Data Management**: CSV, Parquet, Excel, Stata, SAS loading; SQLite, DuckDB queries; quality profiling; cleaning sessions

#### CLI (p2a-cli)
- Full command-line interface for all analytics functions
- Session recording for reproducibility
- Script export/import for automation
- JSON output format for programmatic use

#### MCP Server (p2a-mcp)
- 256 analytics tools exposed via Model Context Protocol
- HTTP transport with SSE streaming for chat
- Session management with SurrealDB persistence
- Audit logging support
- LLM integration (Ollama, Anthropic, OpenAI)

#### Dioxus App (p2a-dioxus)
- Cross-platform GUI (web via WASM, desktop via native)
- Chat interface with streaming LLM responses
- Conversation history with persistence
- Dataset sidebar with live metadata
- Tool call transparency

### Technical
- Pure Rust implementation (no external R/Python dependencies for core algorithms)
- Matrix operations via `faer` 0.22
- DataFrames via `polars` 0.52
- MCP protocol via `rmcp` 0.8
- Visualization via `plotters` (static) and `plotlars` (interactive)

## [0.0.1] - 2025-01-01

### Added
- Initial project structure
- Basic OLS regression
- CSV data loading
- Simple CLI interface

---

## Categories

- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes
