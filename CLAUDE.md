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

# GPU-accelerated build (optional, requires CUDA)
cargo build -p p2a-core --features cuda
cargo test -p p2a-core --features cuda --release  # Tests with GPU dispatch

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

**Linux (Ubuntu/Debian) - x86_64:**
```bash
sudo apt-get install libopenblas-dev libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
```

**Linux (Ubuntu/Debian) - ARM64 (aarch64):**
```bash
# Core build dependencies
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

**Dioxus CLI** (for p2a-dioxus):
```bash
cargo install dioxus-cli
rustup target add wasm32-unknown-unknown
```

### ARM64 Build Notes

On ARM64 (aarch64), the project includes a `.cargo/config.toml` that enables `opt-level = 1` for debug builds. This is required because p2a-mcp is a large binary that exceeds ARM64's ±128MB relocation range in pure debug mode.

- **Debug builds**: Work with `opt-level = 1` (configured automatically)
- **Release builds**: Work without modification
- **Desktop builds**: Require `libxdo-dev` for X11 automation support

## Project Overview

prompt2analytics is a Rust workspace (edition 2024, requires Rust 1.85+) exposing 257 econometrics, statistics, ML, and visualization methods through multiple interfaces:

- **p2a-core**: Core analytics library (all algorithms)
- **p2a-cli**: Command-line interface (`p2a` binary) with session recording, script export, and JSON output
- **p2a-mcp**: MCP server exposing 257 tools with LLM integration
- **p2a-dioxus**: Cross-platform GUI (web via WASM, desktop via native)

## CLI (p2a-cli)

The CLI provides direct access to all analytics functions with session-based reproducibility.

### Session Pattern

Use `--session` to record commands for reproducibility:
```bash
# All commands in a session are recorded to the JSON file
p2a --session analysis.json data load sales.csv --name sales
p2a --session analysis.json reg ols sales -y price -x sqft bedrooms --robust hc1
p2a --session analysis.json viz scatter sales -x sqft -y price -f scatter.png

# Export session to executable script
p2a script export analysis.json -o analysis.sh

# Replay a script
p2a script run analysis.sh
```

### Command Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `data` | Load, describe, generate, export | `data load`, `data describe`, `data head` |
| `reg` | OLS, clustered SEs, HAC, bootstrap, quantile | `reg ols`, `reg clustered`, `reg hac` |
| `panel` | FE, RE, Hausman, HDFE, FEGLM, GMM | `panel fe`, `panel re`, `panel hdfe` |
| `causal` | IV, DiD, RD, matching, IPW, synth | `causal iv`, `causal did`, `causal rd` |
| `discrete` | Logit, probit, ordered, multinomial, count | `discrete logit`, `discrete mlogit` |
| `ts` | ARIMA, VAR, GARCH, Holt-Winters, STL | `ts arima`, `ts var`, `ts garch` |
| `stats` | T-tests, ANOVA, non-parametric, ACF/PACF | `stats t-test-two`, `stats anova` |
| `spatial` | SAR, SEM, SAC, Moran's I | `spatial sar`, `spatial moran` |
| `survival` | Kaplan-Meier, Cox PH, log-rank | `survival km`, `survival cox` |
| `ml` | K-means, DBSCAN, PCA, t-SNE, RF, SVM | `ml kmeans`, `ml pca` |
| `munge` | Filter, join, reshape, aggregate, clean | `munge filter`, `munge join` |
| `viz` | Static (PNG) and interactive (HTML) charts | `viz scatter`, `viz histogram` |

### Output Formats

```bash
p2a --format text ...   # Human-readable (default)
p2a --format json ...   # Programmatic use
p2a --format table ...  # Tabular display
```

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

### Feature Flags

```bash
# Build with all features
cargo build -p p2a-core --all-features

# Specific features
cargo build -p p2a-core --features spectral-analysis  # Spectral analysis (spectrum, periodogram)
cargo build -p p2a-core --features cuda               # GPU acceleration (requires CUDA toolkit)
```

### GPU Acceleration (Optional)

The `cuda` feature enables transparent GPU dispatch for core linear algebra operations via cuBLAS and cuSOLVER. GPU paths are selected automatically when problem sizes exceed calibrated thresholds.

**GPU-accelerated operations** (via `linalg/gpu/` module):
- **xtx (X'X)**: 2-7x speedup for k>=30 via cuBLAS DGEMM
- **PCA**: 4-13x speedup via GPU covariance eigendecomposition
- **K-means distances**: 2-3x speedup for d>=20 via DGEMM reformulation
- **matmul**: 1.4-2.4x speedup for near-square matrices
- **Sandwich estimators**: 2-3x speedup for k>=50 (HC0-HC3, HAC)
- **cholesky_inverse**: Marginal; mainly avoids host-device round-trips

**Not GPU-accelerated** (CPU always faster):
- xty (X'y): DGEMV is bandwidth-bound; CPU OpenBLAS is faster
- Small matrices (k<30, n<5K): Transfer overhead exceeds compute savings
- Tall-skinny matmul: CPU BLAS optimized for these shapes

Dispatch thresholds are configurable via environment variables (`P2A_GPU_XTX_MIN_NKK`, etc.) and calibrated on DGX Spark. See `performance/reports/gpu_performance.md` for full benchmark results and threshold rationale.

```bash
# GPU benchmarks (95 configurations)
cargo bench -p p2a-core --bench gpu_benchmarks --features cuda

# CPU-only baseline for comparison
P2A_GPU_XTX_MIN_NKK=999999999999 P2A_GPU_XTX_MIN_K=999999 \
  cargo bench -p p2a-core --bench gpu_benchmarks --features cuda
```

### Module Organization (p2a-core)

```
src/
├── errors.rs              # EconError, EconResult types
├── cache.rs               # Thread-safe LRU cache with memory limits and TTL
├── memory.rs              # Memory monitoring, pressure detection, cleanup
├── linalg/
│   ├── matrix_ops.rs      # xtx, xty, safe_inverse, cholesky (via faer), GPU dispatch
│   └── gpu/               # Optional CUDA acceleration (#[cfg(feature = "cuda")])
│       ├── mod.rs          # GpuContext singleton (OnceLock)
│       ├── context.rs      # CUDA device init, capability detection
│       ├── memory.rs       # ndarray <-> CudaSlice (row-major handling)
│       ├── dispatch.rs     # Size thresholds (env-configurable)
│       ├── blas.rs         # cuBLAS: xtx (DSYRK→DGEMM), xty (DGEMV), matmul (DGEMM)
│       ├── solver.rs       # cuSOLVER: cholesky_inverse, sandwich_meat
│       └── kernels.rs      # Custom: pairwise_distances (K-means)
│   └── design.rs          # DesignMatrix, demeaning functions
├── traits/
│   └── estimator.rs       # LinearEstimator trait, SignificanceLevel, p-value helpers
│
├── regression/            # Regression methods
│   ├── ols.rs             # OLS, HC0-HC3, clustered SEs, HAC (Newey-West), bootstrap, Driscoll-Kraay
│   ├── diagnostics.rs     # JB, BP, DW, VIF, Breusch-Godfrey, RESET, Wald, Harvey-Collier
│   ├── nls.rs             # Nonlinear least squares (Levenberg-Marquardt)
│   ├── loess.rs           # Local polynomial regression (LOESS/LOWESS)
│   ├── gls.rs             # Generalized least squares (AR1, custom correlation)
│   ├── smooth_spline.rs   # Smoothing splines with GCV
│   ├── step.rs            # Stepwise selection (forward, backward, both)
│   ├── quantreg.rs        # Quantile regression (interior point, simplex)
│   ├── marginal_effects.rs # Marginal effects and contrasts
│   ├── sensemakr.rs       # Sensitivity analysis (Cinelli & Hazlett)
│   └── evalue.rs          # E-values for unmeasured confounding
│
├── stats/                 # Statistical tests (50+ methods)
│   ├── ttest.rs           # One-sample, two-sample, paired t-tests
│   ├── anova.rs           # One-way, two-way ANOVA
│   ├── manova.rs          # Multivariate ANOVA (Pillai, Wilks, Hotelling, Roy)
│   ├── chisq.rs           # Chi-squared (goodness-of-fit, independence)
│   ├── fisher.rs          # Fisher exact test
│   ├── wilcoxon.rs        # Wilcoxon rank-sum and signed-rank
│   ├── kruskal.rs         # Kruskal-Wallis test
│   ├── friedman.rs        # Friedman test
│   ├── shapiro.rs         # Shapiro-Wilk normality test
│   ├── ks.rs              # Kolmogorov-Smirnov test
│   ├── bartlett.rs        # Bartlett's test for homogeneity of variance
│   ├── tukey.rs           # Tukey HSD post-hoc test
│   ├── factanal.rs        # Factor analysis (MLE with rotation)
│   ├── cancor.rs          # Canonical correlation analysis
│   ├── acf.rs             # ACF, PACF, CCF
│   ├── boxtest.rs         # Box-Ljung, Box-Pierce tests
│   ├── pptest.rs          # Phillips-Perron unit root test
│   ├── power.rs           # Power analysis (t-test, prop test, ANOVA)
│   ├── robust.rs          # Robust statistics (fivenum, IQR, MAD, ECDF, density)
│   ├── spline.rs          # Spline interpolation and approximation
│   ├── weighted.rs        # Weighted mean and covariance
│   └── ...                # 30+ more statistical tests
│
├── econometrics/          # Econometric methods (60+ methods)
│   ├── panel.rs           # FE, RE, Hausman, Panel GLS, Arellano-Bond GMM, PVCM, PMG
│   ├── iv.rs              # 2SLS, first-stage diagnostics, Sargan test
│   ├── did.rs             # Canonical 2x2 DiD
│   ├── staggered_did.rs   # Callaway-Sant'Anna staggered DiD
│   ├── etwfe.rs           # Extended two-way fixed effects (Wooldridge)
│   ├── bacon.rs           # Goodman-Bacon decomposition
│   ├── discrete.rs        # Logit, Probit, Multinomial, Ordered, NegBin, ZIP, ZINB, Hurdle, Mixed logit
│   ├── feglm.rs           # GLM with HDFE (IRLS + weighted MAP)
│   ├── hdfe.rs            # High-dimensional fixed effects
│   ├── rd.rs              # Sharp/Fuzzy RD with CCT robust inference
│   ├── rdmulti.rs         # Multi-cutoff RD
│   ├── synth.rs           # Synthetic control (classic + gsynth)
│   ├── scpi.rs            # Synthetic control with prediction intervals
│   ├── treatment.rs       # IPW, doubly robust estimation
│   ├── tmle.rs            # Targeted MLE
│   ├── ctmle.rs           # Collaborative TMLE
│   ├── ltmle.rs           # Longitudinal TMLE
│   ├── doubleml.rs        # Double/Debiased ML (PLR, PLIV, IRM, IIVM)
│   ├── matching.rs        # Propensity score matching (MatchIt)
│   ├── weightit.rs        # Flexible IPW (entropy balancing)
│   ├── cbps.rs            # Covariate balancing propensity scores
│   ├── twang.rs           # GBM propensity scores
│   ├── mediation.rs       # Causal mediation analysis
│   ├── medflex.rs         # Natural effect models
│   ├── survival.rs        # Kaplan-Meier, Cox PH, AFT, competing risks
│   ├── spatial.rs         # SAR, SEM, SAC models
│   ├── spatialprobit.rs   # Spatial probit models
│   ├── splm.rs            # Spatial panel models (SPML, SPGM)
│   ├── sphet.rs           # Spatial GMM with heteroskedasticity
│   ├── timeseries.rs      # VAR, VARMA, VECM, IRF, Granger causality
│   ├── panel_unit_root.rs # LLC, IPS, Hadri panel unit root tests
│   └── ...                # ivmte, hettx, stdreg, gformula, bpbounds, sbw
│
├── forecasting/           # Time series forecasting
│   ├── arima_model.rs     # ARIMA modeling and forecasting
│   ├── holtwinters.rs     # Holt-Winters exponential smoothing
│   ├── ar.rs              # AR model fitting (Yule-Walker, OLS, MLE)
│   ├── stl.rs             # STL decomposition
│   ├── mstl.rs            # Multiple seasonal decomposition (MSTL)
│   ├── decompose.rs       # Classical decomposition (additive/multiplicative)
│   ├── kalman.rs          # Kalman filter and smoother
│   ├── structts.rs        # Structural time series (local level, trend, BSM)
│   ├── changepoint.rs     # PELT and binary segmentation
│   ├── garch.rs           # GARCH(p,q) volatility modeling
│   ├── causal_impact.rs   # Bayesian structural time series causal inference
│   └── tsutils.rs         # lag, embed, diffinv, filter, window, arima_sim, runmed
│
├── ml/                    # Machine learning
│   ├── clustering.rs      # K-means (k-means++), DBSCAN, Hierarchical (Ward, single, complete, average)
│   ├── reduction.rs       # PCA (via SVD), t-SNE
│   ├── trees.rs           # Random Forest (CART)
│   └── svm.rs             # Linear SVM (SMO)
│
├── simulation/            # Data simulation
│   └── generator.rs       # Synthetic data generation for testing
│
├── visualization/         # Chart generation
│   ├── charts.rs          # Static charts (plotters) - PNG output
│   ├── heatmap.rs         # Correlation heatmaps
│   └── interactive.rs     # Interactive charts (plotlars/Plotly) - HTML output
│
├── export/                # Export formats
│   ├── latex.rs           # LaTeX tables (OLS, Panel, Discrete)
│   ├── markdown.rs        # Markdown tables for documentation
│   ├── html.rs            # Self-contained HTML tables
│   └── csv.rs             # CsvExport trait for all result types
│
├── reports/               # Report generation
│   └── html.rs            # HTML report builder
│
└── data/                  # Data management
    ├── quality.rs         # DataQualityProfile for LLM-assisted cleaning
    ├── verification.rs    # Cleaning verification and preview
    ├── cleaning_session.rs # Rollback-enabled cleaning sessions
    ├── database.rs        # SQLite and DuckDB connectivity
    ├── stata.rs           # Stata .dta file support
    ├── sas.rs             # SAS .sas7bdat file support
    └── munging/           # Data manipulation (reshape, aggregate, join, transform)
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

Common error variants:
- `EconError::InvalidInput(String)` - Bad input data
- `EconError::SingularMatrix` - Non-invertible matrix
- `EconError::ColumnNotFound(String)` - Missing column
- `EconError::InsufficientObservations` - Not enough data
- `EconError::ConvergenceFailure(String)` - Optimization didn't converge

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

Additional variance estimators in `regression/ols.rs`:
- `vcov_hac()` - HAC (Newey-West) for time series
- `vcov_bootstrap()` - Bootstrap covariance (pairs, residual, wild)
- `vcov_driscoll_kraay()` - Panel-robust SEs (cross-sectional dependence)

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

### Config Pattern for Complex Methods

Complex methods use a builder-style config:
```rust
let config = StaggeredDidConfig {
    comparison_group: ComparisonGroup::NeverTreated,
    estimation_method: AttEstimationMethod::Ipw,
    anticipation: 0,
    aggregation: Aggregation::Simple,
    ..Default::default()
};
let result = run_staggered_did(dataset, &config)?;
```

## MCP Server (p2a-mcp)

### Module Organization

The MCP server exposes 257 tools organized into modular handler files:

```
crates/p2a-mcp/src/
├── server.rs              # AnalyticsServer struct + router composition
├── tools/
│   ├── mod.rs             # Re-exports
│   ├── registry.rs        # Tool metadata for documentation
│   ├── requests/          # Request structs by category
│   │   ├── mod.rs
│   │   ├── causal.rs      # Causal inference requests
│   │   ├── data.rs        # Data management requests
│   │   ├── discrete.rs    # Discrete choice requests
│   │   ├── hypothesis.rs  # Hypothesis testing requests
│   │   ├── ml.rs          # Machine learning requests
│   │   ├── munging.rs     # Data munging requests
│   │   ├── panel.rs       # Panel data requests
│   │   ├── regression.rs  # Regression requests
│   │   ├── spatial.rs     # Spatial econometrics requests
│   │   ├── stats.rs       # Statistics requests
│   │   ├── timeseries.rs  # Time series requests
│   │   └── ...            # Other category modules
│   └── handlers/          # Tool implementations
│       ├── mod.rs
│       ├── causal.rs      # 40+ causal inference tools
│       ├── data.rs        # Data management tools
│       ├── discrete.rs    # Discrete choice tools
│       ├── hypothesis.rs  # 20 hypothesis testing tools
│       ├── ml.rs          # ML tools
│       ├── munging.rs     # 40+ data munging tools
│       ├── panel.rs       # Panel data tools
│       ├── regression.rs  # Regression tools
│       ├── spatial.rs     # Spatial econometrics tools
│       ├── stats.rs       # Statistics tools
│       ├── timeseries.rs  # 30+ time series tools
│       └── ...            # Other category modules
```

### Adding a New Tool

1. Define the request struct in `tools/requests/<category>.rs`:
```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MyToolRequest {
    #[schemars(description = "Name or ID of a previously loaded dataset.")]
    pub dataset: String,
    #[schemars(description = "Description of this parameter.")]
    pub param: String,
}
```

2. Add the tool handler in `tools/handlers/<category>.rs`:
```rust
#[tool(description = "My tool description")]
async fn my_tool(
    &self,
    Parameters(request): Parameters<MyToolRequest>,
) -> Result<CallToolResult, McpError> {
    // Implementation
    Ok(CallToolResult::success(vec![Content::text(result.to_string())]))
}
```

3. Import the request type in the handler module and ensure it's re-exported from `tools/requests/mod.rs`

### Router Composition

Each handler module defines a router via `#[tool_router(router = <name>_router, vis = "pub")]`.
These are composed in `server.rs`:

```rust
let tool_router = Self::tool_router()
    + Self::utils_router()
    + Self::database_router()
    + Self::data_router()
    + Self::viz_router()
    + Self::ml_router()
    + Self::stats_router()
    + Self::hypothesis_router()
    + Self::regression_router()
    + Self::panel_router()
    + Self::discrete_router()
    + Self::causal_router()
    + Self::timeseries_router()
    + Self::spatial_router()
    + Self::munging_router()
    + Self::survival_router()
    + Self::cleaning_router();
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

## Docker Deployment

Docker is for **deployment**, not development. For development, run services natively.

```bash
# Build and run backend
docker compose up --build

# With local LLM (Ollama)
docker compose --profile with-ollama up --build

# Health check
curl http://localhost:8080/health
```

For development, prefer native execution for faster iteration:
```bash
# Terminal 1: Backend
cargo run -p p2a-mcp --features full -- --transport http --host 127.0.0.1 --port 8080 --cors-permissive

# Terminal 2: Frontend
cd crates/p2a-dioxus && dx serve
```

## Validation & Benchmarking

### Validation Framework (`validation/`)

All methods are validated against reference R implementations. Validation has two layers:

1. **Rust tests** with `test_validate_*` prefix compare outputs to known R values
2. **Validation documents** in `validation/[category]/method.md` record R code, expected values, and precision

```bash
# Run all validation tests
cargo test -p p2a-core -- test_validate

# Run specific method validation
cargo test -p p2a-core -- test_validate_ols
cargo test -p p2a-core -- test_validate_hdfe

# Full validation (Rust + R scripts)
./validation/run_validation.sh

# Filter by category
./validation/run_validation.sh --category stats
```

Tolerance guidelines: coefficients to 1e-10 for n>1000, 1e-6 for n<100. Iterative methods (HDFE, MLE) may have slightly larger differences. See `validation/README.md` for details.

Key directories:
- `validation/regression/` - OLS, robust SEs, GLS, LOESS, sensemakr, E-value
- `validation/econometrics/` - Panel, IV, DiD, discrete choice, spatial, survival
- `validation/stats/` - 50+ statistical tests
- `validation/forecasting/` - ARIMA, MSTL, Holt-Winters
- `validation/ml/` - Clustering, PCA, t-SNE
- `validation/datasets/` - Reference datasets (Grunfeld, Longley, Iris)
- `validation/VALIDATION_STATUS.md` - Current coverage report

### Performance Benchmarks (`performance/`)

The benchmarking system measures Rust vs R performance using matching data-generating processes and matching output statistics (distribution percentiles + memory).

#### Benchmark pipeline

```
Rust benchmarks (19 files)          R benchmarks (67 scripts)
  crates/p2a-core/benches/           performance/comparisons/r_comparison/
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
  paper/code/generate_paper_figures.R + generate_paper_tables.R
         │                              │
         ▼                              ▼
  paper/figures/*.pdf           paper/tables/*.tex
```

#### Running benchmarks

```bash
# Full pipeline (validation + Rust bench + R bench + merge)
./performance/comparisons/run_all.sh

# Rust benchmarks only (outputs JSON)
cargo bench -p p2a-core --bench comprehensive_benchmarks

# R benchmarks only (outputs CSV per script)
cd performance/comparisons/r_comparison && Rscript r_benchmark_runner.R

# Merge existing results only
./performance/comparisons/run_all.sh --merge-only

# Quick mode (validation + comprehensive R benchmark only)
./performance/comparisons/run_all.sh --quick
```

#### Rust benchmark framework

`crates/p2a-core/benches/bench_utils.rs` provides:
- Distribution statistics: min, p25, median, p75, max, mean, std (matching R's `bench::mark()`)
- Memory tracking: physical_mem before/after/peak via `memory_stats` crate
- Iterations per second
- JSON serialization for merging with R results
- Reproducible data via seeded ChaCha8Rng (seed=42)

The benchmark harness is custom (not Criterion): `harness = false` in `Cargo.toml`.

Key benchmark files:
- `comprehensive_benchmarks.rs` - Master benchmark covering all method categories
- `regression_benchmarks.rs` - OLS variants, NLS, LOESS, GLS, smooth spline
- `econometrics_benchmarks.rs` - Panel, IV, DiD, synthetic control, RD, spatial
- `hypothesis_benchmarks.rs` - 20+ hypothesis tests
- `forecasting_benchmarks.rs` - ARIMA, STL, MSTL, Holt-Winters, Kalman
- `clustering_benchmarks.rs` - K-means, DBSCAN, hierarchical, PCA, t-SNE

#### R benchmark framework

`performance/comparisons/r_comparison/benchmark_*.R` (67 scripts) each benchmark the equivalent R function using `bench::mark()` with identical DGPs (same seeds, sample sizes). Orchestrated by `r_benchmark_runner.R`.

#### Merge pipeline

`performance/comparisons/r_comparison/merge_results.R`:
1. Loads all timestamped R CSVs + latest Rust JSON
2. Normalizes method names across languages (e.g., `lagsarlm` → `SAR`)
3. Matches on method + sample size
4. Computes `speedup = R_median / Rust_median`
5. Assigns module categories
6. Outputs: `comparison_speed.csv`, `comparison_memory.csv`, `validation_coverage.csv`

#### Paper exhibit generation

```bash
cd paper/code
Rscript generate_paper_figures.R    # → paper/figures/*.pdf and *.png
Rscript generate_paper_tables.R     # → paper/tables/*.tex
```

Generated artifacts:
- `paper/figures/benchmark_speedup_violin.pdf` - Violin plots by module
- `paper/figures/benchmark_boxplots.pdf` - Box plots by module
- `paper/figures/benchmark_histogram.pdf` - Speedup distribution histogram
- `paper/figures/benchmark_memory.pdf` - Memory comparison
- `paper/tables/tab_speedup_by_module.tex` - Module-level speedup summary
- `paper/tables/tab_benchmark_summary.tex` - Representative method benchmarks

#### Results directory

```
performance/comparisons/r_comparison/results/
├── comparison_speed.csv          # Merged speed comparison (tracked in git)
├── comparison_memory.csv         # Merged memory comparison (tracked in git)
├── validation_coverage.csv       # Method coverage matrix (tracked in git)
└── r_*_2026*.csv / rust_*.json   # Raw timestamped results (gitignored)
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
6. **ARM64 builds**: On aarch64-linux, debug builds require `opt-level = 1` due to relocation range limits with large binaries. This is configured in `.cargo/config.toml`. Release builds work without modification.
7. **Disk space**: The `target/` directory can grow large. Use `cargo clean` to reclaim space.

## Key Files

**Core Implementation:**
- `crates/p2a-core/src/regression/ols.rs` - OLS with robust SEs, HAC, bootstrap
- `crates/p2a-core/src/linalg/matrix_ops.rs` - Linear algebra primitives
- `crates/p2a-core/src/traits/estimator.rs` - LinearEstimator trait

**Major Econometrics:**
- `crates/p2a-core/src/econometrics/panel.rs` - Panel data (FE, RE, GMM)
- `crates/p2a-core/src/econometrics/discrete.rs` - All discrete choice models
- `crates/p2a-core/src/econometrics/staggered_did.rs` - Callaway-Sant'Anna DiD
- `crates/p2a-core/src/econometrics/synth.rs` - Synthetic control methods
- `crates/p2a-core/src/econometrics/tmle.rs` - TMLE family (tmle, ctmle, ltmle)
- `crates/p2a-core/src/econometrics/spatial.rs` - Spatial econometrics

**Statistics:**
- `crates/p2a-core/src/stats/mod.rs` - All 50+ statistical tests exported
- `crates/p2a-core/src/stats/robust.rs` - Robust statistics (IQR, MAD, ECDF)
- `crates/p2a-core/src/stats/power.rs` - Power analysis

**Forecasting:**
- `crates/p2a-core/src/forecasting/mod.rs` - All forecasting methods exported
- `crates/p2a-core/src/forecasting/kalman.rs` - State-space models
- `crates/p2a-core/src/forecasting/garch.rs` - Volatility modeling

**MCP Server:**
- `crates/p2a-mcp/src/server.rs` - AnalyticsServer struct and router composition
- `crates/p2a-mcp/src/tools/handlers/` - Tool implementations (257 tools across 17 modules)
- `crates/p2a-mcp/src/tools/requests/` - Request type definitions
- `crates/p2a-mcp/src/transport/http.rs` - HTTP transport with SSE streaming
- `crates/p2a-mcp/src/db/` - SurrealDB persistence layer

**Dioxus App:**
- `crates/p2a-dioxus/src/components/chat_panel.rs` - Main chat interface
- `crates/p2a-dioxus/src/state/` - State management (chat, session, settings)
- `crates/p2a-dioxus/src/api/sse.rs` - SSE streaming for chat

**Validation:**
- `validation/README.md` - Validation framework overview and method index
- `validation/VALIDATION_STATUS.md` - Current validation coverage by category
- `validation/run_validation.sh` - Master validation runner (Rust + R)
- `validation/[category]/method.md` - Per-method validation documents with R code and expected values

**GPU Acceleration:**
- `crates/p2a-core/src/linalg/gpu/` - GPU module (cuBLAS + cuSOLVER wrappers)
- `crates/p2a-core/src/linalg/gpu/dispatch.rs` - Calibrated dispatch thresholds
- `crates/p2a-core/benches/gpu_benchmarks.rs` - GPU vs CPU benchmarks (95 configurations)
- `performance/reports/gpu_performance.md` - GPU benchmark results and threshold rationale

**Benchmarking:**
- `crates/p2a-core/benches/bench_utils.rs` - Custom benchmark runner (distribution stats + memory)
- `crates/p2a-core/benches/comprehensive_benchmarks.rs` - Master Rust benchmark (all methods)
- `performance/comparisons/run_all.sh` - Full pipeline orchestration script
- `performance/comparisons/r_comparison/r_benchmark_runner.R` - R benchmark orchestrator
- `performance/comparisons/r_comparison/merge_results.R` - Merge Rust JSON + R CSV → comparison CSVs
- `performance/comparisons/r_comparison/results/comparison_speed.csv` - Merged speed comparison
- `performance/comparisons/r_comparison/results/comparison_memory.csv` - Merged memory comparison

**Paper Exhibits:**
- `paper/code/generate_paper_figures.R` - Generate all paper figures from comparison CSVs
- `paper/code/generate_paper_tables.R` - Generate all paper LaTeX tables from comparison CSVs
- `paper/figures/` - Generated benchmark figures (PDF + PNG)
- `paper/tables/` - Generated LaTeX tables

**Documentation:**
- `DEVELOPMENT_REPORT.md` - Detailed development history and current status
- `docs/guides/TESTING.md` - Test runtime expectations, validation framework
- `docs/guides/DATA_SECURITY.md` - Data write locations, offline capability
- `docs/security/PROMPT_INJECTION.md` - MCP security considerations

**Export Module:**
- `crates/p2a-core/src/export/latex.rs` - LaTeX table builders
- `crates/p2a-core/src/export/csv.rs` - CsvExport trait implementations
- `crates/p2a-core/src/export/html.rs` - Self-contained HTML export
- `crates/p2a-core/src/export/markdown.rs` - Markdown table builders
