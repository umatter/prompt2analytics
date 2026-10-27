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
cargo fmt --all -- --check               # What CI gates on
cargo fmt                                # Apply formatting

# What CI (.github/workflows/ci.yml) actually runs on the free runner. Use these
# feature sets when a full build is too heavy or when reproducing a CI failure:
cargo test -p p2a-core --tests --no-default-features --features ci
cargo test -p p2a-mcp  --tests --no-default-features --features "http,llm,auth"
cargo test -p p2a-cli  --tests --no-default-features --features spectral-analysis
cargo check -p p2a-dioxus --no-default-features --features web --target wasm32-unknown-unknown

# Run CLI
cargo run -p p2a-cli -- <args>

# Run MCP server (HTTP mode for development)
cargo run -p p2a-mcp --features full -- --transport http --host 127.0.0.1 --port 8080 --cors-permissive

# Terminal chat client (requires running MCP server)
cargo run -p p2a-chat

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

On ARM64 (aarch64), debug builds need `opt-level = 1` because p2a-mcp is a large binary that exceeds ARM64's ±128MB relocation range in pure debug mode. **`.cargo/` is gitignored**, so this is not shipped with the repo: create `.cargo/config.toml` locally on an ARM64 machine with

```toml
[profile.dev]
opt-level = 1
```

- **Debug builds**: Work once the local `.cargo/config.toml` above exists
- **Release builds**: Work without modification
- **Desktop builds**: Require `libxdo-dev` for X11 automation support

## Project Overview

prompt2analytics is a Rust workspace (edition 2024, requires Rust 1.88+) exposing 270 econometrics, statistics, ML, and visualization methods through multiple interfaces:

- **p2a-core**: Core analytics library (all algorithms)
- **p2a-cli**: Command-line interface (`p2a` binary) with session recording, script export, and JSON output
- **p2a-mcp**: MCP server exposing 270 tools with LLM integration, plus an HTTP API documented in `API_ENDPOINTS.md`
- **p2a-chat**: Lightweight terminal REPL that speaks to a running `p2a-mcp` HTTP server (no direct p2a-core dependency). Reach for it when you want to exercise the MCP tool-call path end-to-end from a shell without the Dioxus UI.
- **p2a-dioxus**: Cross-platform GUI (web via WASM, desktop via native)

Out-of-scope directories (ignore when searching unless specifically asked): `archive/`, `_resources/`, `target/`, `crates/p2a-dioxus/node_modules/`.

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

`p2a-core` defaults to `full` = `visualization + forecasting + file-formats + database + spectral-analysis`.
The `ci` alias is `full` minus `database`: the `database` feature pulls in `duckdb-sys`, which compiles
~1.5 GB of native objects and dominates build time, so CI and quick local iterations drop it.
`forecasting` and `visualization` gate whole modules (`#[cfg(feature = ...)]` in `lib.rs`), so code that
touches them must be feature-gated too. `p2a-cli` re-exports `database` and `spectral-analysis` (both on by
default); `p2a-mcp` defaults to no features and needs `--features full` (or `embedded`) for the HTTP/LLM/DB
server; `p2a-dioxus` has no default and is selected per platform (`web`, `desktop`, `mobile`) by `dx`.

```bash
cargo build -p p2a-core --all-features                  # everything incl. cuda (needs CUDA toolkit)
cargo build -p p2a-core --no-default-features --features ci   # fast: skips duckdb-sys
cargo build -p p2a-core --features spectral-analysis    # Spectral analysis (spectrum, periodogram)
cargo build -p p2a-core --features cuda                 # GPU acceleration
```

### GPU Acceleration (Optional)

The `cuda` feature enables transparent GPU dispatch for core linear algebra operations via cuBLAS and cuSOLVER. GPU paths are selected automatically when problem sizes exceed calibrated thresholds.

**GPU-accelerated operations** (via `linalg/gpu/` module):
- **xtx (X'X)**: 2-7x speedup for k>=30 via cuBLAS DGEMM
- **PCA**: 4-13x speedup via GPU covariance eigendecomposition
- **K-means distances**: 2-3x speedup for d>=20 via DGEMM reformulation
- **matmul**: 1.4-2.4x speedup for near-square matrices
- **Sandwich estimators**: 2-3x speedup for k>=50 (HC0-HC3, HAC)
- **cholesky_inverse**: CPU passthrough (no cuSOLVER path implemented); the O(k^3) inverse for small k is not worth a device round-trip

**Not GPU-accelerated** (CPU always faster):
- xty (X'y): DGEMV is bandwidth-bound; CPU OpenBLAS is faster
- Small matrices (k<30, n<5K): Transfer overhead exceeds compute savings
- Tall-skinny matmul: CPU BLAS optimized for these shapes

Dispatch thresholds are configurable via environment variables (`P2A_GPU_XTX_MIN_NKK`, etc.) and calibrated on DGX Spark. Threshold rationale is documented in `crates/p2a-core/src/linalg/gpu/dispatch.rs`.

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
├── resource.rs            # try_* fallible allocation, alloc ceiling, CancelToken/Budget, checkpoint()
├── linalg/
│   ├── matrix_ops.rs      # xtx, xty, safe_inverse, cholesky (via faer), GPU dispatch
│   └── gpu/               # Optional CUDA acceleration (#[cfg(feature = "cuda")])
│       ├── mod.rs          # GpuContext singleton (OnceLock)
│       ├── context.rs      # CUDA device init, capability detection
│       ├── memory.rs       # ndarray <-> CudaSlice (row-major handling)
│       ├── dispatch.rs     # Size thresholds (env-configurable)
│       ├── blas.rs         # cuBLAS: xtx (DSYRK→DGEMM), xty (DGEMV), matmul (DGEMM)
│       ├── solver.rs       # cuSOLVER: sandwich_meat (cholesky_inverse is a CPU passthrough)
│       └── kernels.rs      # Custom: pairwise_distances (K-means)
│   └── design.rs          # DesignMatrix, demeaning functions
├── traits/
│   └── estimator.rs       # LinearEstimator trait, SignificanceLevel, p-value helpers
│
├── diagnostics/           # Identification warnings (non-blocking) for IV, DiD, matching/IPW, RD
│                          #   design: docs/design/IDENTIFICATION_WARNINGS.md
├── spatial/               # Spatial *infrastructure*: neighbors, weights matrices, Moran/Geary
│                          #   (the SAR/SEM/SAC estimators themselves live in econometrics/spatial.rs)
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
├── ml/                    # Machine learning (35+ files)
│   ├── clustering.rs      # K-means (k-means++), DBSCAN, Hierarchical (Ward, single, complete, average)
│   ├── reduction.rs       # PCA (via SVD), t-SNE
│   ├── trees.rs           # Random Forest (CART)
│   ├── svm.rs             # Linear SVM (SMO)
│   ├── gbm.rs, xgboost.rs, lightgbm.rs, adaboost.rs, mboost.rs   # Boosting family
│   ├── bart.rs, bart_causal.rs, causal_forest.rs                 # Bayesian trees, heterogeneous effects
│   ├── shap.rs, pdp.rs, ice.rs, interpretability.rs              # Model interpretation
│   └── *_fast.rs          # Optimized variants of svm/xgboost/mboost/evaluation kept alongside originals
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

fn my_function(lag: usize, n: usize) -> EconResult<MyResult> {
    if lag >= n {
        return Err(EconError::InvalidSpecification {
            message: format!("lag must be below the number of observations ({n}), got {lag}"),
        });
    }
    // ...
}
```

Common error variants (field names as in `errors.rs`):
- `EconError::InvalidSpecification { message }` - a parameter the data cannot support; name the bound
- `EconError::ColumnNotFound { column, available }` - Missing column
- `EconError::SingularMatrix { context, suggestion }` - Non-invertible matrix
- `EconError::InsufficientData { required, provided, context }` - Not enough observations
- `EconError::ConvergenceFailure { iterations, last_change, suggestion }` - Optimization didn't converge
- `EconError::Computation(String)` - a numerical step failed
- `EconError::ResourceLimit(String)` - an impossible or over-ceiling allocation (`resource::try_*`)
- `EconError::Cancelled(String)` - the call's budget was cancelled or its deadline passed (`resource::checkpoint()`)

Modules with their own error type (`VisualizationError`, `MungeError`, `GenerationError`,
`DatabaseError`, `SasError`) map resource errors into their invalid-data variant with `map_err`.

## Common Patterns

### Resource guards (every entry point)

`p2a_core::resource` (`crates/p2a-core/src/resource.rs`) is how core code stays safe on any input,
whichever crate calls it. Core never knows about "hosted"; the server's profile only sets the
allocation ceiling and the deadline.
- Validate parameters against the data (lags, bandwidths, folds, orders below what `n` supports;
  tolerances finite and positive) and return `InvalidSpecification` naming the bound.
- Buffers sized by a caller (or by `n * n`) go through `resource::try_vec`/`try_with_capacity`/
  `try_zeros1`/`try_zeros2`, with size arithmetic through `checked_len`.
- Call `resource::checkpoint()?` once per outer iteration of every user-sized loop (replications,
  trees, permutations, draws, folds, grid points, optimiser iterations). Inside rayon, capture
  `resource::current_budget()` before the parallel iterator and call `.check()` in the closure.
- Derive each replicate's RNG from `(seed, index)` instead of materialising all index vectors.
- Filter NaN/Inf before value-driven loops and sort with `f64::total_cmp`.
- Replace a quadratic or cubic algorithm where a standard efficient one exists, with an equivalence
  test against a naive reference on noisy data.
- Method-level caps (outcome categories, `trials`, `committees`, RD order, tree depth 512, ...) hold
  in every mode; `docs/guides/DEPLOYMENT.md`, "Core invariants", lists them.

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

The MCP server exposes 270 tools organized into modular handler files. For the HTTP API surface (sessions, datasets, tools, LLM chat, SSE streaming, conversation persistence), see `API_ENDPOINTS.md` at the repo root — that is the authoritative reference.

```
crates/p2a-mcp/src/
├── server.rs              # AnalyticsServer struct + router composition; call_tool_with_session runs the
│                          #   request limits before deserialising (stdio: the ServerHandler::call_tool impl)
├── config.rs              # clap CLI args, each with a P2A_* env var; DeploymentMode/ResourceProfile are
│                          #   decided here and nowhere else (knob list: docs/guides/DEPLOYMENT.md,
│                          #   "Configuration reference")
├── exec.rs                # ToolGovernor (slots, deadline, RSS admission, quotas, panics, audit),
│                          #   ToolCallError (status/code/Retry-After), ExecConfig, StuckWatchdog
├── path_jail.rs           # validate_data_path() / validate_data_path_for_write(): every user-supplied
│                          #   path MUST go through these; write_atomically() for writes; display_path()
│                          #   for messages; per-session scope (enter_session_scope, remove_session_dir)
├── audit.rs               # Per-tool-call audit log (sanitized args, outcome, sizes; 0600, rotation)
├── session.rs             # The one registry of live sessions (TTL, LRU eviction, per-session datasets)
├── persistent_session.rs  # SurrealDB-backed conversations over that registry (feature `db`)
├── llm/                   # Provider abstraction: anthropic.rs, openai.rs, ollama.rs; tools.rs holds the
│                          #   *curated* LLM-facing tool subset, context.rs/tokens.rs manage the budget;
│                          #   egress.rs: host allowlist, resolve-check-pin resolver, no redirects
├── transport/             # http.rs (shared types, build_app, bind/serve, /health), middleware.rs (CORS,
│                          #   token, body limit, Host check), routes_sessions.rs, routes_tools.rs,
│                          #   llm_routes.rs (chat, SSE, tool executors), conversation.rs, websocket.rs
├── tools/
│   ├── mod.rs             # Re-exports
│   ├── registry.rs        # Tool metadata for documentation
│   ├── limits/            # Request limits: mod.rs (Limits, hosted()/off(), overrides, LimitViolation),
│   │                      #   rules.rs (PARAM_RULES per request field, ROW_RULES per tool)
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

4. Classify every integer, float and array field of the request struct, and the numeric elements
   of its arrays (at `a[]`), in `tools/limits/rules.rs`:
   a `Max(Family)`, `ArrayLen`, `ArrayElemMax`, `UniqueList`, `SquareLen` or `Tolerance` rule, or
   `Exempt("reason")` when the field is not a cost parameter. The exhaustiveness test
   (`every_cost_property_of_every_tool_is_classified`, built with the `http` feature, so CI runs it)
   fails otherwise.

5. Add a `ROW_RULES` entry when the method needs O(n²) memory or O(n²) or O(n³) time in the
   dataset's rows.

HTTP-side code never calls `call_tool_with_session` directly: it goes through `ToolGovernor::call`
(or `ToolGovernor::run_blocking` for non-tool blocking work), which is where deadlines, slots,
quotas, session-scoped paths and the audit entry come from.

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
    + Self::cleaning_router()
    + Self::search_router();
```

`search_router` (`handlers/search.rs`) is a meta-layer, not an analytics category: it exposes
`search_tools`, `list_tool_categories`, and `tool_info` so an LLM can discover any of the 270 tools by natural-language query.
It exists because most LLM APIs cap a request at 128 tool definitions, so `llm/tools.rs` sends only a
curated subset and relies on search for the rest. A new tool is reachable through search automatically
once its handler is registered; if it should be in the always-visible subset, add it to `llm/tools.rs` too.
The server's own chat loop does not offer the search meta-tools to its model (they are in `INTERNAL_TOOLS`,
which the loop refuses along with unknown names); registered tools outside the subset stay callable by name.

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

### Rust Benchmarks (`crates/p2a-core/benches/`)

The Rust benchmarks emit JSON with distribution statistics formatted to match
R's `bench::mark()`, for optional cross-language (R vs Rust) comparison.

```bash
# Rust benchmarks (outputs rust_*.json to performance/results/)
cargo bench -p p2a-core --bench comprehensive_benchmarks
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

## Companion paper repo (`../prompt2analytics-paper`)

The JSS manuscript ("Chat-First Data Analytics") lives in a **separate, private** sibling checkout,
`~/tools/prompt2analytics-paper`, split out of this repo in June 2026. It has its own `CLAUDE.md`
(paper build, exhibit pipeline, LLM evaluation harness); read that before touching paper files.
Division of labour: **validation stays with the code it certifies** (this repo: `validation/`,
`test_validate_*` tests, `crates/p2a-core/benches/`); **anything that produces a paper exhibit and its
inputs is paper-side** (R benchmark scripts, comparison CSVs, e2e evaluation results, figures, tables).
Never commit paper material here; `p2a-paper.zip` and similar strays do not belong in either tree.

`SOFTWARE_VERSION` in the paper repo pins the software commit the paper describes (currently tag
`v0.1.0`, results produced at `d6c6d2d`). Bump it whenever benchmarks, validation coverage, or
evaluation results are regenerated against newer software.

### What in this repo the paper depends on

Changing any of these silently changes or breaks a paper exhibit; check the paper side when you do:

| This repo | Paper consumer | What breaks |
|---|---|---|
| `bench_utils.rs` JSON schema and method names emitted by `comprehensive_benchmarks`, `unified_benchmarks`, `benchmark_n10000` | `performance/results/rust_*.json` → `merge_results.R` (name normalization across R/Rust) | Renaming a benchmark or a JSON field drops the method from the speed/memory comparison |
| MCP **tool names** (270 handlers) | `paper/code/e2e_eval/test_cases.json` expects exact names for the 96 single-turn prompts (`regression_ols`, `panel_fixed_effects`, `rd_estimate`, `ts_arima_fit`, ...); `rescore.py` scores them with an alias table | Renaming a tool turns every single-turn prompt targeting it into a scored miss. Multi-turn (MT1–MT8) scoring only checks that *some* tool was called, so names there are documentation |
| HTTP API contract in `API_ENDPOINTS.md` (`/api/sessions`, `/api/tools/load_dataset`, `/api/llm/chat/stream`, `/health`) | `run_evaluation.py` (backend URL from `P2A_BASE_URL`, default `localhost:8081`, the Dioxus dev-proxy port used for the paper runs) | Endpoint or SSE event changes break the eval harness |
| `p2a` **text output format** for `data describe/head`, `reg ols` (text and `--format json`), `panel fe/re/hausman/hdfe`, diagnostics | `paper/code/generate_outputs.sh` → `paper/generated/cli_*.tex`, included verbatim; finds the binary via `SW_REPO`/`P2A`/`P2A_DATA_DIR` (default sibling checkout, release build) | Output formatter changes alter exhibits. `make jss` there re-runs the capture whenever the script is newer than the `.tex` files, so a rebuild after touching the script silently re-captures from whatever `p2a` you have built |
| `validation/datasets/` (Grunfeld etc.) | CLI captures above, `rust_validation/scripts/run_validation.sh`, several R benchmark DGPs | Moving a dataset breaks `make generate` |
| The LLM system prompt and dataset-context format (`llm/context.rs`, `llm/tools.rs`) | e2e evaluation accuracy (`run_evaluation.py` results) | Tier 0 bounded and delimited the dataset block (JSON-quoted names and values, byte and column caps), so re-running the evaluation against a newer `SOFTWARE_VERSION` can shift the accuracy numbers. Tool names and SSE events are unchanged |
| Counts and claims stated as prose in `paper/sections/*.tex`: "over 200 methods", test-function and validation-test counts, ~62% line coverage, version, Rust version, and the description of what CI runs | none (hand-written) | They drift silently; re-derive them when re-pinning `SOFTWARE_VERSION` |

### Paper-side code that points back here

- `paper/code/mc_validation/` and `paper/code/rust_validation/rust_runner/` are nested Cargo crates with a
  **path dependency on this checkout** (`../prompt2analytics/crates/p2a-core` relative to the paper root).
  They build against whatever is checked out here, not the pinned tag, so check out `SOFTWARE_VERSION`'s
  commit (or a worktree of it) before regenerating paper numbers. Their `Cargo.lock` files are gitignored.
- `paper/Makefile` `validate*`/`benchmark*` targets `cd` into this repo via `SW_REPO` (default sibling);
  `paper/Dockerfile.reproduce` clones this repo from GitHub at `--build-arg P2A_REF` (default `v0.1.0`).
- `build_replication_archive.sh` runs `git -C ../prompt2analytics archive` at the pinned commit, so the
  sibling checkout must contain that commit (fetch tags before packaging).
- The `/implement_metrics`, `/implement_next`, and `/validate-method` commands and the
  `econometrics-implementer` agent **write R benchmark scripts into**
  `../prompt2analytics-paper/performance/comparisons/r_comparison/benchmark_<method>.R` and update
  `../prompt2analytics-paper/performance/reports/`. Adding a method therefore produces commits in both
  repos; if the paper checkout is absent those steps are skipped, not failed.

### Cross-repo workflow when software and paper change together

1. Here: implement, `cargo test -p p2a-core -- test_validate`, then `cargo bench -p p2a-core --bench comprehensive_benchmarks` → `rust_*.json`.
2. Copy the JSON from `crates/p2a-core/performance/results/` (the bench writes relative to the crate dir; gitignored here) into `../prompt2analytics-paper/performance/results/`. `make -C paper benchmark-full` in the paper repo does steps 1–3.
3. There: `Rscript r_benchmark_runner.R` and `Rscript merge_results.R` in `performance/comparisons/r_comparison/`.
4. There: regenerate exhibits (`cd paper && make exhibits`, needs a built `p2a` via `P2A=`), then `make jss`.
5. There: update `SOFTWARE_VERSION` to the software commit (tag it here first if this is a release).
6. Commit in each repo separately; the software repo is public, the paper repo is private.

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

### Agents and Tooling

- `.claude/agents/econometrics-implementer.md` - Expert Rust econometrics implementer subagent (use for new method implementation)
- `.claude/tooling/validation/` - Validation helper scripts used by `/validate-method`

## Testing Guidelines

### What CI checks (and what it deliberately does not)

`.github/workflows/ci.yml` runs on the free GitHub runner and is kept cheap on purpose:

- **Gating**: `cargo fmt --all -- --check`; `cargo test --tests` for p2a-core (`ci` feature), p2a-mcp
  (`http,llm,auth`), p2a-cli (`spectral-analysis`); doctests for p2a-core (`ci`) and p2a-mcp
  (`http,llm,auth`) in their own step; plus a wasm32 `cargo check` + host tests for p2a-dioxus (`web`).
- **Doctests are gated** and, as of Oct 2026, p2a-core runs 81 passed / 0 failed / 190 ignored. A doc example
  that stops compiling is a build failure, so keep `use p2a_core::...` paths in doc comments pointing at
  what `mod.rs` actually re-exports. The 190 examples fenced as ignore are NOT compiled and can still rot silently.
- **p2a-mcp integration tests** are one file per topic in `crates/p2a-mcp/tests/` (`governor.rs`,
  `limits_e2e.rs`, `sessions.rs`, `llm_egress.rs`, `isolation.rs`, `integration_tests.rs`); their feature
  gates and commands are in `docs/guides/TESTING.md`. `sessions.rs` (needs `db`) and the `/ws` test (needs
  `websocket`) are outside CI: run `cargo test -p p2a-mcp --features full` (or
  `--no-default-features --features "http,llm,auth,db"`) when touching sessions, the DB or the websocket.
- **Do not call `limits::install` or `resource::set_alloc_ceiling_bytes` per test** (process-global, and tests
  in one binary run in parallel). Select limits with the thread-local `limits::with_limits` and
  `resource::with_alloc_ceiling`. The exceptions: a test binary whose tests all share one configuration installs
  it once inside a `OnceLock` (`crates/p2a-mcp/tests/isolation.rs`), and p2a-core's `global_ceiling_round_trips`
  sets only a harmless ceiling (`usize::MAX - 1`) and restores `None`. `transport::http::serve` and
  `start_http_transport` install the limits of the configuration they serve and refuse a start whose limits or
  process policies differ from those of a server still running in the process (held until its last connection or
  websocket session closes), so a test serves a configuration whose resource profile is off, with no overrides and
  the default policies, or runs in a binary whose servers share one config (`crates/p2a-mcp/tests/serve_limits.rs`).
- **Clippy runs but does not gate**: p2a-core has a pre-existing backlog of clippy violations, and
  `Cargo.toml` `[workspace.lints]` allows a long list of lints (dead_code, unused_imports, too_many_arguments, ...).
  Do not "fix" the allow-list to make a build green; do keep new code warning-free.
- **Excluded from CI**: release builds, the `database` feature (duckdb-sys), `cuda`, the Dioxus desktop
  target (WebKit/GTK), and `p2a-chat` tests (it has no test surface). These are exercised locally or on
  the self-hosted runner. A weekly `audit.yml` runs `cargo audit`; its RUSTSEC exemptions live as flags
  in that file (not `.cargo/audit.toml`, which is gitignored and would never reach CI).
- Only one test is `#[ignore]`d (an HDBSCAN benchmark in `ml/advanced_clustering_mod/advanced.rs`);
  run it with `--release -- --ignored`. `docs/guides/TESTING.md` lists expected runtimes and memory needs.
- **`tests/benchmark_ml_rust.rs` and `tests/benchmark_fast_implementations.rs` are benchmarks, not
  correctness tests**, and `cargo test -p p2a-core` runs them. In a debug build the ML one has been
  observed to burn 16+ minutes of CPU. Use `--lib` for a fast correctness loop.
- **Wall-clock assertions**: four exist (`test_fast_mboost_componentwise_speed` and the panel scaling
  tests `test_random_effects_scaling_many_entities`, `test_panel_gls_scaling_many_entities`,
  `test_gmm_scaling_many_entities`), and all of them are release-only (`cfg!(debug_assertions)` guards).
  Do not add ungated timing assertions — they fail on loaded or slower machines and make `cargo test`
  machine-dependent, which the paper's reviewers will run.

### Two traps on a dev box: disk and rustdoc

Both produce failures that look like broken code and are not.

1. **A full disk makes `ld` die with SIGBUS**, which rustdoc reports as `Couldn't compile the test` —
   indistinguishable from a broken doc example. `target/debug` has been measured at 127 GB. **Run
   `df -h /` before believing any link failure.** Mitigations: `rm -rf target/debug/incremental` (pure
   cache) and `TMPDIR=/dev/shm` so linker temporaries never touch the full disk. A doctest run that
   showed 53 spurious failures in 418 s went to 0 failures in 94 s after doing both.
2. **A distro `rustdoc` may not match its `libLLVM`**, so `cargo test` dies at the doctest step with
   `error while loading shared libraries: libLLVM.so...`. Run doctests through the rustup toolchain,
   setting `RUSTC`/`RUSTDOC` explicitly (calling that cargo directly still resolves `rustdoc` from PATH):

```bash
TC=~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin
TMPDIR=/dev/shm RUSTC=$TC/rustc RUSTDOC=$TC/rustdoc CARGO_TARGET_DIR=/tmp/p2a-doc \
  $TC/cargo test -p p2a-core --doc --no-default-features --features ci
```

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
6. **ARM64 builds**: On aarch64-linux, debug builds require `opt-level = 1` due to relocation range limits with large binaries. Create `.cargo/config.toml` locally (see ARM64 Build Notes); it is gitignored. Release builds work without modification.
8. **Filesystem access in p2a-mcp**: any tool or HTTP handler that takes a path must call `path_jail::validate_data_path` first (writers: `validate_data_path_for_write` + `write_atomically`), and messages show `display_path` only, never absolute paths. Paths are session-scoped automatically, because the governor enters `path_jail::enter_session_scope` on the blocking thread that runs the handler; handlers must therefore not spawn threads or tasks that validate paths (those have no scope and are refused). The hosted demo (p2a.qamelab.org) runs in open public mode with the hosted resource profile; see `docs/guides/DEPLOYMENT.md` for the threat model before changing anything on the HTTP surface.
9. **Native Dioxus builds embed the backend**: `desktop`/`mobile` features depend on `p2a-mcp` and start an in-process server (`start_embedded_server` in `p2a-dioxus/src/main.rs`); the web build talks to `P2A_BACKEND_URL` baked in at build time. A change to `p2a-mcp`'s public API can break the desktop build even when the web build passes.
7. **Disk space**: The `target/` directory can grow large. Use `cargo clean` to reclaim space.
10. **Deployment mode and resource profile are decided only in `config.rs`** (`DeploymentMode`, `ResourceProfile`). Local, stdio, desktop and the paper's evaluation runs are `Off`: no caps, no deadline, no per-session directories. `P2A_LIMITS=hosted` turns the server layer on anywhere, and open public mode always has it. Never key behaviour to the bind address.

## Key Files

**Core Implementation:**
- `crates/p2a-core/src/regression/ols.rs` - OLS with robust SEs, HAC, bootstrap
- `crates/p2a-core/src/linalg/matrix_ops.rs` - Linear algebra primitives
- `crates/p2a-core/src/traits/estimator.rs` - LinearEstimator trait
- `crates/p2a-core/src/resource.rs` - Resource guards: fallible allocation, alloc ceiling, budgets, `checkpoint()`

**Major Econometrics:**
- `crates/p2a-core/src/econometrics/panel/` - Panel data (FE, RE, GLS, GMM)
- `crates/p2a-core/src/econometrics/discrete/` - All discrete choice models
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
- `crates/p2a-mcp/src/tools/handlers/` - Tool implementations (270 tools across 17 modules)
- `crates/p2a-mcp/src/tools/requests/` - Request type definitions
- `crates/p2a-mcp/src/exec.rs` - Tool governor: slots, deadlines, memory admission, quotas, error codes
- `crates/p2a-mcp/src/tools/limits/` - Request-limit catalogue (`Limits`, `PARAM_RULES`, `ROW_RULES`)
- `crates/p2a-mcp/src/transport/http.rs` - Shared HTTP types, router composition, startup (`build_app`, `bind`, `serve`), `/health`
- `crates/p2a-mcp/src/transport/llm_routes.rs` - LLM chat routes, SSE streaming, tool executors
- `crates/p2a-mcp/src/llm/egress.rs` - LLM host allowlist and resolve-check-pin resolver
- `crates/p2a-mcp/src/db/` - SurrealDB persistence layer
- `crates/p2a-mcp/src/path_jail.rs` - Filesystem path validation (P2A_DATA_ROOT, per-session directories)
- `crates/p2a-mcp/src/tools/handlers/search.rs` - Tool search meta-tools (128-tool API limit workaround)
- `crates/p2a-mcp/src/llm/tools.rs` - Curated LLM-facing tool subset and few-shot examples

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

**Benchmarking:**
- `crates/p2a-core/benches/bench_utils.rs` - Custom benchmark runner (distribution stats + memory)
- `crates/p2a-core/benches/comprehensive_benchmarks.rs` - Master Rust benchmark (all methods)

**Documentation:**
- `API_ENDPOINTS.md` - Complete HTTP API reference (sessions, datasets, tools, LLM chat, SSE)
- `docs/guides/TESTING.md` - Test runtime expectations, validation framework
- `docs/guides/DATA_SECURITY.md` - Data write locations, offline capability
- `docs/guides/DEPLOYMENT.md` - BYOK web deployment, deployment modes, in-process controls, every `P2A_*` knob and limit key, release and redeploy steps
- `docs/guides/CLI_REFERENCE.md` - Full `p2a` command reference; `docs/cookbook.md` - worked workflows
- `docs/design/IDENTIFICATION_WARNINGS.md` - Design of the `diagnostics/` identification-warning system
- `docs/security/PROMPT_INJECTION.md` - MCP security considerations
- `.github/workflows/ci.yml` - Source of truth for which feature sets are tested on every push

**Export Module:**
- `crates/p2a-core/src/export/latex.rs` - LaTeX table builders
- `crates/p2a-core/src/export/csv.rs` - CsvExport trait implementations
- `crates/p2a-core/src/export/html.rs` - Self-contained HTML export
- `crates/p2a-core/src/export/markdown.rs` - Markdown table builders
