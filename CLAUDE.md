# Development Guidance for prompt2analytics

This file provides context and guidance for AI assistants working on this codebase.

## Project Overview

prompt2analytics is a Rust-based analytics toolkit that exposes econometrics, ML, and visualization capabilities through the Model Context Protocol (MCP). It consists of three crates:

- **p2a-core**: Core analytics library (all algorithms)
- **p2a-mcp**: MCP server exposing 55 tools
- **p2a-desktop**: Tauri desktop application with LLM integration

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
│   └── timeseries.rs   # VAR, VARMA, VECM, IRF
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

### Session State

The MCP server maintains a `DatasetStore` for loaded datasets:
```rust
let datasets = self.datasets.lock().await;
let dataset = datasets.get(&request.dataset)
    .ok_or_else(|| McpError::invalid_request("Dataset not found", None))?;
```

## Desktop Application (p2a-desktop)

### Architecture

- **Backend**: Rust/Tauri spawns p2a-mcp as subprocess
- **Frontend**: SvelteKit with Svelte 5 runes
- **Communication**: JSON-RPC 2.0 over stdin/stdout

### LLM Integration

Three providers implemented:
- `OllamaProvider` - Local Ollama server
- `AnthropicProvider` - Claude API
- `OpenAIProvider` - GPT models

All implement the `LlmProvider` trait with streaming support.

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
2. **polars version**: Using 0.46; `is_numeric()` was removed, use custom dtype checking
3. **No formula parsing**: Use column names directly, not R-style formulas
4. **Serde serialization**: Use `#[serde(skip)]` for large internal matrices in result structs
5. **Error handling**: Use `EconError` for econometric errors, `McpError` for MCP errors

## Files to Know

- `crates/p2a-core/src/regression/ols.rs` - Main OLS implementation
- `crates/p2a-core/src/linalg/matrix_ops.rs` - Core linear algebra
- `crates/p2a-core/src/traits/estimator.rs` - LinearEstimator trait
- `crates/p2a-mcp/src/server.rs` - All 55 MCP tool definitions
- `DEVELOPMENT_REPORT.md` - Detailed development history and status
