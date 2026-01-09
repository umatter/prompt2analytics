---
name: rust-econometrics-patterns
description: Rust implementation patterns for econometrics in p2a-core. Use when implementing new statistical methods.
---

# p2a-core Implementation Patterns

## Before You Start: Check Existing Code

**ALWAYS check for existing implementations before writing new code.**

### Search for Existing Methods

```bash
# Search for method name/abbreviation
Grep: "GLS\|Generalized Least Squares" in crates/p2a-core/src/

# Check what's already implemented
Read: crates/p2a-core/src/regression/mod.rs
Read: crates/p2a-core/src/econometrics/mod.rs
```

### Reusable Components

Before implementing, check if these components can be reused:

| Component | Location | What it provides |
|-----------|----------|------------------|
| X'X, X'y | `linalg/matrix_ops.rs` | Gram matrix, cross-product |
| Matrix inverse | `linalg/matrix_ops.rs` | `safe_inverse()` via Cholesky |
| Design matrix | `linalg/design.rs` | `DesignMatrix`, demeaning |
| Robust SEs | `regression/ols.rs` | HC0-HC3 variance estimators |
| Clustered SEs | `regression/ols.rs` | One-way, two-way clustering |
| P-value helpers | `traits/estimator.rs` | t, F, chi-squared p-values |
| Panel demeaning | `econometrics/panel.rs` | Entity/time demeaning |
| MLE optimization | `econometrics/discrete.rs` | Newton-Raphson pattern |

**If functionality already exists, REUSE it. Don't duplicate code.**

---

## Module Organization

New methods go in the appropriate module under `crates/p2a-core/src/`:

| Module | Methods |
|--------|---------|
| `regression/ols.rs` | OLS, robust SEs, clustered SEs |
| `regression/diagnostics.rs` | JB, BP, DW, VIF, condition number |
| `econometrics/panel.rs` | Fixed Effects, Random Effects, Hausman |
| `econometrics/iv.rs` | 2SLS with first-stage diagnostics |
| `econometrics/did.rs` | Difference-in-Differences |
| `econometrics/discrete.rs` | Logit, Probit (Newton-Raphson MLE) |
| `econometrics/timeseries.rs` | VAR, VARMA, VECM, IRF |

## API Pattern (Column-Based)

All regression functions use explicit column names:

```rust
pub fn run_new_method(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    // method-specific parameters
) -> EconResult<NewMethodResult> {
    // Implementation
}
```

**NOT** formula-based like R:
```rust
// This is NOT how it works - DO NOT USE
run_ols("y ~ x1 + x2")
```

## Using LinearEstimator Trait

If the method produces coefficients and standard errors, implement `LinearEstimator`:

```rust
use crate::traits::estimator::LinearEstimator;

impl LinearEstimator for NewMethodResult {
    fn coefficients(&self) -> &Array1<f64> { &self.coefficients }
    fn std_errors(&self) -> &Array1<f64> { &self.std_errors }
    fn t_values(&self) -> Array1<f64> {
        &self.coefficients / &self.std_errors
    }
    fn p_values(&self) -> Array1<f64> {
        self.t_values().mapv(|t| t_test_p_value(t, self.df()))
    }
    fn residuals(&self) -> Array1<f64> { self.residuals.clone() }
    fn n_obs(&self) -> usize { self.n_obs }
    fn df(&self) -> usize { self.n_obs - self.coefficients.len() }
    // ... implement all required methods
}
```

## Matrix Operations

Use functions from `linalg/matrix_ops.rs`:

```rust
use crate::linalg::matrix_ops::{xtx, xty, safe_inverse, cholesky};

let xtx_matrix = xtx(&x);           // X'X
let xty_vector = xty(&x, &y);       // X'y
let inv = safe_inverse(&m)?;        // Safe matrix inverse via Cholesky
let (lower, _) = cholesky(&m)?;     // Cholesky decomposition
```

## P-Value Calculation

Use helpers from `traits/estimator.rs`:

```rust
use crate::traits::estimator::{t_test_p_value, f_test_p_value, chi_squared_p_value};

let p = t_test_p_value(t_stat, df);           // Two-tailed t-test
let p = f_test_p_value(f_stat, df1, df2);     // F-test
let p = chi_squared_p_value(chi2, df);        // Chi-squared test
```

These functions handle edge cases (NaN, Inf) gracefully.

## Design Matrix

Use `DesignMatrix` from `linalg/design.rs`:

```rust
use crate::linalg::design::DesignMatrix;

let dm = DesignMatrix::from_dataset(dataset, x_cols, intercept)?;
let x = dm.view();  // ArrayView2<f64>
```

## Error Handling

Use `EconError` and `EconResult<T>` from `errors.rs`:

```rust
use crate::errors::{EconError, EconResult};

fn my_function() -> EconResult<MyResult> {
    if invalid_input {
        return Err(EconError::InvalidInput("descriptive message".to_string()));
    }
    if singular_matrix {
        return Err(EconError::SingularMatrix);
    }
    // ...
    Ok(result)
}
```

Common error types:
- `EconError::InvalidInput(String)` - Bad input data
- `EconError::SingularMatrix` - Non-invertible matrix
- `EconError::ColumnNotFound(String)` - Missing column
- `EconError::InsufficientObservations` - Not enough data

## Robust Standard Errors

The `CovarianceType` enum controls variance estimation:

```rust
pub enum CovarianceType {
    Standard,  // Homoskedastic (classical)
    HC0,       // White's heteroskedasticity-consistent
    HC1,       // HC0 with small-sample correction (n/(n-k))
    HC2,       // HC1 with leverage adjustment
    HC3,       // HC2 with more aggressive correction
}
```

## Result Struct Pattern

```rust
#[derive(Debug, Clone)]
pub struct NewMethodResult {
    pub coefficients: Array1<f64>,
    pub std_errors: Array1<f64>,
    pub residuals: Array1<f64>,
    pub fitted_values: Array1<f64>,
    pub n_obs: usize,
    pub r_squared: f64,
    // Method-specific fields...

    #[serde(skip)]  // Skip large internal matrices in serialization
    pub vcov: Array2<f64>,
}
```

## Test Data Guidelines

Test data should have noise (not perfect linear relationships):

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

## Key Files to Reference

- `crates/p2a-core/src/regression/ols.rs` - OLS implementation example
- `crates/p2a-core/src/linalg/matrix_ops.rs` - Matrix utilities
- `crates/p2a-core/src/traits/estimator.rs` - LinearEstimator trait
- `crates/p2a-core/src/errors.rs` - Error types
- `crates/p2a-mcp/src/server.rs` - MCP tool definitions

## Implementation Checklist

- [ ] **Check for existing implementation first**
- [ ] **Identify reusable components** (see table above)
- [ ] Choose appropriate module location
- [ ] Design column-based API
- [ ] Implement core algorithm (**reusing existing code where possible**)
- [ ] Use EconError for error handling
- [ ] Implement LinearEstimator trait if applicable
- [ ] Write unit tests with noisy data
- [ ] Compare results with reference implementation
- [ ] Add MCP tool if user-facing
- [ ] Update documentation
