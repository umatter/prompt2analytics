# Implementation Plan: GLM with High-Dimensional Fixed Effects (FEGLM)

## Summary

Implement **FEGLM** (Fixed Effects GLM) in p2a-core, replicating the functionality of R's `alpaca::feglm()`. This extends the existing HDFE (linear) and Logit/Probit (no FE) implementations to support **generalized linear models with absorbed high-dimensional fixed effects**.

## Research Summary

### Source Package: [alpaca](https://cran.r-project.org/web/packages/alpaca/index.html)
- Author: Amrei Stammann
- Reference: [Stammann (2018) "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects"](https://arxiv.org/abs/1707.01815)

### Core Algorithm: IRLS with Weighted Demeaning

The algorithm combines:
1. **Iteratively Reweighted Least Squares (IRLS)** - Standard GLM estimation
2. **Method of Alternating Projections (MAP)** - Efficient fixed effects absorption
3. **Weighted Frisch-Waugh-Lovell theorem** - Partialing out fixed effects in weighted regression

### Mathematical Formulation

**GLM with Fixed Effects:**
```
g(μᵢ) = ηᵢ = Xᵢβ + Σⱼ Dⱼαⱼ

where:
  g(·)    = link function (logit, probit, log, etc.)
  μᵢ     = E[Yᵢ]
  Xᵢβ    = structural parameters (coefficients of interest)
  Σⱼ Dⱼαⱼ = absorbed fixed effects (nuisance parameters)
```

**IRLS Update (each iteration t):**
```
1. Compute working response: z⁽ᵗ⁾ = η⁽ᵗ⁾ + (y - μ⁽ᵗ⁾) × [∂η/∂μ]

2. Compute working weights: w⁽ᵗ⁾ = 1/Var(z) = [∂μ/∂η]² / Var(Y)

3. Weighted demean z and X by fixed effects (using MAP with weights w)

4. Solve weighted least squares: β⁽ᵗ⁺¹⁾ = (X̃'WX̃)⁻¹ X̃'Wz̃
   where X̃, z̃ are demeaned versions

5. Update linear predictor: η⁽ᵗ⁺¹⁾ = Xβ⁽ᵗ⁺¹⁾ + (fixed effects via back-substitution)

6. Check convergence: |β⁽ᵗ⁺¹⁾ - β⁽ᵗ⁾| < tolerance
```

**Working Weights by Family:**

| Family | Link | Variance V(μ) | Weight w = [∂μ/∂η]²/V(μ) |
|--------|------|---------------|--------------------------|
| Binomial | logit | μ(1-μ) | μ(1-μ) |
| Binomial | probit | μ(1-μ) | φ(η)²/[Φ(η)(1-Φ(η))] |
| Poisson | log | μ | μ |
| Gamma | log | μ² | 1/μ |
| Gaussian | identity | σ² | 1 (reduces to linear HDFE) |

## Existing Reusable Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `demean_map()` | `hdfe.rs:240` | MAP algorithm for demeaning |
| `demean_by_factor()` | `hdfe.rs:184` | Single-factor demeaning |
| `extract_factor_info()` | `hdfe.rs:98` | Extract FE groups from dataset |
| `logistic_cdf/pdf` | `traits/estimator.rs` | Logit link function |
| `normal_cdf/pdf` | `traits/estimator.rs` | Probit link function |
| `safe_inverse` | `linalg/matrix_ops.rs` | Safe matrix inversion |
| `DesignMatrix` | `linalg/design.rs` | Build design matrix from columns |
| `EconError` | `errors.rs` | Error handling |

## Implementation Plan

### Phase 3a: Core Data Structures

Create `crates/p2a-core/src/econometrics/feglm.rs`:

```rust
/// GLM family specification for FEGLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GlmFamily {
    /// Binomial with logit link: P(Y=1) = 1/(1+exp(-η))
    Logit,
    /// Binomial with probit link: P(Y=1) = Φ(η)
    Probit,
    /// Poisson with log link: E[Y] = exp(η)
    Poisson,
    /// Gaussian with identity link: E[Y] = η (reduces to linear HDFE)
    Gaussian,
}

/// Configuration for FEGLM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeglmConfig {
    /// Maximum IRLS iterations
    pub max_iter: usize,          // Default: 25
    /// Convergence tolerance for coefficients
    pub tolerance: f64,           // Default: 1e-8
    /// MAP tolerance for demeaning
    pub map_tolerance: f64,       // Default: 1e-8
    /// Maximum MAP iterations per IRLS step
    pub map_max_iter: usize,      // Default: 10000
    /// Minimum weight threshold (avoid division by zero)
    pub weight_min: f64,          // Default: 1e-10
}

/// Result from FEGLM estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeglmResult {
    pub family: GlmFamily,
    pub dep_var: String,
    pub variables: Vec<String>,
    pub fe_dimensions: Vec<String>,
    pub fe_counts: Vec<usize>,

    // Coefficients and inference
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub significance: Vec<SignificanceLevel>,

    // Fit statistics
    pub log_likelihood: f64,
    pub log_likelihood_null: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub pseudo_r_squared: f64,  // McFadden's
    pub aic: f64,
    pub bic: f64,

    // For Poisson/negative binomial
    pub dispersion: f64,

    // Convergence
    pub iterations: usize,
    pub converged: bool,
    pub final_change: f64,

    // Dimensions
    pub n_obs: usize,
    pub n_positive: usize,  // For binomial
    pub df_resid: usize,
    pub df_absorbed: usize,
}
```

### Phase 3b: Link and Variance Functions

```rust
impl GlmFamily {
    /// Link function: g(μ) → η
    pub fn link(&self, mu: f64) -> f64;

    /// Inverse link: g⁻¹(η) → μ
    pub fn inv_link(&self, eta: f64) -> f64;

    /// Derivative: ∂μ/∂η (for working weights)
    pub fn mu_eta(&self, eta: f64) -> f64;

    /// Variance function: V(μ)
    pub fn variance(&self, mu: f64) -> f64;

    /// Working weight: w = (∂μ/∂η)² / V(μ)
    pub fn working_weight(&self, eta: f64, mu: f64) -> f64;

    /// Working response: z = η + (y - μ) × (∂η/∂μ)
    pub fn working_response(&self, y: f64, eta: f64, mu: f64) -> f64;

    /// Log-likelihood contribution for observation
    pub fn log_lik(&self, y: f64, mu: f64) -> f64;

    /// Deviance contribution
    pub fn deviance(&self, y: f64, mu: f64) -> f64;
}
```

### Phase 3c: Weighted Demeaning

Extend `hdfe.rs` with weighted versions:

```rust
/// Weighted demean by a single factor (weighted group means).
fn weighted_demean_by_factor(
    data: &Array1<f64>,
    weights: &Array1<f64>,
    factor: &FactorInfo,
) -> Array1<f64>;

/// Weighted MAP demeaning for FEGLM.
fn weighted_demean_map(
    data: &Array1<f64>,
    weights: &Array1<f64>,
    factors: &[FactorInfo],
    tolerance: f64,
    max_iter: usize,
) -> (Array1<f64>, usize, f64, bool);
```

### Phase 3d: Main FEGLM Function

```rust
/// Run Generalized Linear Model with High-Dimensional Fixed Effects.
///
/// # Arguments
/// * `dataset` - Dataset containing the variables
/// * `y_col` - Name of the outcome variable
/// * `x_cols` - Names of regressor columns
/// * `fe_cols` - Names of fixed effect columns to absorb
/// * `family` - GLM family (Logit, Probit, Poisson, Gaussian)
/// * `config` - Optional configuration
///
/// # Returns
/// `FeglmResult` with coefficients, standard errors, and fit statistics.
pub fn run_feglm(
    dataset: &Dataset,
    y_col: &str,
    x_cols: &[&str],
    fe_cols: &[&str],
    family: GlmFamily,
    config: Option<FeglmConfig>,
) -> EconResult<FeglmResult>;
```

### Algorithm Pseudocode

```rust
fn run_feglm(...) -> EconResult<FeglmResult> {
    // 1. Extract y, X, and factor info
    let y = extract_column(y_col)?;
    let x = DesignMatrix::from_dataframe(dataset, x_cols, false)?; // No intercept
    let factors = fe_cols.iter().map(extract_factor_info).collect()?;

    // 2. Initialize
    let n = y.len();
    let k = x.ncols();
    let mut beta = Array1::zeros(k);
    let mut eta = Array1::zeros(n);  // Linear predictor

    // Initial guess for eta based on family
    match family {
        GlmFamily::Logit | GlmFamily::Probit => {
            let p_bar = y.mean().clamp(0.01, 0.99);
            eta = Array1::from_elem(n, family.link(p_bar));
        }
        GlmFamily::Poisson => {
            eta = y.mapv(|yi| (yi.max(0.1)).ln());
        }
        GlmFamily::Gaussian => {
            eta = y.clone();
        }
    }

    // 3. IRLS loop
    for iter in 0..config.max_iter {
        // 3a. Compute μ and working weights/response
        let mu = eta.mapv(|e| family.inv_link(e));
        let weights = eta.iter().zip(mu.iter())
            .map(|(&e, &m)| family.working_weight(e, m).max(config.weight_min))
            .collect();
        let z = y.iter().zip(eta.iter()).zip(mu.iter())
            .map(|((&yi, &ei), &mi)| family.working_response(yi, ei, mi))
            .collect();

        // 3b. Weighted demean z and X by fixed effects
        let sqrt_w = weights.mapv(|w| w.sqrt());
        let z_weighted = &z * &sqrt_w;
        let x_weighted = broadcast_mul(&x, &sqrt_w);  // Scale rows by sqrt(w)

        let (z_demeaned, _, _, _) = demean_map(&z_weighted, &factors, ...);
        let (x_demeaned, _, _, _) = demean_matrix_map(&x_weighted, &factors, ...);

        // 3c. Weighted least squares on demeaned data
        // Since we pre-scaled by sqrt(w), this is just OLS
        let xtx = xtx(&x_demeaned.view());
        let xty = xty(&x_demeaned.view(), &z_demeaned);
        let (xtx_inv, _) = safe_inverse(&xtx.view())?;
        let beta_new = xtx_inv.dot(&xty);

        // 3d. Check convergence
        let change = (&beta_new - &beta).mapv(|d| d.abs()).sum();
        if change < config.tolerance {
            converged = true;
            break;
        }
        beta = beta_new;

        // 3e. Update linear predictor
        // η = Xβ + (absorbed FE)
        // The FE are implicitly included via the demeaning
        eta = x.dot(&beta);  // Partial eta (without FE)

        // Recover FE contribution from residuals of working regression
        let fitted_demeaned = x_demeaned.dot(&beta);
        let fe_contribution = &z_weighted - &fitted_demeaned;
        // Add back group means to eta
        eta = add_back_fe_means(&eta, &fe_contribution, &factors, &weights);
    }

    // 4. Compute variance-covariance and standard errors
    // Using sandwich estimator for robustness

    // 5. Compute fit statistics (log-likelihood, deviance, etc.)

    Ok(FeglmResult { ... })
}
```

### Phase 3e: MCP Tool

Add to `crates/p2a-mcp/src/server.rs`:

```rust
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FeglmRequest {
    pub dataset: String,
    pub y: String,
    pub x: Vec<String>,
    pub fe: Vec<String>,
    pub family: String,  // "logit", "probit", "poisson", "gaussian"
    pub max_iter: Option<usize>,
    pub tolerance: Option<f64>,
}

#[tool(description = "Run Generalized Linear Model with high-dimensional fixed effects (FEGLM).
Supports Logit, Probit, Poisson families with multiple absorbed fixed effects.
Equivalent to R's alpaca::feglm().")]
async fn feglm(
    &self,
    Parameters(request): Parameters<FeglmRequest>,
) -> Result<CallToolResult, McpError>;
```

## File Structure

```
crates/p2a-core/src/econometrics/
├── mod.rs              # Add: mod feglm; and exports
├── hdfe.rs             # Extend: weighted_demean_* functions
└── feglm.rs            # NEW: ~800 lines
    ├── GlmFamily enum and methods
    ├── FeglmConfig, FeglmResult structs
    ├── run_feglm() main function
    └── Tests with R validation
```

## Test Strategy

### Unit Tests

```rust
#[test]
fn test_feglm_logit_two_way() {
    // Compare against alpaca::feglm(y ~ x1 + x2 | id + time, family=binomial())
}

#[test]
fn test_feglm_poisson_firm_year() {
    // Compare against alpaca::feglm(count ~ x | firm + year, family=poisson())
}

#[test]
fn test_feglm_matches_discrete_when_no_fe() {
    // With empty fe_cols, should match run_logit/run_probit
}

#[test]
fn test_feglm_gaussian_matches_hdfe() {
    // Gaussian family should match run_hdfe exactly
}
```

### R Validation Script

```r
# validation/scripts/feglm_validation.R
library(alpaca)

# Test 1: Logit with two-way FE
set.seed(42)
n <- 1000
data <- data.frame(
  id = factor(sample(50, n, replace=TRUE)),
  time = factor(sample(20, n, replace=TRUE)),
  x1 = rnorm(n),
  x2 = rnorm(n)
)
# Generate binary outcome with FE
id_eff <- rnorm(50)[data$id]
time_eff <- rnorm(20)[data$time]
data$y <- rbinom(n, 1, plogis(0.5*data$x1 - 0.3*data$x2 + id_eff + time_eff))

mod <- feglm(y ~ x1 + x2 | id + time, data=data, family=binomial("logit"))
summary(mod)
# Save coefficients for Rust comparison
```

## Validation Document

Create `validation/econometrics/feglm.md`:

| Test Case | R Code | Tolerance | Rust Test |
|-----------|--------|-----------|-----------|
| Logit, 2-way FE | `feglm(y ~ x | id + t, binomial())` | coef: 1e-4, SE: 1e-3 | `test_validate_feglm_logit` |
| Probit, 2-way FE | `feglm(y ~ x | id + t, binomial("probit"))` | coef: 1e-4, SE: 1e-3 | `test_validate_feglm_probit` |
| Poisson, 2-way FE | `feglm(y ~ x | id + t, poisson())` | coef: 1e-4, SE: 1e-3 | `test_validate_feglm_poisson` |

## Performance Benchmarks

Create `crates/p2a-core/benches/feglm_benchmarks.rs`:

```rust
fn feglm_logit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("feglm_logit");
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let dataset = generate_binary_panel(*size, 50, 20);
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| b.iter(|| run_feglm(&dataset, "y", &["x"], &["id", "t"], GlmFamily::Logit, None))
        );
    }
}
```

## Dependencies

No new crates required. Uses existing:
- `ndarray` 0.16 - Matrix operations
- `faer` 0.22 - Safe matrix inverse
- `statrs` 0.18 - Normal distribution for probit

## References

1. Stammann, A. (2018). ["Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects"](https://arxiv.org/abs/1707.01815). ArXiv e-prints.
2. Gaure, S. (2013). "lfe: Linear Group Fixed Effects". *The R Journal*, 5(2), 104-117.
3. McCullagh, P. & Nelder, J.A. (1989). *Generalized Linear Models*. 2nd ed. Chapman & Hall.
4. R package `alpaca`: https://cran.r-project.org/package=alpaca

## Implementation Checklist

### Core Implementation (Phase 3)
- [ ] Create `feglm.rs` with `GlmFamily` enum and methods
- [ ] Implement link/variance/weight functions for each family
- [ ] Implement weighted demeaning functions in `hdfe.rs`
- [ ] Implement `run_feglm()` with IRLS + weighted MAP
- [ ] Add `FeglmResult` Display trait implementation

### MCP Tool (Phase 3a)
- [ ] Add `FeglmRequest` struct
- [ ] Add `feglm` tool handler
- [ ] Update tool definitions list
- [ ] Add usage example to MCP_TOOL_EXAMPLES.md

### Validation Tests (Phase 4)
- [ ] Create `validation/scripts/feglm_validation.R`
- [ ] Implement `test_validate_feglm_logit_against_alpaca`
- [ ] Implement `test_validate_feglm_probit_against_alpaca`
- [ ] Implement `test_validate_feglm_poisson_against_alpaca`
- [ ] Create `validation/econometrics/feglm.md`

### Documentation (Phase 5)
- [ ] Add FEGLM section to ECONOMETRICS_GUIDE.md
- [ ] Update DEVELOPMENT_REPORT.md

### Benchmarks (Phase 6)
- [ ] Add Criterion benchmark for FEGLM
- [ ] Create R benchmark script
- [ ] Document performance comparison

## Estimated Scope

- `feglm.rs`: ~800 lines
- Extensions to `hdfe.rs`: ~100 lines
- MCP tool: ~50 lines
- Tests: ~400 lines
- Documentation: ~200 lines

Total: ~1550 lines
