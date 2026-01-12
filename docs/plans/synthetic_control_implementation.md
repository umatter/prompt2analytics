# Synthetic Control Method Implementation Plan

## Overview

Implement the Synthetic Control Method (SCM) as described by Abadie, Diamond, and Hainmueller (2010) for comparative case studies. This method creates a weighted combination of control units to estimate the counterfactual outcome for a treated unit.

## References

### Original Papers
- Abadie, A. & Gardeazabal, J. (2003). "The Economic Costs of Conflict: A Case Study of the Basque Country." *American Economic Review*, 93(1), 112-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493-505.
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.

### Reference Implementations
- R package `Synth` (Abadie, Diamond, Hainmueller)
- R package `tidysynth` (Eric Dunford)

## Mathematical Formulation

### Problem Setup
Let:
- $J+1$ units, where unit 1 is treated and units 2,...,J+1 are donors
- $T$ time periods, with treatment occurring at time $T_0$
- $Y_{jt}$ = outcome for unit $j$ at time $t$
- $X_j$ = vector of predictors for unit $j$ (pre-treatment characteristics)

### Optimization Problem

**Objective**: Find weights $W^* = (w_2, ..., w_{J+1})$ that minimize:

```
||X_1 - X_0 W||_V = sqrt((X_1 - X_0 W)' V (X_1 - X_0 W))
```

**Subject to**:
- $w_j \geq 0$ for all $j$
- $\sum_{j=2}^{J+1} w_j = 1$

Where:
- $X_1$ = (k × 1) vector of predictors for treated unit
- $X_0$ = (k × J) matrix of predictors for donor units
- $V$ = (k × k) positive semidefinite diagonal matrix of predictor weights

### Nested Optimization

1. **Outer loop**: Optimize $V$ to minimize pre-treatment MSPE:
   ```
   V* = argmin_V sum_{t=1}^{T_0} (Y_{1t} - sum_j w_j*(V) Y_{jt})^2
   ```

2. **Inner loop**: For given $V$, solve QP for $W$:
   ```
   W*(V) = argmin_W (X_1 - X_0 W)' V (X_1 - X_0 W)
   s.t. w_j >= 0, sum_j w_j = 1
   ```

### Treatment Effect Estimation

The estimated treatment effect at time $t > T_0$ is:
```
τ_t = Y_{1t} - sum_{j=2}^{J+1} w_j* Y_{jt}
```

### Inference via Placebo Tests

For each donor unit $j$, apply SCM as if $j$ were treated:
1. Construct synthetic control for unit $j$
2. Compute RMSPE ratio: (post-treatment RMSPE) / (pre-treatment RMSPE)
3. P-value = Rank of treated unit's ratio / Total units

## Implementation Design

### New Dependencies

Add to `crates/p2a-core/Cargo.toml`:
```toml
# Quadratic programming solver (pure Rust, Goldfarb-Idnani algorithm)
quadprog = "0.1"
```

### File Structure

```
crates/p2a-core/src/econometrics/
├── mod.rs                 # Add synth exports
└── synth.rs               # NEW: Synthetic control implementation
```

### API Design

```rust
// Configuration
pub struct SynthConfig {
    /// Time period of treatment (first post-treatment period)
    pub treatment_time: i64,
    /// Unit ID/name of the treated unit
    pub treated_unit: String,
    /// Pre-treatment periods to use for optimization
    pub optimization_window: Option<(i64, i64)>,
    /// Predictor aggregation functions (mean, first, last, custom time window)
    pub predictor_aggregation: PredictorAggregation,
    /// V matrix optimization method
    pub v_method: VOptimization,
    /// Tolerance for QP solver
    pub qp_tolerance: f64,
    /// Maximum iterations for V optimization
    pub max_iter: usize,
    /// Whether to run placebo tests for inference
    pub run_placebos: bool,
}

pub enum VOptimization {
    /// Data-driven: minimize pre-treatment MSPE
    DataDriven,
    /// Equal weights for all predictors
    Equal,
    /// User-specified weights
    Custom(Vec<f64>),
}

// Main function
pub fn run_synthetic_control(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    config: SynthConfig,
) -> EconResult<SynthResult>

// Predictor specification
pub struct PredictorSpec {
    pub column: String,
    pub aggregation: TimeAggregation,
    pub time_window: Option<(i64, i64)>,
}

pub enum TimeAggregation {
    Mean,
    First,
    Last,
    Sum,
}
```

### Result Structure

```rust
pub struct SynthResult {
    // Basic info
    pub treated_unit: String,
    pub treatment_time: i64,
    pub n_donors: usize,
    pub n_pre_periods: usize,
    pub n_post_periods: usize,

    // Weights
    pub unit_weights: Vec<(String, f64)>,  // (unit_name, weight)
    pub predictor_weights: Vec<(String, f64)>,  // V diagonal

    // Fit diagnostics
    pub predictor_balance: Vec<PredictorBalance>,
    pub pre_treatment_mspe: f64,
    pub pre_treatment_rmspe: f64,

    // Treatment effects
    pub treatment_effects: Vec<TimeEffect>,  // Effect at each post-period
    pub average_effect: f64,
    pub cumulative_effect: f64,

    // Time series
    pub actual_outcome: Vec<(i64, f64)>,
    pub synthetic_outcome: Vec<(i64, f64)>,

    // Placebo inference (optional)
    pub placebo_results: Option<PlaceboResults>,
}

pub struct PredictorBalance {
    pub predictor: String,
    pub treated_value: f64,
    pub synthetic_value: f64,
    pub difference: f64,
}

pub struct TimeEffect {
    pub time: i64,
    pub effect: f64,
    pub actual: f64,
    pub synthetic: f64,
}

pub struct PlaceboResults {
    pub rmspe_ratios: Vec<(String, f64)>,  // (unit, ratio)
    pub treated_rank: usize,
    pub p_value: f64,
    pub n_units: usize,
}
```

### Implementation Steps

#### Step 1: Core QP Solver Wrapper
Create a wrapper around the `quadprog` crate for our specific constraints.

```rust
fn solve_weights_qp(
    x0: &Array2<f64>,  // Donor predictors (k × J)
    x1: &Array1<f64>,  // Treated predictors (k × 1)
    v: &Array1<f64>,   // Predictor weights (k × 1)
) -> EconResult<Array1<f64>>
```

#### Step 2: V Optimization
Implement outer loop for V matrix optimization using Nelder-Mead or BFGS.

```rust
fn optimize_v(
    z0: &Array2<f64>,  // Donor pre-treatment outcomes (T0 × J)
    z1: &Array1<f64>,  // Treated pre-treatment outcomes (T0 × 1)
    x0: &Array2<f64>,  // Donor predictors
    x1: &Array1<f64>,  // Treated predictors
    config: &SynthConfig,
) -> EconResult<(Array1<f64>, Array1<f64>)>  // (V*, W*)
```

#### Step 3: Data Preparation
Convert panel data to the required matrix format.

```rust
fn prepare_synth_data(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    treated_unit: &str,
    treatment_time: i64,
) -> EconResult<SynthData>

struct SynthData {
    x1: Array1<f64>,        // Treated predictors
    x0: Array2<f64>,        // Donor predictors
    z1: Array1<f64>,        // Treated pre-treatment outcomes
    z0: Array2<f64>,        // Donor pre-treatment outcomes
    y1_post: Array1<f64>,   // Treated post-treatment outcomes
    y0_post: Array2<f64>,   // Donor post-treatment outcomes
    donor_units: Vec<String>,
    predictor_names: Vec<String>,
    pre_times: Vec<i64>,
    post_times: Vec<i64>,
}
```

#### Step 4: Treatment Effect Calculation

```rust
fn calculate_effects(
    synth_data: &SynthData,
    weights: &Array1<f64>,
) -> Vec<TimeEffect>
```

#### Step 5: Placebo Tests

```rust
fn run_placebo_tests(
    dataset: &Dataset,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    predictors: &[PredictorSpec],
    config: &SynthConfig,
) -> EconResult<PlaceboResults>
```

### MCP Tool

Add to `crates/p2a-mcp/src/server.rs`:

```rust
#[derive(Deserialize, JsonSchema)]
pub struct SyntheticControlRequest {
    pub dataset: String,
    pub outcome: String,
    pub unit_col: String,
    pub time_col: String,
    pub treated_unit: String,
    pub treatment_time: i64,
    pub predictors: Vec<String>,
    pub predictor_time_window: Option<(i64, i64)>,
    pub run_placebos: Option<bool>,
}

#[tool(description = "Run synthetic control method for comparative case studies")]
async fn synthetic_control(&self, request: SyntheticControlRequest) -> Result<String, McpError>
```

## Test Strategy

### Test 1: Known Weights Recovery
Create data where optimal weights are known:
```rust
// Y_treated = 0.5 * Y_donor1 + 0.5 * Y_donor2
// Verify recovered weights ≈ [0.5, 0.5, 0, 0, ...]
```

### Test 2: California Tobacco Data (Classic Example)
Replicate results from Abadie et al. (2010) using California tobacco data.

### Test 3: No Treatment Effect
Verify method returns ~0 effect when no actual treatment effect exists.

### Test 4: Placebo Inference
Verify p-values are well-calibrated under known DGP.

## Timeline

1. **Core Implementation**: Implement QP wrapper, data preparation, main algorithm
2. **Inference**: Add placebo tests
3. **Testing**: Unit tests, validation against R
4. **Documentation**: Update ECONOMETRICS_GUIDE.md, add examples
5. **MCP Integration**: Add tool to server

## Validation Plan

1. Compare with R `Synth` package on California tobacco data
2. Verify weight recovery on synthetic data
3. Check placebo p-value calibration
