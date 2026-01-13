# Implementation Plan: Survival Analysis Suite

## Summary

Implement a comprehensive survival analysis module in p2a-core with four main methods:
1. **Kaplan-Meier estimator** - Non-parametric survival curve estimation
2. **Cox Proportional Hazards** - Semi-parametric regression
3. **Accelerated Failure Time (AFT) models** - Parametric survival regression
4. **Competing Risks / Aalen-Johansen** - Multi-state survival analysis

## Phase 1: Core Data Structures

Create `crates/p2a-core/src/econometrics/survival.rs` with:

### Survival Data Types
```rust
/// Observation with censoring status
pub struct SurvivalObservation {
    pub time: f64,        // Event or censoring time
    pub event: bool,      // true = event observed, false = censored
    pub event_type: u8,   // For competing risks (0 = censored, 1,2,3... = event types)
}

/// Censoring types
pub enum CensoringType {
    Right,      // Most common: event not observed by end of study
    Left,       // Event occurred before observation began
    Interval,   // Event occurred between two time points
}
```

## Phase 2: Kaplan-Meier Estimator

### Mathematical Formulation
```
Ŝ(t) = ∏(t_i ≤ t) (1 - d_i / n_i)

where:
  d_i = number of events at time t_i
  n_i = number at risk just before t_i
```

### API Design
```rust
pub struct KaplanMeierResult {
    pub times: Vec<f64>,           // Distinct event times
    pub survival: Vec<f64>,        // S(t) estimates
    pub std_errors: Vec<f64>,      // Greenwood's formula
    pub ci_lower: Vec<f64>,        // 95% CI lower bound
    pub ci_upper: Vec<f64>,        // 95% CI upper bound
    pub n_at_risk: Vec<usize>,     // Risk set at each time
    pub n_events: Vec<usize>,      // Events at each time
    pub n_censored: Vec<usize>,    // Censored at each time
    pub median_survival: Option<f64>,
}

pub fn run_kaplan_meier(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    group_col: Option<&str>,     // For stratified KM
    conf_level: f64,             // Default 0.95
) -> EconResult<Vec<KaplanMeierResult>>;  // One per group
```

### Variance (Greenwood's Formula)
```
Var(Ŝ(t)) = Ŝ(t)² × Σ(t_i ≤ t) d_i / (n_i × (n_i - d_i))
```

### Log-Rank Test (for comparing groups)
```rust
pub struct LogRankResult {
    pub chi_squared: f64,
    pub df: usize,
    pub p_value: f64,
}

pub fn log_rank_test(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    group_col: &str,
) -> EconResult<LogRankResult>;
```

## Phase 3: Cox Proportional Hazards Model

### Mathematical Formulation
Hazard function:
```
h(t|X) = h₀(t) × exp(β'X)
```

Partial log-likelihood:
```
ℓ(β) = Σᵢ δᵢ [β'xᵢ - log(Σⱼ∈R(tᵢ) exp(β'xⱼ))]

where:
  δᵢ = event indicator (1 if event, 0 if censored)
  R(tᵢ) = risk set at time tᵢ
```

Score function (gradient):
```
U(β) = ∂ℓ/∂β = Σᵢ δᵢ [xᵢ - x̄(β, tᵢ)]

where x̄(β, t) = Σⱼ∈R(t) xⱼ exp(β'xⱼ) / Σⱼ∈R(t) exp(β'xⱼ)
```

Hessian (information matrix):
```
H(β) = -∂²ℓ/∂β∂β' = Σᵢ δᵢ V(β, tᵢ)

where V(β, t) = weighted variance of X in risk set
```

### API Design
```rust
pub enum TiesMethod {
    Breslow,    // Default, faster
    Efron,      // More accurate with many ties
}

pub struct CoxConfig {
    pub ties: TiesMethod,
    pub max_iter: usize,
    pub tolerance: f64,
    pub robust_se: bool,  // Sandwich estimator
}

pub struct CoxResult {
    pub variables: Vec<String>,
    pub coefficients: Vec<f64>,       // β
    pub std_errors: Vec<f64>,
    pub z_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub hazard_ratios: Vec<f64>,      // exp(β)
    pub hr_ci_lower: Vec<f64>,        // 95% CI for HR
    pub hr_ci_upper: Vec<f64>,
    pub log_likelihood: f64,
    pub log_likelihood_null: f64,
    pub concordance: f64,             // C-statistic
    pub concordance_se: f64,
    pub wald_test: f64,               // Overall model test
    pub wald_p_value: f64,
    pub score_test: f64,              // Alternative test
    pub score_p_value: f64,
    pub n_obs: usize,
    pub n_events: usize,
    pub converged: bool,
    pub iterations: usize,
}

pub fn run_cox_ph(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    x_cols: &[&str],
    config: Option<CoxConfig>,
) -> EconResult<CoxResult>;
```

### Newton-Raphson Algorithm
```
β^(k+1) = β^(k) - H^(-1)(β^(k)) × U(β^(k))
```

## Phase 4: Accelerated Failure Time (AFT) Models

### Mathematical Formulation
```
log(T) = μ + β'X + σε

where ε follows a specified distribution:
  - Exponential: ε ~ Gumbel (minimum)
  - Weibull: ε ~ Gumbel (minimum)
  - Log-normal: ε ~ Normal
  - Log-logistic: ε ~ Logistic
```

### API Design
```rust
pub enum AftDistribution {
    Exponential,
    Weibull,
    LogNormal,
    LogLogistic,
}

pub struct AftConfig {
    pub distribution: AftDistribution,
    pub max_iter: usize,
    pub tolerance: f64,
}

pub struct AftResult {
    pub distribution: AftDistribution,
    pub variables: Vec<String>,
    pub coefficients: Vec<f64>,
    pub std_errors: Vec<f64>,
    pub z_stats: Vec<f64>,
    pub p_values: Vec<f64>,
    pub acceleration_factors: Vec<f64>,  // exp(β)
    pub scale: f64,                       // σ for Weibull/LogNormal
    pub shape: Option<f64>,               // Shape parameter if applicable
    pub log_likelihood: f64,
    pub aic: f64,
    pub bic: f64,
    pub n_obs: usize,
    pub n_events: usize,
    pub converged: bool,
}

pub fn run_aft(
    dataset: &Dataset,
    time_col: &str,
    event_col: &str,
    x_cols: &[&str],
    config: Option<AftConfig>,
) -> EconResult<AftResult>;
```

### Likelihood with Right Censoring
```
L = ∏ᵢ f(tᵢ)^δᵢ × S(tᵢ)^(1-δᵢ)

log L = Σᵢ [δᵢ × log f(tᵢ) + (1-δᵢ) × log S(tᵢ)]
```

## Phase 5: Competing Risks / Aalen-Johansen

### Mathematical Formulation
Cause-specific hazard for event type k:
```
λₖ(t) = lim(Δt→0) P(t ≤ T < t+Δt, event=k | T ≥ t) / Δt
```

Cumulative incidence function (CIF):
```
F̂ₖ(t) = Σ(tᵢ ≤ t) Ŝ(tᵢ₋₁) × dₖᵢ / nᵢ

where:
  Ŝ(t) = Kaplan-Meier for all-cause survival
  dₖᵢ = events of type k at time tᵢ
```

### API Design
```rust
pub struct AalenJohansenResult {
    pub event_type: u8,
    pub times: Vec<f64>,
    pub cumulative_incidence: Vec<f64>,  // CIF
    pub std_errors: Vec<f64>,
    pub ci_lower: Vec<f64>,
    pub ci_upper: Vec<f64>,
}

pub struct CompetingRisksResult {
    pub event_types: Vec<u8>,
    pub cifs: Vec<AalenJohansenResult>,  // One per event type
    pub n_obs: usize,
    pub n_events_by_type: Vec<usize>,
    pub n_censored: usize,
}

pub fn run_competing_risks(
    dataset: &Dataset,
    time_col: &str,
    event_type_col: &str,  // 0 = censored, 1,2,3... = event types
    conf_level: f64,
) -> EconResult<CompetingRisksResult>;
```

## Phase 6: MCP Tools

Add to `crates/p2a-mcp/src/server.rs`:

1. `survival_kaplan_meier` - Kaplan-Meier curve estimation
2. `survival_log_rank` - Log-rank test for group comparison
3. `survival_cox_ph` - Cox proportional hazards regression
4. `survival_aft` - Accelerated failure time models
5. `survival_competing_risks` - Aalen-Johansen estimator

## Phase 7: Visualization

Add to `crates/p2a-core/src/visualization/charts.rs`:

1. `survival_curve_plot` - Kaplan-Meier curves with CI bands
2. `cumulative_incidence_plot` - Stacked CIF plot for competing risks
3. `hazard_ratio_plot` - Forest plot for Cox regression coefficients

## File Structure

```
crates/p2a-core/src/econometrics/
├── mod.rs              # Add: mod survival; and exports
└── survival.rs         # NEW: ~1500 lines
    ├── Types: SurvivalObservation, CensoringType, etc.
    ├── Kaplan-Meier: run_kaplan_meier, log_rank_test
    ├── Cox PH: run_cox_ph with Newton-Raphson
    ├── AFT: run_aft for Weibull/LogNormal/LogLogistic
    └── Competing Risks: run_competing_risks (Aalen-Johansen)
```

## Dependencies

No new crates needed. Use existing:
- `ndarray` - Matrix operations
- `faer` - Linear algebra (matrix inverse)
- `statrs` - Normal, Chi-squared distributions (already used)

## Test Strategy

1. **Unit tests** in `survival.rs`:
   - Compare against known results from R `survival` package
   - Use standard datasets (e.g., lung cancer, veteran)

2. **Validation** in `validation/econometrics/survival.md`:
   - Document R code for each test case
   - Compare coefficients, SEs, p-values within tolerances

## References

1. Cox, D.R. (1972). "Regression Models and Life Tables". *JRSS B*, 34:187-220.
2. Kaplan, E.L. & Meier, P. (1958). "Nonparametric Estimation from Incomplete Observations". *JASA*, 53:457-481.
3. Aalen, O.O. & Johansen, S. (1978). "An Empirical Transition Matrix". *Scandinavian J. Statistics*, 5:141-150.
4. R package `survival` (Therneau & Grambsch). https://cran.r-project.org/package=survival
5. Klein, J.P. & Moeschberger, M.L. (2003). *Survival Analysis: Techniques for Censored and Truncated Data*. Springer.

## Implementation Order

1. **Kaplan-Meier** (simplest, foundational)
2. **Log-rank test** (uses KM infrastructure)
3. **Cox PH** (most used, moderate complexity)
4. **AFT models** (parametric, uses similar MLE pattern as discrete.rs)
5. **Competing risks** (builds on KM)
6. **MCP tools** (after core is tested)
7. **Visualization** (final polish)

## Estimated Scope

- ~1500 lines of Rust code for survival.rs
- ~200 lines for MCP tool definitions
- ~150 lines for visualization additions
- ~300 lines for tests
