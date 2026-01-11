# Implementation Plan: Causal Inference Methods from causalweight

## Overview

This plan describes the implementation of causal inference methods inspired by the R `causalweight` package into `p2a-core`. The methods provide treatment effect estimation using inverse probability weighting (IPW) and doubly robust estimation.

## Source Package

- **Package**: [causalweight](https://cran.r-project.org/web/packages/causalweight/index.html) (R package)
- **Authors**: Bodory, Huber, et al.
- **Key Reference**: Huber, M. (2014). "Identifying causal mechanisms (primarily) based on inverse probability weighting." *Journal of Applied Econometrics*, 29, 920-943.

---

## Phase 0 Summary: Existing Implementations

### Already Implemented
| Method | Location | Notes |
|--------|----------|-------|
| DiD (2x2) | `econometrics/did.rs` | Basic difference-in-differences |
| IV/2SLS | `econometrics/iv.rs` | Full implementation with diagnostics |
| Logit/Probit | `econometrics/discrete.rs` | For propensity score estimation |
| Panel FE/RE | `econometrics/panel.rs` | Fixed and random effects |
| HDFE | `econometrics/hdfe.rs` | High-dimensional FE |

### To Be Implemented
| Method | Priority | Description |
|--------|----------|-------------|
| IPW Treatment Effects | High | ATE/ATT via propensity score weighting |
| Doubly Robust (AIPW) | High | Augmented IPW with outcome regression |
| Causal Mediation | Medium | Direct/indirect effect decomposition |
| LATE with Weighting | Lower | For future extension |

---

## Implementation Scope

### Phase 1: IPW Treatment Effects (`treatweight`)

**New File**: `crates/p2a-core/src/econometrics/treatment.rs`

#### Mathematical Formulation

**Propensity Score** (using existing Logit):
```
p(X) = P(D=1|X) = Λ(X'β)
```

**ATE (Normalized/Hajek)**:
```
ATE_ipw = Σ[w₁(D,p(X))·Y] / Σ[w₁] - Σ[w₀(D,p(X))·Y] / Σ[w₀]

where:
  w₁ = D / p(X)        (treated weight)
  w₀ = (1-D) / (1-p(X)) (control weight)
```

**ATT (Average Treatment Effect on Treated)**:
```
ATT_ipw = Σ[D·Y] / Σ[D] - Σ[w₀ᵃᵗᵗ·Y] / Σ[w₀ᵃᵗᵗ]

where:
  w₀ᵃᵗᵗ = (1-D)·p(X) / (1-p(X))
```

**Trimming**: Discard observations where p(X) < trim or p(X) > 1-trim (default trim=0.05)

#### API Design

```rust
/// Configuration for IPW treatment effect estimation
pub struct IpwConfig {
    /// Trimming threshold for propensity scores (default: 0.05)
    pub trim: f64,
    /// Whether to estimate ATE (false) or ATT (true)
    pub att: bool,
    /// Number of bootstrap replications for SE (default: 999)
    pub bootstrap: usize,
    /// Use normalized (Hajek) weights (default: true)
    pub normalized: bool,
}

/// Result from IPW treatment effect estimation
pub struct IpwResult {
    /// Estimated treatment effect (ATE or ATT)
    pub effect: f64,
    /// Standard error (via bootstrap)
    pub std_error: f64,
    /// 95% confidence interval
    pub ci_lower: f64,
    pub ci_upper: f64,
    /// p-value
    pub p_value: f64,
    /// Number of observations (after trimming)
    pub n_obs: usize,
    /// Number trimmed
    pub n_trimmed: usize,
    /// Propensity score summary statistics
    pub ps_summary: PropensityScoreSummary,
}

pub fn run_ipw_treatment(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    covariates: &[&str],
    config: IpwConfig,
) -> EconResult<IpwResult>
```

#### Implementation Steps

1. Use existing `run_logit()` to estimate propensity scores
2. Apply trimming to propensity scores
3. Compute IPW weights for ATE or ATT
4. Compute point estimate using normalized (Hajek) estimator
5. Bootstrap for standard errors and confidence intervals
6. Return comprehensive results

---

### Phase 2: Doubly Robust Estimation (`ATETDML` simplified)

**Enhancement to**: `crates/p2a-core/src/econometrics/treatment.rs`

#### Mathematical Formulation

**AIPW Estimator for ATE**:
```
τ_AIPW = (1/n) Σᵢ [
    μ̂⁽¹⁾(Xᵢ) - μ̂⁽⁰⁾(Xᵢ)
    + Dᵢ/ê(Xᵢ) · (Yᵢ - μ̂⁽¹⁾(Xᵢ))
    - (1-Dᵢ)/(1-ê(Xᵢ)) · (Yᵢ - μ̂⁽⁰⁾(Xᵢ))
]
```

where:
- `μ̂⁽ᵈ⁾(X)` = outcome model prediction for treatment d
- `ê(X)` = estimated propensity score

**AIPW for ATT**:
```
τ_ATT_AIPW = (1/n₁) Σᵢ [
    Dᵢ(Yᵢ - μ̂⁽⁰⁾(Xᵢ))
    - (1-Dᵢ)·p̂(Xᵢ)/(1-p̂(Xᵢ)) · (Yᵢ - μ̂⁽⁰⁾(Xᵢ))
]
```

#### API Design

```rust
/// Doubly robust estimator method
pub enum DRMethod {
    /// Augmented IPW (default)
    AIPW,
    /// IPW only (not doubly robust)
    IPW,
    /// Regression adjustment only (not doubly robust)
    Regression,
}

/// Configuration for doubly robust estimation
pub struct DoublyRobustConfig {
    /// Estimation method
    pub method: DRMethod,
    /// Trimming threshold
    pub trim: f64,
    /// Estimate ATE (false) or ATT (true)
    pub att: bool,
    /// Number of bootstrap replications
    pub bootstrap: usize,
}

/// Result from doubly robust estimation
pub struct DoublyRobustResult {
    /// Estimated treatment effect
    pub effect: f64,
    /// Standard error
    pub std_error: f64,
    /// 95% CI
    pub ci_lower: f64,
    pub ci_upper: f64,
    /// p-value
    pub p_value: f64,
    /// Number of observations
    pub n_obs: usize,
    /// Method used
    pub method: DRMethod,
    /// Outcome model R² (treated)
    pub outcome_r2_treated: f64,
    /// Outcome model R² (control)
    pub outcome_r2_control: f64,
}

pub fn run_doubly_robust(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    covariates: &[&str],
    config: DoublyRobustConfig,
) -> EconResult<DoublyRobustResult>
```

#### Implementation Steps

1. Estimate propensity scores using logit
2. Fit separate outcome regressions for treated and control groups
3. Compute AIPW estimator combining both
4. Bootstrap for inference
5. Return results with diagnostics

---

### Phase 3: Causal Mediation Analysis (`medweight`)

**New File**: `crates/p2a-core/src/econometrics/mediation.rs`

#### Mathematical Formulation

Following Huber (2014), the natural direct and indirect effects are:

**Natural Direct Effect (NDE)**:
Effect of treatment on outcome NOT through the mediator

**Natural Indirect Effect (NIE)**:
Effect of treatment on outcome through the mediator

**Decomposition**:
```
ATE = NDE + NIE
```

**IPW-based identification** (Huber 2014, Eq. 6-7):

Weights for mediation:
```
w_NDE = D / p(X) + (1-D)·p(M,X) / ((1-p(X))·(1-p(M,X)))
w_NIE = computed similarly with different conditioning
```

#### API Design

```rust
/// Result from causal mediation analysis
pub struct MediationResult {
    /// Total effect (ATE)
    pub total_effect: f64,
    /// Natural direct effect
    pub direct_effect: f64,
    /// Natural indirect effect
    pub indirect_effect: f64,
    /// Proportion mediated
    pub proportion_mediated: f64,
    /// Standard errors
    pub se_total: f64,
    pub se_direct: f64,
    pub se_indirect: f64,
    /// p-values
    pub p_total: f64,
    pub p_direct: f64,
    pub p_indirect: f64,
    /// Number of observations
    pub n_obs: usize,
}

pub fn run_mediation_analysis(
    dataset: &Dataset,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    covariates: &[&str],
    bootstrap: usize,
) -> EconResult<MediationResult>
```

---

## MCP Tools to Add

| Tool Name | Description |
|-----------|-------------|
| `treatment_ipw` | IPW estimation of ATE/ATT |
| `treatment_doubly_robust` | AIPW doubly robust estimation |
| `mediation_analysis` | Causal mediation with direct/indirect effects |

---

## Testing Strategy

### Test Data Requirements

1. **Synthetic data with known DGP**: Generate data where true ATE/ATT is known
2. **Comparison with R causalweight**: Run same data through R package for validation

### Test Cases

1. **IPW Recovery Test**:
   - Generate Y = 0.5*D + X + noise
   - True ATE = 0.5
   - Verify IPW estimate is close

2. **Double Robustness Test**:
   - Test with correct propensity model
   - Test with correct outcome model
   - Test with both correct (should be more efficient)

3. **Mediation Test**:
   - Generate D → M → Y with known direct/indirect effects
   - Verify decomposition

---

## File Structure

```
crates/p2a-core/src/econometrics/
├── mod.rs              # Add exports
├── treatment.rs        # NEW: IPW and DR estimation
├── mediation.rs        # NEW: Causal mediation
├── did.rs              # Existing
├── iv.rs               # Existing
├── panel.rs            # Existing
└── ...
```

---

## Dependencies

All dependencies are already available:
- `ndarray` 0.16 - Matrix operations
- `statrs` 0.18 - Statistical distributions (for bootstrap CIs)
- Existing `run_logit` for propensity scores
- Existing `run_ols` for outcome regression

No new dependencies required.

---

## References

### Primary Sources
- Horvitz, D.G. & Thompson, D.J. (1952). "A Generalization of Sampling Without Replacement from a Finite Universe." *JASA*, 47(260), 663-685.
- Robins, J.M., Rotnitzky, A. & Zhao, L.P. (1994). "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed." *JASA*, 89(427), 846-866.
- Huber, M. (2014). "Identifying Causal Mechanisms (Primarily) Based on Inverse Probability Weighting." *J. Applied Econometrics*, 29, 920-943.
- Bang, H. & Robins, J.M. (2005). "Doubly Robust Estimation in Missing Data and Causal Inference Models." *Biometrics*, 61(4), 962-973.

### R Package Reference
- Bodory, H. & Huber, M. (2018). "causalweight: An R Package for Causal Inference and Mediation Analysis Based on Inverse Probability Weighting."

### Implementation References
- [psantanna IPW lecture notes](https://psantanna.com/Econ520/Slides/15-ipw/15slides.html)
- [Towards Data Science: Understanding AIPW](https://towardsdatascience.com/understanding-aipw-ed4097dab27a/)

---

## Implementation Order

1. **IPW Treatment Effects** (2-3 hours)
   - Propensity score estimation (reuse logit)
   - IPW weights and ATE/ATT
   - Bootstrap inference
   - MCP tool

2. **Doubly Robust Estimation** (2-3 hours)
   - AIPW estimator
   - Outcome regression models
   - Cross-validation for robustness
   - MCP tool

3. **Causal Mediation** (3-4 hours)
   - More complex weighting schemes
   - Direct/indirect effect decomposition
   - MCP tool

4. **Testing & Validation** (2 hours)
   - Unit tests with known DGP
   - Comparison with R causalweight

5. **Documentation** (1 hour)
   - Update ECONOMETRICS_GUIDE.md
   - Add MCP tool examples

---

## Approval Request

This plan proposes implementing:

1. **IPW Treatment Effects**: ATE and ATT via inverse probability weighting
2. **Doubly Robust (AIPW)**: Augmented IPW with outcome regression
3. **Causal Mediation**: Natural direct and indirect effects

Estimated total effort: 10-15 hours

Please confirm to proceed with implementation.
