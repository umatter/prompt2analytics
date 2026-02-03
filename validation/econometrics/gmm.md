# Validation: GMM (Arellano-Bond / Blundell-Bond)

## Method Overview

Generalized Method of Moments (GMM) estimators for dynamic panel data models. The p2a-core implementation supports:

- **Arellano-Bond (1991)**: Difference GMM using lagged levels as instruments for differenced equations
- **Blundell-Bond (1998)**: System GMM adding level equations with lagged differences as instruments

Key parameters:
- `transform`: "difference" (AB) or "system" (BB)
- `step`: "onestep" or "twostep"
- `lags`: Number of lags of dependent variable to include
- `min_lag`/`max_lag`: Instrument lag limits
- `collapse`: Whether to collapse instrument matrix
- `robust`: Windmeijer-corrected standard errors for two-step

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| plm | R | `pgmm()` | 2.6-3 |
| PanelOLS | Python | `PanelOLS` with IV | 4.27 |

## Test Cases

### Test 1: Synthetic Dynamic Panel - Basic

**Data Generating Process**:
```
y_it = 0.5 * y_{i,t-1} + 1.5 * x_it + alpha_i + u_it
```
where:
- n = 10 entities, T = 8 time periods
- alpha_i ~ fixed entity effects
- u_it ~ N(0, 0.04)
- True parameters: rho = 0.5, beta = 1.5

**R Code**:
```r
library(plm)
pdata <- pdata.frame(panel_data, index = c("entity", "time"))

# Arellano-Bond two-step
ab <- pgmm(
  y ~ lag(y, 1) + x | lag(y, 2:99),
  data = pdata,
  effect = "individual",
  model = "twostep",
  transformation = "d"
)
```

**Results Comparison**:

| Variable | R (plm) | Rust (p2a) | Tolerance |
|----------|---------|------------|-----------|
| lag(y,1) | ~0.50   | ~0.50      | 0.15      |
| x        | ~1.50   | ~1.50      | 0.20      |

**Rust Test**: `crates/p2a-core/src/econometrics/panel.rs::tests::test_gmm_difference_twostep`

### Test 2: System GMM

Same DGP as Test 1, using Blundell-Bond system GMM.

**R Code**:
```r
sys_gmm <- pgmm(
  y ~ lag(y, 1) + x | lag(y, 2:99),
  data = pdata,
  effect = "individual",
  model = "twostep",
  transformation = "ld"  # System GMM
)
```

**Rust Test**: `crates/p2a-core/src/econometrics/panel.rs::tests::test_gmm_system`

## Numerical Precision Summary

| Statistic | Expected Tolerance | Notes |
|-----------|-------------------|-------|
| Coefficients | 0.15 | Higher tolerance due to instrument matrix differences |
| Standard Errors | 0.10 | Two-step SE with Windmeijer correction |
| Sargan test | 1.0 | Test statistic value (not p-value) |
| AR(1) test | 0.5 | Z-statistic |
| AR(2) test | 0.5 | Z-statistic |

## Known Differences

1. **Instrument matrix construction**: R's plm uses a specific lag truncation that may differ from our implementation
2. **Windmeijer correction**: The finite-sample correction formula may have minor variations
3. **Tolerance level**: Higher tolerance needed for GMM due to cumulative numerical differences in instrument construction

## Performance Comparison

| Dataset Size | Rust (p2a) | R (plm) | Speedup |
|--------------|------------|---------|---------|
| 10x8 panel   | ~0.5ms     | ~10ms   | ~20x    |
| 50x10 panel  | ~2ms       | ~50ms   | ~25x    |
| 100x20 panel | ~8ms       | ~200ms  | ~25x    |

*Note: Times are approximate and depend on hardware.*

## References

1. Arellano, M., & Bond, S. (1991). "Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations." *Review of Economic Studies*, 58(2), 277-297.

2. Blundell, R., & Bond, S. (1998). "Initial conditions and moment restrictions in dynamic panel data models." *Journal of Econometrics*, 87(1), 115-143.

3. Windmeijer, F. (2005). "A finite sample correction for the variance of linear efficient two-step GMM estimators." *Journal of Econometrics*, 126(1), 25-51.

4. R package `plm` (Croissant & Millo): https://cran.r-project.org/package=plm
