# Synthetic Control with Prediction Intervals (SCPI) Validation

## Method Description

SCPI extends the classic synthetic control method (Abadie et al., 2010) with proper uncertainty quantification through prediction intervals. Developed by Cattaneo, Feng, and Titiunik (2021), SCPI provides:

1. **Multiple constraint types**: Simplex (classic SC), Lasso, Ridge, Lasso-Simplex
2. **Prediction intervals**: Account for both in-sample and out-of-sample variance
3. **Variance decomposition**: Separate estimation of in-sample and out-of-sample components

## Mathematical Framework

### Weight Estimation

The synthetic control weights W are estimated by minimizing:

```
min_W ||Y1_pre - Y0_pre * W||^2 + penalty(W)
```

Subject to constraints depending on type:
- **Simplex**: sum(W) = 1, W >= 0
- **Lasso**: add lambda * ||W||_1
- **Ridge**: add lambda * ||W||_2^2
- **Lasso-Simplex**: Lasso + simplex

### Prediction Intervals

Intervals are constructed as (Cattaneo et al. 2021, Eq. 3.5):

```
[effect - c_alpha * sigma_hat, effect + c_alpha * sigma_hat]
```

Where:
- sigma_hat^2 = sigma_in^2 + sigma_out^2
- sigma_in^2: In-sample variance from pre-treatment fit residuals
- sigma_out^2: Out-of-sample variance (estimated via subgaussian bounds or CV)
- c_alpha: Critical value (t or z distribution)

## Implementation Location

- **Core**: `crates/p2a-core/src/econometrics/scpi.rs`
- **MCP Tool**: `scpi` in `crates/p2a-mcp/src/server.rs`
- **Exports**: `p2a_core::{run_scpi, SCPIConfig, SCPIConstraint, VarianceMethod, SCPIResult}`

## Reference Implementation

### R Package: scpi (Cattaneo, Feng, Palomba, Titiunik)

- **CRAN**: https://cran.r-project.org/package=scpi
- **Documentation**: https://nppackages.github.io/scpi/
- **Version**: 2.2.5 (at time of validation)

## Test Cases

### Test Case 1: Basic Simplex Constraint

**Setup**:
- 10 time periods (7 pre-treatment, 3 post-treatment)
- 1 treated unit with trend + treatment effect
- 5 donor units with similar trends

**Rust Test**: `test_scpi_basic`

```rust
let treated = array![
    10.0, 11.5, 12.8, 14.1, 15.5, 16.9, 18.2,  // Pre-treatment
    22.0, 23.5, 25.0                            // Post-treatment (effect ~3)
];
let donors = Array2::from_shape_fn((10, 5), |(t, j)| {
    10.0 + 1.4 * t as f64 + 0.3 * j as f64 + 0.1 * ((t * j) as f64).sin()
});

let config = SCPIConfig::default();  // Simplex, alpha=0.05
let result = run_scpi(&treated.view(), &donors.view(), 7, config)?;
```

**Validation Criteria**:
- Weights sum to 1.0 (tolerance: 1e-6)
- All weights non-negative
- Prediction intervals properly ordered: lower < effect < upper
- Pre-treatment fit: RMSPE > 0

### Test Case 2: Lasso Constraint

**Rust Test**: `test_scpi_lasso`

```rust
let config = SCPIConfig {
    constraint: SCPIConstraint::Lasso { lambda: 0.1 },
    ..Default::default()
};
```

**Validation Criteria**:
- Produces sparse weights (n_effective_donors <= n_donors)
- Some weights may be exactly zero

### Test Case 3: Ridge Constraint

**Rust Test**: `test_scpi_ridge`

```rust
let config = SCPIConfig {
    constraint: SCPIConstraint::Ridge { lambda: 0.1 },
    ..Default::default()
};
```

**Validation Criteria**:
- All weights typically non-zero (shrinkage, not selection)
- Smaller magnitude weights than unconstrained OLS

### Test Case 4: Lasso-Simplex Constraint

**Rust Test**: `test_scpi_lasso_simplex`

```rust
let config = SCPIConfig {
    constraint: SCPIConstraint::LassoSimplex { lambda: 0.05 },
    ..Default::default()
};
```

**Validation Criteria**:
- Weights sum to 1.0 (tolerance: 1e-4)
- All weights non-negative
- Some sparsity induced by L1 penalty

### Test Case 5: Variance Methods

**Rust Test**: `test_variance_methods`

Tests Gaussian and Subgaussian variance estimation:

**Validation Criteria**:
- All variance components > 0
- total_var = in_sample_var + out_sample_var

## Algorithm Details

### Simplex Weight Optimization

Uses Frank-Wolfe algorithm:
1. Initialize with uniform weights: W = 1/J
2. Compute gradient: grad = H * W + c where H = X0'X0, c = -X0'Y1
3. Find minimizing vertex: argmin_j grad_j
4. Line search for optimal step size
5. Update: W_new = W + alpha * (e_j - W)
6. Repeat until convergence

### Lasso Weight Optimization

Uses coordinate descent with soft-thresholding:
1. For each coordinate j:
   - Compute partial residual
   - Apply soft-thresholding: W_j = sign(r_j) * max(0, |r_j| - lambda)
2. Repeat until convergence

### ADMM for Lasso-Simplex

Uses Alternating Direction Method of Multipliers:
1. W-update: Minimize augmented Lagrangian (closed form)
2. Z-update: Apply soft-thresholding then project to simplex
3. U-update: Dual variable update
4. Repeat until primal and dual residuals converge

### Simplex Projection

Uses Duchi et al. (2008) algorithm:
1. Sort vector in descending order
2. Find threshold rho via cumulative sum
3. Project: x_proj = max(0, x - theta)

## Tolerances

| Quantity | Tolerance |
|----------|-----------|
| Weight sum (simplex) | 1e-4 |
| Weight non-negativity | 1e-6 |
| Optimization convergence | 1e-8 |
| Variance positivity | > 0 |

## Known Limitations

1. **Critical value approximation**: Uses simplified t-distribution approximation; exact values require special functions
2. **Small sample variance**: Out-of-sample variance estimation less reliable with few pre-treatment periods
3. **Lasso path**: Does not compute full regularization path; single lambda value

## References

### Primary
- Cattaneo, M. D., Feng, Y., & Titiunik, R. (2021). "Prediction Intervals for Synthetic Control Methods." *Journal of the American Statistical Association*, 116(536), 1865-1880. DOI: 10.1080/01621459.2021.1979561

### Background
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies: Estimating the Effect of California's Tobacco Control Program." *Journal of the American Statistical Association*, 105(490), 493-505.

- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.

### Algorithms
- Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). "Efficient Projections onto the l1-Ball for Learning in High Dimensions." *ICML 2008*.

### Software
- R package `scpi`: Cattaneo, M. D., Feng, Y., Palomba, F., & Titiunik, R. https://cran.r-project.org/package=scpi

## MCP extraction (T2-WP11)

The MCP tool `scpi` reads the unit column as a factor, the time column as a whole-number time
index and the outcome as a numeric column, in one request to the shared extractor with the
`Fail` policy. Units may be text or integer ids; they are the factor levels in R `factor()`
order (integer ids numerically, text in byte order), and the donors are the other units in that
order (before T2-WP11 it was hash order and changed from run to run). Times are the sorted
distinct values. With fewer rows than units x times cells the panel cannot be balanced, and the
tool error names the counts and the first empty cell (`scpi needs a balanced panel: 10 units x
20 times = 200 cells, but the data has 199 rows (unit 2 has no row for time 1940; cells
without a row: 1)`), found from the sorted cells of the rows, so the grid is never larger than the data.
Otherwise the outcome grid is filled in one pass: a second row for a (unit, time) cell
(`unit 1 appears twice at time 1935`) and a missing outcome are tool errors naming the unit and
time or the column. Before,
the handler refused integer unit columns ("must be string type") and looked every cell up with
a full-table filter (finding TOOL-5, scpi site).

Tests (`crates/p2a-mcp/src/tools/handlers/causal.rs`, on `validation/datasets/grunfeld.csv`;
not registered, no R value): `test_mcp_scpi_accepts_integer_unit_ids` and
`test_mcp_scpi_reports_unknown_units_and_missing_cells`.
