# Goodman-Bacon Decomposition Validation

## Method Overview

The Goodman-Bacon (2021) decomposition breaks down a two-way fixed effects (TWFE) difference-in-differences estimate into a weighted average of all possible 2x2 DiD comparisons between timing groups. This reveals:

1. Which comparisons contribute to the overall estimate
2. The weight assigned to each comparison
3. Potential biases from "forbidden" comparisons (later vs. earlier treated)

## Implementation Details

**File**: `/home/umatter/tools/p2a-stats/crates/p2a-core/src/econometrics/bacon.rs`

**Function**: `bacon_decomp(dataset, outcome_col, unit_col, time_col, treatment_col)`

### Comparison Types

| Type | Description | Clean? |
|------|-------------|--------|
| Treated vs Never-Treated | Units that eventually get treated vs. units that never do | Yes |
| Treated vs Not-Yet-Treated | Early-treated vs. later-treated (before later is treated) | Yes |
| Later vs Earlier Treated | Late-treated vs. early-treated (both already treated) | Potentially problematic |

### Weight Formula

Following Goodman-Bacon (2021), weights depend on:

```
w_{kl} ∝ (n_k / n) × (n_l / n) × V(D_{kl})
```

Where:
- `n_k, n_l` = group sizes
- `n` = total observations
- `V(D_{kl})` = variance of treatment indicator in the comparison window

## Reference Implementations

### R Package: bacondecomp

```r
# Install
install.packages("bacondecomp")

# Usage
library(bacondecomp)
library(bacondecomp)

# Example with built-in data
data(castle)
result <- bacon(l_homicide ~ post,
                data = castle,
                id_var = "state",
                time_var = "year")

# View components
print(result)
aggregate(result)
```

### Stata: bacondecomp

```stata
// Install
ssc install bacondecomp

// Usage
bacondecomp y post, id(state) time(year)
```

## Test Cases

### Test Case 1: Synthetic Panel with Known Properties

**Data Generation (R)**:
```r
set.seed(42)

# 4 units, 5 time periods (2000-2004)
# Unit 1: treated in 2001
# Unit 2: treated in 2003
# Units 3-4: never treated
# True treatment effect: 2.0

n_units <- 4
n_times <- 5
years <- 2000:2004
treatment_times <- c(2001, 2003, 0, 0)  # 0 = never treated
true_effect <- 2.0

# Generate panel
data <- expand.grid(unit = 1:n_units, year = years)
data$g <- treatment_times[data$unit]
data$treated <- as.numeric(data$year >= data$g & data$g > 0)

# Outcome with unit FE, time trend, and treatment effect
data$y <- data$unit * 2 +           # unit fixed effect
          (data$year - 2000) * 0.5 + # time trend
          data$treated * true_effect + # treatment effect
          rnorm(nrow(data), 0, 0.5)    # noise

# Run decomposition
library(bacondecomp)
result <- bacon(y ~ treated, data = data, id_var = "unit", time_var = "year")
```

**Expected Properties**:
- 2 timing groups: 2001 and 2003
- 2 never-treated units
- Weights should sum to 1
- Overall estimate should be close to 2.0

### Test Case 2: Validation Against R

**Test Data**:
- Same synthetic panel as Test Case 1

**Rust Test Function**: `test_bacon_decomp_basic`

**Comparison Criteria**:
| Metric | Tolerance |
|--------|-----------|
| Weight sum | |diff| < 0.01 |
| Component presence | Must have all types |
| Estimates | Should be reasonable (< 10.0) |

## References

### Original Paper

Goodman-Bacon, A. (2021). "Difference-in-Differences with Variation in Treatment Timing". *Journal of Econometrics*, 225(2), 254-277. https://doi.org/10.1016/j.jeconom.2021.03.014

### Related Methods

- Callaway, B., & Sant'Anna, P.H.C. (2021). "Difference-in-Differences with Multiple Time Periods". *Journal of Econometrics*, 225(2), 200-230.

- de Chaisemartin, C., & D'Haultfoeuille, X. (2020). "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects". *American Economic Review*, 110(9), 2964-2996.

- Sun, L., & Abraham, S. (2021). "Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects". *Journal of Econometrics*, 225(2), 175-199.

### Software

- R: `bacondecomp` package (Flack & Sant'Anna) - https://cran.r-project.org/package=bacondecomp
- Stata: `bacondecomp` (Goodman-Bacon, Goldring, & Nichols) - https://github.com/tgoldring/bacondecomp

## Validation Status

| Test | Status | Notes |
|------|--------|-------|
| Basic decomposition | PASS | Weights sum to 1, all comparison types present |
| Comparison types | PASS | Identifies all three types correctly |
| Weight distribution | PASS | Non-negative weights, type sums match total |
| Estimate reasonableness | PASS | Estimates within expected range |

## Known Limitations

1. **Assumes no treatment reversals**: Once treated, units remain treated
2. **Balanced panel preferred**: Unbalanced panels may affect weights
3. **Precision with few timing groups**: Standard errors not computed (use bootstrap separately)

## Usage Example (Rust)

```rust
use p2a_core::econometrics::bacon_decomp;
use p2a_core::data::Dataset;

let result = bacon_decomp(
    &dataset,
    "outcome",      // Y variable
    "state",        // Unit ID
    "year",         // Time period
    "treated",      // Binary treatment (0/1)
)?;

println!("TWFE estimate: {:.4}", result.overall_estimate);
println!("Weight from clean comparisons: {:.2}%",
         result.treated_vs_never * 100.0);

if result.later_vs_earlier > 0.3 {
    println!("WARNING: High weight from forbidden comparisons");
}
```
