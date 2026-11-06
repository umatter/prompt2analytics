# Goodman-Bacon Decomposition Validation

## Method Overview

The Goodman-Bacon (2021) decomposition breaks down a two-way fixed effects (TWFE) difference-in-differences estimate into a weighted average of all possible 2x2 DiD comparisons between timing groups. This reveals:

1. Which comparisons contribute to the overall estimate
2. The weight assigned to each comparison
3. Potential biases from "forbidden" comparisons (later vs. earlier treated)

## Implementation Details

**File**: `crates/p2a-core/src/econometrics/bacon.rs`

**Function**: `bacon_decomp(dataset, outcome_col, unit_col, time_col, treatment_col)`

### Comparison Types

| Type | Description | Clean? |
|------|-------------|--------|
| Treated vs Never-Treated | Units that eventually get treated vs. units that never do | Yes |
| Treated vs Not-Yet-Treated | Early-treated vs. later-treated (before later is treated) | Yes |
| Later vs Earlier Treated | Late-treated vs. early-treated (both already treated) | Potentially problematic |

### Weight Formula

Goodman-Bacon (2021), Theorem 1, as `bacondecomp::bacon` computes it (`calculate_weights`):

```
s_{kl} ∝ (n_k + n_l)^2 V_{kl},   V_{kl} = n_{kl} (1 - n_{kl}) D_{kl} (1 - D_{kl})
       = n_T n_C D (1 - D)
```

Where:
- `n_T`, `n_C` = rows of the treated and the control group in the comparison window
  (`n_{kl} = n_T / (n_T + n_C)`);
- `D` = the treated group's share of post-treatment rows in the window;
- the weights are normalised to sum to 1.

On a balanced panel without always-treated units the weighted sum of the 2x2 estimates is
the two-way fixed-effects coefficient (`overall_estimate`, "Overall TWFE Estimate"). Before
CAUS-10 the window length entered linearly instead of squared: timing comparisons were
overweighted and the weighted sum was 0.8056 against fixest's 0.8712 on the R case panel.

Unbalanced panels: `bacondecomp::bacon` refuses them. p2a still decomposes, and warns
("the panel is unbalanced (k of N units are not observed in every period); ... the weighted sum
of the 2x2 estimates is then not the two-way fixed-effects coefficient").

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

## R reference: case `bacon_twfe`

`validation/r/cases/causal_did_cov.R`, case `bacon_twfe`: `bacondecomp::bacon(y ~ D, data,
id_var = "unit", time_var = "time")` 0.1.1 and `fixest::feols(y ~ D | unit + time)` on
`validation/datasets/generated/cs_cov_panel.csv` (150 units x 8 periods, 1200 rows; cohorts 3, 5,
7 and never treated; `D = 1(first_treat > 0 & time >= first_treat)`). bacondecomp's never-treated
code 99999 is recoded 0; its "Earlier vs Later Treated" is p2a's `TreatedVsNotYetTreated`.

| Treated vs control | Estimate | Weight |
|---|---|---|
| 3 vs never | 1.9538168775149178 | 0.18522230685963997 |
| 3 vs 5 | 0.8938374947112511 | 0.02565850138315359 |
| 3 vs 7 | 1.541938138650405 | 0.0336767830653891 |
| 5 vs never | 1.0643198769609463 | 0.3951409213005653 |
| 5 vs 3 | -0.2871065796015502 | 0.05131700276630719 |
| 5 vs 7 | 0.5505695192061021 | 0.053882852904622545 |
| 7 vs never | 0.23548397169554036 | 0.19448342220262194 |
| 7 vs 3 | -1.1045313166488573 | 0.0336767830653891 |
| 7 vs 5 | -0.35856764379807304 | 0.02694142645231127 |

TWFE (feols) 0.8711820033888621 (the weighted sum, 0.871182003388858). Type shares: treated vs
untreated 0.7748466503628272 (estimate 1.0689141319470727), earlier vs later treated
0.11321813735316523 (0.92324697836209), later vs earlier treated 0.11193521228400756
(-0.5502364557691009).

**Test**: `test_validate_bacon_twfe_identity` (registry `validation/methods/causal_did_cov.json`,
method `bacon_decomposition`): every component estimate and weight, the overall, the type shares
and type estimates at `Tol::coef(1200)` (1e-10 relative); no warning.

**Fixture tests with R values** (not registered):
- `test_bacon_overall_equals_within_twfe`: on the 24-unit fixture the overall equals a within
  estimator computed in the test and `feols` 1.9787499999999989 (1e-12), and the nine weights
  equal bacondecomp's (0.05, 0.10, 0.15, 0.10, 0.10, 0.20, 0.10, 0.05, 0.15);
- `test_validate_bacon_heterogeneous_effects_forbidden_comparisons`: overall against the within
  estimator and `feols` 3.2335937499999998, weights 0.125 / 0.3125 / 0.1875 / 0.375;
- `test_bacon_components_match_bacondecomp`: estimates, windows, weights (0.14285714285714285,
  0.2857142857142857, 0.21428571428571427, 0.35714285714285715) and `feols` 1.2801817789851873;
- `test_bacon_unbalanced_components_pinned` (2x2 estimates and counts on an unbalanced panel,
  against the 2x2 definition in R) and `test_bacon_unbalanced_panel_warns` (the warning).

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

## Missing values and column types

`bacon_decomp` (MCP `bacon_decomp`) reads the outcome, unit, time and treatment columns through
one `ColumnRequest` (in that order):

- a row with a missing value (null or NaN) in any of them is dropped from all of them, and
  `BaconDecompResult::missing` and the last Display line report it;
- unit ids may be integers, fractional numbers, strings, categoricals or dates; the time column
  holds whole numbers or dates (a fractional time is an error naming the row, where it was
  truncated before);
- a text outcome is `NonNumericColumn`, an absent column `ColumnNotFound`.

The 2x2 components are unchanged for numeric ids. Fixture tests, not R-pinned
(`crates/p2a-core/src/econometrics/bacon.rs`): `test_na_bacon_string_unit_ids`,
`test_na_bacon_omits_missing_rows`, `test_na_bacon_errors_name_the_column`. The method is
R-pinned through the case `bacon_twfe` above.

## Validation Status

| Test | Status | Notes |
|------|--------|-------|
| Basic decomposition | PASS | Weights sum to 1, all comparison types present |
| Comparison types | PASS | Identifies all three types correctly |
| Weight distribution | PASS | Non-negative weights, type sums match total |
| Estimate reasonableness | PASS | Estimates within expected range |

## Known Limitations

1. **Assumes no treatment reversals**: Once treated, units remain treated
2. **Balanced panels**: on an unbalanced panel the weighted sum is not the TWFE coefficient (a
   warning says so; bacondecomp refuses such panels)
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
