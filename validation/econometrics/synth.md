# Validation: Synthetic Control Method (SCM)

## Method Overview

The Synthetic Control Method constructs a weighted combination of control (donor) units to create a synthetic counterfactual for a treated unit. Developed by Abadie, Diamond, and Hainmueller.

**Optimization Problem**:
```
W* = argmin_W ||X₁ - X₀W||_V
subject to: w_j ≥ 0, Σw_j = 1
```

Where V is chosen to minimize pre-treatment MSPE.

**Treatment Effect**: τ_t = Y_{1t} - Σ_j w_j Y_{jt}

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| Synth | R | `synth()` | 1.1-6 |
| tidysynth | R | `synthetic_control()` | 0.2.0 |

## Test Cases

### Test 1: Perfect Synthetic Match

**Data Generating Process**:
```
Unit A (treated): pre-treatment = 0.5*B + 0.5*C exactly
3 pre-treatment periods, 3 post-treatment periods
Treatment effect = 5 units starting at t=4
```

**R Code (tidysynth)**:
```r
library(tidysynth)
library(dplyr)

# Create panel data
set.seed(42)
panel <- tibble(
  unit = rep(c("A", "B", "C"), each = 6),
  time = rep(1:6, 3),
  outcome = c(
    # Unit A: pre = 0.5*B + 0.5*C, post = pre + 5
    10, 11, 12, 18, 19, 20,  # A (treatment effect = 5 at t>=4)
    8, 10, 12, 14, 16, 18,   # B
    12, 12, 12, 12, 12, 12   # C
  ),
  x1 = c(
    5, 5, 5, 5, 5, 5,  # A: mean = 5 = 0.5*4 + 0.5*6
    4, 4, 4, 4, 4, 4,  # B: mean = 4
    6, 6, 6, 6, 6, 6   # C: mean = 6
  )
)

# Run synthetic control
synth_out <- panel %>%
  synthetic_control(
    outcome = outcome,
    unit = unit,
    time = time,
    i_unit = "A",
    i_time = 4,
    generate_placebos = FALSE
  ) %>%
  generate_predictor(time_window = 1:3, x1 = mean(x1)) %>%
  generate_weights(optimization_window = 1:3) %>%
  generate_control()

# Extract weights
synth_out %>% grab_unit_weights()
# Expected: B ≈ 0.5, C ≈ 0.5

# Extract treatment effects
synth_out %>% grab_synthetic_control()
```

**Results Comparison**:

| Statistic | R (tidysynth) | p2a Rust | Tolerance |
|-----------|---------------|----------|-----------|
| Weight B | ~0.5 | ~0.5 | 0.1 |
| Weight C | ~0.5 | ~0.5 | 0.1 |
| Pre-RMSPE | ~0 | ~0 | 0.5 |
| Avg Effect | ~5.0 | ~5.0 | 0.5 |

**Rust Test**: `crates/p2a-core/src/econometrics/synth.rs::tests::test_basic_synth`

---

### Test 2: California Tobacco Control (Classic Example)

This is the canonical example from Abadie et al. (2010).

**Data**: California cigarette sales 1970-2000, treatment in 1988 (Proposition 99)

**R Code (Synth package)**:
```r
library(Synth)
data(synth.data)

# Prepare data matrices
dataprep.out <- dataprep(
  foo = synth.data,
  predictors = c("lnincome", "beer", "age15to24", "retprice"),
  predictors.op = "mean",
  time.predictors.prior = 1980:1988,
  special.predictors = list(
    list("cigsale", 1988, "mean"),
    list("cigsale", 1980, "mean"),
    list("cigsale", 1975, "mean")
  ),
  dependent = "cigsale",
  unit.variable = "unit.num",
  time.variable = "year",
  treatment.identifier = 3,  # California
  controls.identifier = c(1,2,4:39),
  time.optimize.ssr = 1970:1988,
  time.plot = 1970:2000
)

# Run synth
synth.out <- synth(data.prep.obj = dataprep.out)

# Results
synth.tables <- synth.tab(dataprep.res = dataprep.out, synth.res = synth.out)
print(synth.tables)

# Main donors: Utah (~0.33), Nevada (~0.24), Montana (~0.16), Colorado (~0.16)
# Treatment effect: ~-20 packs by 2000
```

**Expected Results**:

| Statistic | R (Synth) | p2a Rust | Tolerance |
|-----------|-----------|----------|-----------|
| Top donor | Utah (~0.33) | Similar | - |
| RMSPE (pre) | ~1.8 | Similar | 0.5 |
| Effect 2000 | ~-20 | Similar | 5 |

**Note**: Exact replication requires the `synth.data` dataset.

---

### Test 3: QP Solver Validation

**Verify simplex-constrained QP solver**:

Minimize: ||x||² subject to Σx = 1, x ≥ 0
Expected: x = [0.5, 0.5] for n=2

**R Code**:
```r
library(quadprog)

# QP: min 0.5 x'Dx + d'x
# s.t. A'x >= b

D <- diag(2)
d <- rep(0, 2)

# Equality: sum(x) = 1 → Aeq'x = 1
# Inequality: x >= 0

Amat <- cbind(c(1, 1), diag(2))  # Equality + bounds
bvec <- c(1, 0, 0)

sol <- solve.QP(D, d, Amat, bvec, meq = 1)
sol$solution
# [1] 0.5 0.5
```

**Rust Test**: `crates/p2a-core/src/econometrics/synth.rs::tests::test_qp_solver`

---

### Test 4: Placebo Inference

**R Code (tidysynth)**:
```r
library(tidysynth)

# Use built-in smoking dataset
data(smoking)

synth_out <- smoking %>%
  synthetic_control(
    outcome = cigsale,
    unit = state,
    time = year,
    i_unit = "California",
    i_time = 1988,
    generate_placebos = TRUE
  ) %>%
  generate_predictor(time_window = 1980:1988,
                     lnincome = mean(lnincome),
                     retprice = mean(retprice),
                     age15to24 = mean(age15to24)) %>%
  generate_predictor(time_window = 1984:1988, cigsale = mean(cigsale)) %>%
  generate_weights(optimization_window = 1970:1988) %>%
  generate_control()

# P-value from placebo tests
synth_out %>% grab_significance()
# California should rank high (low p-value)
```

**Validation Criteria**:
- RMSPE ratio for treated unit > most placebos
- P-value calculated as rank / n_units

---

## Numerical Precision Summary

| Test Case | n_donors | Weight Precision | Effect Precision |
|-----------|----------|------------------|------------------|
| Simple 2x3 | 2 | < 0.1 | < 0.5 |
| California | 38 | - | < 5 |
| QP solver | n/a | < 0.01 | n/a |

## Known Differences

1. **V Optimization**: R Synth uses Nelder-Mead; we use coordinate descent. Results may differ slightly.
2. **QP Solver**: R uses quadprog (Goldfarb-Idnani); we use Frank-Wolfe projected gradient.
3. **Predictor Aggregation**: Must match time windows exactly for replication.
4. **Numerical Tolerance**: Different tolerances may yield slightly different weights.

## Running the Tests

```bash
# Run synth validation tests
cargo test -p p2a-core -- synth::tests

# Run with output
cargo test -p p2a-core -- synth --nocapture
```

## Running R Comparison

```bash
# From validation/r_comparison directory
Rscript synth_comparison.R
```

## References

- Abadie, A. & Gardeazabal, J. (2003). "The Economic Costs of Conflict: A Case Study of the Basque Country." *American Economic Review*, 93(1), 112-132.
- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies." *JASA*, 105(490), 493-505.
- Abadie, A. (2021). "Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects." *Journal of Economic Literature*, 59(2), 391-425.
