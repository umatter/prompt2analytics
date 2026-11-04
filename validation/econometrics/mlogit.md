# Validation: McFadden Conditional Logit (mlogit)

R reference: `validation/r/cases/discrete_mlogit_asc.R` (T3-WP18).

## Method Overview

The McFadden conditional logit model is used for discrete choice analysis where individuals choose among a set of alternatives. It differs from standard multinomial logit by supporting:

1. **Alternative-specific variables** (e.g., price, quality) - same coefficient across alternatives
2. **Individual-specific variables** (e.g., income, age) - different coefficient for each alternative

**Model Specification**:
```
U_ij = β'x_ij + γ_j'z_i + ε_ij

where:
- U_ij = utility of alternative j for individual i
- x_ij = alternative-specific variables (vary by both i and j)
- z_i = individual-specific variables (constant across alternatives)
- β = generic coefficients (same for all alternatives)
- γ_j = alternative-specific coefficients (different for each j)
- ε_ij ~ Type I extreme value (Gumbel) distribution
```

**Choice Probability**:
```
P(y_i = j) = exp(V_ij) / Σ_k exp(V_ik)
```

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| mlogit | R | `mlogit()` | 1.1-1 |
| survival | R | `clogit()` | 3.5-7 |

## Test Cases

### Test 1: Alternative-Specific Variables Only

**Data Generating Process**:
```
3 choosers, 3 alternatives each
Alternative-specific variable: price
Utility = -0.1 * price + error
```

**R Code (mlogit)**:
```r
library(mlogit)

# Create choice data
panel <- data.frame(
  chooser = rep(1:3, each = 3),
  alt = rep(1:3, 3),
  price = c(10, 15, 20, 12, 18, 22, 8, 14, 16),
  chosen = c(1, 0, 0, 1, 0, 0, 1, 0, 0)
)

# Convert to mlogit format
mdata <- dfidx(panel, choice = "chosen", idx = c("chooser", "alt"))

# Fit model
result <- mlogit(chosen ~ price | 0, data = mdata)
summary(result)
```

**Results Comparison**:

| Statistic | R (mlogit) | p2a Rust | Tolerance |
|-----------|------------|----------|-----------|
| β_price | ~-0.1 | ~-0.1 | 0.05 |
| Log-likelihood | ~-3.3 | ~-3.3 | 0.1 |

**Rust tests** (not R-pinned: no committed R case is compared; the R pin of the conditional logit is `econometrics::discrete::conditional_logit::tests::test_validate_conditional_logit_mixed_logit_data`, below):
- `econometrics::discrete::conditional_logit::tests::test_mlogit_basic` (`crates/p2a-core/src/econometrics/discrete/conditional_logit.rs`; a copy of the same test is `econometrics::discrete::tests::test_mlogit_basic` in `discrete/mod.rs`): 5 choice situations, 3 alternatives, 2 coefficients and a negative cost coefficient.

---

### Test 2: With Individual-Specific Variables

**Description**: Model with both alternative-specific and individual-specific variables.

**R Code**:
```r
# Add individual-specific variable
panel$income <- rep(c(50, 80, 60), each = 3)

mdata <- dfidx(panel, choice = "chosen", idx = c("chooser", "alt"))

# price is alt-specific, income is ind-specific (needs different coef per alt)
result <- mlogit(chosen ~ price | income, data = mdata)
summary(result)
```

**Rust tests** (not R-pinned: no committed R case is compared; the R pin of the conditional logit is `econometrics::discrete::conditional_logit::tests::test_validate_conditional_logit_mixed_logit_data`, below):
- superseded by `econometrics::discrete::conditional_logit::tests::test_validate_mlogit_fishing_asc` (R-pinned on `disc_mlogit_fishing`; section rewritten with the validation docs).

---

### Test 3: R pin on `mixed_logit.csv` (Newton-polished mlogit)

Case `disc_mixed_logit` in [`validation/r/cases/discrete_mixed.R`](../r/cases/discrete_mixed.R)
(registry `validation/methods/discrete_mixed.json`, id `conditional_logit_choice_data`;
300 choice situations x 3 alternatives a, b, c). mlogit's own conditional logit stops with
`|gradient|` ~ 1e-3, 1.5e-5 relative from the MLE, so the case re-solves the same
likelihood with Newton steps on numDeriv's gradient and Hessian (`clogit_*_tight`; SEs from
`numDeriv::hessian`) and records mlogit's estimate as `clogit_coef`.

```r
cl <- mlogit(choice ~ x + w | 0, data = dfidx(d, idx = c("id", "alt"), choice = "choice"))
```

| Key | x | w |
|---|---|---|
| `clogit_coef_tight` | 0.5894383442012155 | -0.5523202005586988 |
| `clogit_se_tight` | 0.0839638977069438 | 0.08344026152700368 |
| `clogit_loglik_tight` | -281.77611853131634 | |
| `clogit_coef` (mlogit's own) | 0.5894296210995991 | -0.5523119773028726 |

**Rust test** (R-pinned): `econometrics::discrete::conditional_logit::tests::test_validate_conditional_logit_mixed_logit_data`:
beta 1e-6 and SEs 1e-5 relative, log-likelihood 1e-6, mlogit's own estimate at 1e-4,
`converged`, 300 situations. The mixed logit with no random coefficient gives the same fit
([mixed_logit.md](mixed_logit.md)).

---

## Missing values

Every column (choice-situation id, alternative id, choice indicator, alternative- and
individual-specific variables) comes from one request with `NaPolicy::Fail`: a missing
cell is refused with `NullValues` naming the column, as R's mlogit refuses it (case
`disc_choice_na`: `mlogit_refuses` = [1, 1, 1] for a missing choice, attribute and id;
mlogit's messages are "impossible to coerce the choice variable to a logical" and "missing
value where TRUE/FALSE needed"). Before, a missing attribute was read as 0 and a missing id
grouped the rows into one situation "".

Not R-pinned tests (`crates/p2a-core/src/econometrics/discrete/conditional_logit.rs`):

- `econometrics::discrete::conditional_logit::tests::test_na_conditional_logit_missing_cells_refused`:
  `run_conditional_logit` and `run_mlogit` refuse a missing choice, attribute and id.
- `econometrics::discrete::conditional_logit::tests::test_na_conditional_logit_int64_attribute`:
  an Int64 attribute gives the fit of its Float64 copy.
- `econometrics::discrete::conditional_logit::tests::test_na_conditional_logit_text_attribute_refused`:
  a text attribute is `NonNumericColumn`.

## Data types

The choice indicator and the attributes may be Float64, any integer type or Boolean. The
ids may be text, integers or floats; situations and alternatives follow R `factor()` order
(numbers numerically, text by byte), so integer alternative ids 2..12 are ordered 2 < 10
and the default reference alternative is the first in that order (it followed string
order). The likelihood is summed over the situations in that order.

## Numerical Precision Summary

| Component | Expected Precision |
|-----------|-------------------|
| β coefficients | Within 0.05 of R |
| γ coefficients | Within 0.1 of R |
| Log-likelihood | Within 0.1 of R |
| Standard errors | Within 10% of R |

## Known Differences

1. **Reference alternative**: the first alternative in R `factor()` order, as mlogit
2. **Convergence criteria**: Slight differences in Newton-Raphson tolerance
3. **Standard errors**: Computed from Hessian inverse, may differ slightly

## Performance Comparison

Benchmarks run on criterion (Rust) with 100 samples each, compared against R mlogit package.

| Dataset Size | Rust (µs) | R (ms) | Speedup |
|--------------|-----------|--------|---------|
| 50 choosers × 3 alts (150 obs) | 95 | 7.2 | **76x** |
| 100 choosers × 3 alts (300 obs) | 156 | 8.0 | **51x** |
| 200 choosers × 4 alts (800 obs) | 401 | 10.3 | **26x** |
| 500 choosers × 3 alts (1500 obs) | 674 | 13.0 | **19x** |
| 1000 choosers × 3 alts (3000 obs) | 1,450 | 20.5 | **14x** |
| 2000 choosers × 5 alts (10000 obs) | 4,754 | 49.9 | **10x** |

The Rust implementation uses Newton-Raphson optimization with:
- Pre-computed feature matrices to avoid repeated data extraction
- Analytical gradients and Hessian for fast convergence
- Symmetry exploitation in Hessian computation
- Numerically stable softmax (log-sum-exp trick)

## References

- McFadden, D. (1974). "Conditional Logit Analysis of Qualitative Choice Behavior."
  In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105-142). Academic Press.
- Train, K. E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.). Cambridge University Press.
- R Package: `mlogit` (Yves Croissant). https://cran.r-project.org/package=mlogit
