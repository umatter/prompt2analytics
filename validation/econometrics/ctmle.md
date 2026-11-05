# Validation: Collaborative Targeted Maximum Likelihood Estimation (C-TMLE)

## Method Overview

Collaborative TMLE (van der Laan & Gruber, 2010) extends TMLE by using a data-adaptive procedure to select covariates for the treatment mechanism (propensity score) model. It builds the propensity score collaboratively with the outcome model to minimize the bias-variance tradeoff for the target parameter.

**Key Parameters**:
- `y_col`: Outcome variable
- `treatment_col`: Binary treatment indicator
- `x_cols`: Candidate covariates for propensity score
- `max_covariates`: Maximum covariates to include in propensity model
- `propensity_bounds`: Truncation bounds for propensity scores (default: [0.01, 0.99])

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| ctmle | R | `ctmle()` | 0.1.x |
| tmle | R | `tmle()` (standard TMLE for comparison) | 1.5.x |

## Test Cases

### Test 1: ATE Recovery

**Data Generating Process**:
```
- n = 500
- 5 covariates, only 2 are confounders
- True ATE = 2.0
- Treatment probability depends on X1 and X2
- Outcome depends on treatment, X1, X2
```

**R Code**:
```r
library(ctmle)

set.seed(42)
n <- 500
X <- matrix(rnorm(n * 5), n, 5)

# Only X1 and X2 are confounders
ps <- plogis(0.5 * X[,1] + 0.3 * X[,2])
A <- rbinom(n, 1, ps)

# Outcome with ATE = 2.0
Y <- 2.0 * A + X[,1] + 0.5 * X[,2] + rnorm(n, sd = 0.5)

result <- ctmle(Y = Y, A = A, W = data.frame(X),
                family = "gaussian",
                gn_candidates = seq(1, 5))
result

# ATE estimate should be near 2.0
```

**Rust Tests**:
- `test_ctmle_ate_recovers_dgp`
- `test_ctmle_selection_path_properties`
- `test_ctmle_se_is_ic_variance`
- `test_ctmle_propensity_bounds`
- R-pinned (case `ctmle_cv`): `test_validate_ctmle_cv_logit_rss`, `test_validate_ctmle_cv_linear_rss`,
  `test_validate_ctmle_cv_backward_and_prespecified`, `test_validate_ctmle_criteria_varic`,
  `test_validate_ctmle_criteria_penrss`, `test_validate_ctmle_cv_risk_uses_training_fluctuation`,
  `test_validate_ctmle_prespecified_order_in_arrays_path`

## Tolerance Levels

| Statistic | Tolerance | Notes |
|-----------|-----------|-------|
| ATE | 1.0 | DGP-based, stochastic |
| Propensity scores | [0.01, 0.99] | Must be bounded |
| Selection path | monotonic | Covariates added greedily |
| Influence curve | mean ~0 | Should be mean-zero |

## Running the Tests

```bash
cargo test -p p2a-core -- ctmle::tests::test_validate
```

## Missing values and column types

Missing values (null or NaN) in the outcome, treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: ctmle is not installed here; not verified.

Tests: `test_validate_na_ctmle_columns` (case `causal_mediation_na`, registry `validation/methods/causal_na.json`, id
`weighting_na_refusal`).

## References

- van der Laan, M.J. & Gruber, S. (2010). "Collaborative Double Robust Targeted Maximum Likelihood Estimation." *The International Journal of Biostatistics*, 6(1).
- Ju, C., Schwab, J., & van der Laan, M.J. (2019). "On adaptive propensity score truncation in causal inference." *Statistical Methods in Medical Research*, 28(6), 1741-1760.
