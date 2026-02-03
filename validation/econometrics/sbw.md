# Stable Balancing Weights (SBW) Validation

## Method Description

Stable Balancing Weights (SBW) directly optimizes for covariate balance rather than modeling the propensity score. The method finds weights that minimize variance while achieving exact or approximate balance on covariate moments.

## Mathematical Formulation

For ATT estimation with control weights w_i, SBW solves:

```
minimize:    (1/2) * w'Hw + c'w  (variance of weights)
subject to:  A * w = b           (balance constraints)
             w_i >= l            (lower bound on weights)
```

Where:
- H = I (identity matrix) for variance minimization
- c = 0 (no linear term for pure variance minimization)
- A = covariate matrix with normalization constraint
- b = target means (treated group means for ATT)
- l = minimum weight (default 0 for non-negativity)

### Exact Balance Solution

For equality-constrained QP, the KKT conditions give:

```
[H    A'][w]   [-c]
[A    0 ][λ] = [b ]
```

Solving this linear system yields optimal weights and Lagrange multipliers.

### Approximate Balance Solution

When exact balance is infeasible, we use penalized QP:

```
minimize:    (1/2) * w'w + ρ * ||A*w - b||²
subject to:  w_i >= l
```

Solved via iterative projected gradient descent.

## Reference Implementation

- **R Package**: sbw (Zubizarreta et al., 2020)
- **URL**: https://cran.r-project.org/package=sbw
- **Version**: 1.1.6

## Test Cases

### Test Case 1: Exact Balance (ATT)

**Synthetic Data**:
- n = 30 observations (10 treated, 20 control)
- k = 2 covariates with known imbalance
- Treated group has higher covariate means

**R Code for Validation**:
```r
library(sbw)

# Create test data
set.seed(42)
n_treat <- 10
n_ctrl <- 20

# Treated: higher x1, x2 means (~1.5, ~0.9)
x1_treat <- c(1.2, 1.5, 1.3, 1.8, 1.4, 1.6, 1.1, 1.7, 1.5, 1.4)
x2_treat <- c(0.8, 0.9, 0.7, 1.0, 0.85, 0.95, 0.75, 1.1, 0.9, 0.85)

# Control: lower x1, x2 means (~0.5, ~0.35)
x1_ctrl <- c(0.3, 0.5, 0.4, 0.7, 0.2, 0.6, 0.1, 0.8, 0.4, 0.5,
             0.35, 0.55, 0.45, 0.75, 0.25, 0.65, 0.15, 0.85, 0.45, 0.55)
x2_ctrl <- c(0.3, 0.4, 0.35, 0.5, 0.25, 0.45, 0.2, 0.55, 0.35, 0.4,
             0.32, 0.42, 0.37, 0.52, 0.27, 0.47, 0.22, 0.57, 0.37, 0.42)

treatment <- c(rep(1, n_treat), rep(0, n_ctrl))
x1 <- c(x1_treat, x1_ctrl)
x2 <- c(x2_treat, x2_ctrl)

data <- data.frame(treatment = treatment, x1 = x1, x2 = x2)

# Run SBW
result <- sbw(
  data,
  ind = "treatment",
  bal = list(x1 = c("mu"), x2 = c("mu")),
  par = list(par = "att")
)

# Check balance
print(summary(result))
print(result$weights)
```

**Expected Results**:
- Treated weights: all 1.0
- Control weights: positive, summing to 20
- Max |Std.Diff| after weighting: < 0.05
- Convergence: Yes

### Test Case 2: Approximate Balance

**Same data as Test 1, but with balance tolerance**:
```r
result_approx <- sbw(
  data,
  ind = "treatment",
  bal = list(x1 = c("mu"), x2 = c("mu")),
  par = list(par = "att"),
  tols = c(0.1, 0.1)  # Allow 0.1 tolerance
)
```

**Expected Results**:
- Max |Std.Diff| after weighting: < 0.2
- Lower weight variance than exact balance
- Higher ESS than exact balance

## Validation Status

| Test Case | Status | Tolerance |
|-----------|--------|-----------|
| Exact Balance ATT | Implemented | max_std_diff < 0.05 |
| Approximate Balance | Implemented | max_std_diff < 0.2 |
| ATC Estimand | Implemented | treated weights = 1 |
| ESS Computation | Implemented | exact formula |
| Balance Statistics | Implemented | std_diff, var_ratio |

## Rust Implementation Notes

1. **Exact Balance**: Uses Lagrangian KKT system solver
2. **Approximate Balance**: Uses penalized gradient descent
3. **Weight Projection**: Ensures non-negativity constraint
4. **Normalization**: Weights normalized to sum to n_reweighted

## References

1. Zubizarreta, J.R. (2015). "Stable Weights that Balance Covariates for
   Estimation with Incomplete Outcome Data". *Journal of the American
   Statistical Association*, 110(511), 910-922.
   DOI: 10.1080/01621459.2015.1023805

2. Zubizarreta, J.R., Cerdeiro, D.A., & Kelz, R.R. (2020). sbw: Stable
   Balancing Weights for Causal Inference and Estimation with Incomplete
   Outcome Data. R package. https://cran.r-project.org/package=sbw

3. Chan, K.C.G., Yam, S.C.P., & Zhang, Z. (2016). "Globally Efficient
   Non-parametric Inference of Average Treatment Effects by Empirical
   Balancing Calibration Weighting". *JRSS-B*, 78(3), 673-700.
