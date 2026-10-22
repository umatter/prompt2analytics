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

With H = I the exact problem (`balance_tol = 0`) is the Euclidean projection of the origin
onto {w : A w = b, w >= l}. With a_i = [x_i, 1] / n for the n reweighted units and
b = [target; 1] (so the weights sum to n), its dual has only k + 1 variables mu:

```
w_i(mu) = max(l, a_i . mu)
g(mu)   = b . mu + sum_i h(a_i . mu),   h(z) = -z^2/2 (z >= l),   l^2/2 - l z (z < l)
grad g  = r(mu) = b - sum_i w_i(mu) a_i
```

The concave, piecewise quadratic dual is maximised by a semismooth Newton method:

```
mu_0 = (A A')^-1 b                   Schur complement: w = A'(A A')^-1 b, the optimum
                                     when no bound binds
S    = {i : a_i . mu > l}            units off the bound
M    = sum_{i in S} a_i a_i'         (k+1) x (k+1); pseudo-inverse if singular
mu  <- mu + t M^-1 r(mu)             t = 1 when the step keeps S, otherwise Armijo
                                     backtracking on g
```

Each step is one (k+1) x (k+1) solve and O(n k^2) work, with O(n + k^2) memory; no n x n
matrix is formed. The bound w >= l holds exactly and exact balance is reached to rounding.
`lambda` is reported as -mu.

Infeasible exact balance is an error (`InvalidSpecification`, suggesting
`balance_tol > 0`), as R `sbw` refuses it too: with `min_weight >= 0` a target mean outside
the reweighted group's range is refused up front, and otherwise a dual that does not converge
(no ascent direction, a failed line search, diverging multipliers or `max_iter` steps) means
that no feasible weights exist.

Before October 2026 the exact path inverted the dense (n+k+1) x (n+k+1) system of the
equality-constrained problem, then clipped negative weights at `min_weight` and renormalised.
That is O(n^3) time and O(n^2) memory, and whenever a bound was active the clipped weights
were neither the QP optimum nor exactly balanced (on Test Case 1 below: control 1 weight
3.3271 instead of 6.0244, ATT 0.5825 instead of 0.5553). When no bound binds, the new solver
returns the old output to floating-point noise, `lambda` included.

### Approximate Balance Solution

When exact balance is infeasible, we use penalized QP:

```
minimize:    (1/2) * w'w + ρ * ||A*w - b||²
subject to:  w_i >= l
```

Solved via iterative projected gradient descent.

## Reference Implementation

- **R Package**: sbw (Zubizarreta et al., 2020), cross-checked against quadprog
- **URL**: https://cran.r-project.org/package=sbw
- **Version**: sbw 1.2 (ATT value below); quadprog 1.5.8 on R 4.3.3 (weights and multipliers, re-run 2026-10-22)

## Test Cases

### Test Case 1: Exact Balance (ATT)

**Synthetic Data** (`create_test_data()` in `crates/p2a-core/src/econometrics/sbw.rs`):
- n = 30 observations (10 treated, 20 control)
- k = 2 covariates; treated means (0.66, 0.55), control ranges cover them
- Outcome y = 1 + 2 x1 - x2 + 0.5 d + e (e fixed, 2 decimals), for the ATT

With exact balance and w >= 0 the optimum is rational: in the `normalize_to_n` scale the
control weights are

```
[2464, 0, 500, 2288, 0, 0, 0, 0, 500, 0, 0, 1096, 0, 1008, 0, 0, 0, 324, 0, 0] / 409
```

They sum to 20, reproduce the treated means exactly, and satisfy the KKT conditions.

**R Code for Validation**:
```r
library(quadprog)

x1 <- c(0.6, 0.7, 0.65, 0.75, 0.6, 0.7, 0.55, 0.8, 0.65, 0.6,
        0.3, 0.5, 0.4, 1.0, 0.35, 0.9, 0.3, 1.2, 0.4, 0.5,
        0.35, 0.6, 0.45, 0.95, 0.32, 0.85, 0.28, 1.1, 0.42, 0.55)
x2 <- c(0.5, 0.6, 0.55, 0.6, 0.5, 0.55, 0.5, 0.65, 0.55, 0.5,
        0.3, 0.4, 0.35, 0.8, 0.3, 0.7, 0.25, 0.9, 0.35, 0.4,
        0.3, 0.5, 0.38, 0.75, 0.28, 0.65, 0.22, 0.85, 0.35, 0.45)
d  <- c(rep(1, 10), rep(0, 20))
y  <- c(2.32, 2.22, 2.30, 2.29, 2.29, 2.37, 2.04, 2.55, 2.22, 2.27,
        1.21, 1.64, 1.56, 2.15, 1.48, 1.98, 1.38, 2.56, 1.38, 1.61,
        1.50, 1.68, 1.42, 2.20, 1.45, 2.01, 1.41, 2.27, 1.51, 1.59)
dat <- data.frame(d = d, y = y, x1 = x1, x2 = x2)

# Exact QP in the n scale: min 1/2 w'w  s.t.  [Xc'; 1'] w / n = [tm; 1],  w >= 0
Xc <- cbind(x1[d == 0], x2[d == 0]); tm <- c(mean(x1[d == 1]), mean(x2[d == 1])); n <- 20
qp <- solve.QP(diag(n), rep(0, n), cbind(t(rbind(t(Xc), 1)) / n, diag(n)),
               c(tm, 1, rep(0, n)), meq = 3)
w <- qp$solution
print(w * 409)                       # 2464 0 500 2288 0 0 0 0 500 0 0 1096 0 1008 0 0 0 324 0 0

# Signed multipliers from the active set (quadprog reports |mu| for equality constraints)
A <- rbind(t(Xc), 1) / n; S <- w > 1e-9
mu <- solve(A[, S] %*% t(A[, S]), c(tm, 1))
print(-mu, digits = 15)              # lambda = -mu

# ATT of the exact optimum
print(mean(y[d == 1]) - sum(w * y[d == 0]) / sum(w), digits = 15)

# R sbw with the quadprog solver (bal_tol = 0 exactly is refused by sbw: "no solution from
# the solver quadprog", because the two-sided inequalities collapse; 1e-8 is the closest)
s <- sbw::sbw(dat, ind = "d", out = "y",
              bal = list(bal_cov = c("x1", "x2"), bal_alg = FALSE, bal_tol = 1e-8, bal_std = "group"),
              sol = list(sol_nam = "quadprog"), par = list(par_est = "att"))
sbw::estimate(s)                     # ATT 0.555337388135859
```

**Expected Results** (asserted by `test_sbw_exact_matches_r_sbw`):

| Quantity | Expected | Tolerance |
|----------|----------|-----------|
| Treated weights | 1 | 1e-12 |
| Control weights | [2464, 0, 500, 2288, 0, 0, 0, 0, 500, 0, 0, 1096, 0, 1008, 0, 0, 0, 324, 0, 0] / 409 | 1e-9 |
| Converged | yes | |
| Max balance violation | 0 | < 1e-10 |
| lambda | [3172.6161369150, -4424.4498777446, 255.0611246939] | 1e-8 relative |
| ATT (exact QP optimum) | 0.555337408312916 | 1e-10 |
| ATT (R sbw 1.2, bal_tol 1e-8) | 0.555337388135859 | 1e-6 |

Further exact-balance checks in the same module:
- `test_sbw_exact_matches_dense_kkt_when_no_bound_binds`: 240 LCG controls, no bound active;
  the weights equal the old dense-KKT solution to 1e-12 and `lambda` to 1e-8 relative.
- `test_sbw_exact_kkt_certificate_with_binding_bounds`: the same controls with bounds active
  for `min_weight` 0, 0.1 and -0.5; an optimality certificate (w_i = max(l, z_i)) on every
  unit, and the maximum weight, sum of squared weights and number of units at the bound
  against `quadprog::solve.QP` (R 4.3.3, quadprog 1.5.8).
- `test_sbw_exact_infeasible_is_an_error`: ATC on the Test Case 1 data is infeasible (the
  control mean of x2, 0.474, is below every treated x2) and is refused, as by R sbw.
- `test_sbw_exact_large_n_is_tractable`: 2,000 treated and 20,000 controls with binding bounds.

### Test Case 2: Approximate Balance

**Same data as Test 1, but with balance tolerance**:
```r
s_approx <- sbw::sbw(dat, ind = "d", out = "y",
                     bal = list(bal_cov = c("x1", "x2"), bal_alg = FALSE, bal_tol = 0.1, bal_std = "group"),
                     sol = list(sol_nam = "quadprog"), par = list(par_est = "att"))
```

**Expected Results**:
- Max |Std.Diff| after weighting: < 0.2
- Weights non-negative

The Rust approximate solver minimises a penalised objective by projected gradient descent
rather than solving R's tolerance-constrained QP, so it is checked against these properties,
not against R's weights.

## Validation Status

| Test Case | Status | Tolerance |
|-----------|--------|-----------|
| Exact Balance ATT | Pinned to quadprog / R sbw | see Test Case 1 |
| Exact balance, no binding bound | Equals old dense solution | weights 1e-12 |
| Exact balance, binding bounds | KKT certificate + quadprog | 1e-9 |
| Infeasible exact balance | Error, as in R sbw | |
| Approximate Balance | Implemented | max_std_diff < 0.2 |
| ATC Estimand | Equals ATT with swapped labels | 1e-12 |
| ESS Computation | Implemented | exact formula |
| Balance Statistics | Implemented | std_diff, var_ratio |

## Rust Implementation Notes

1. **Exact Balance**: dual semismooth Newton in the k + 1 constraint dimensions (Schur-complement first step), O(n k^2) per step
2. **Approximate Balance**: Uses penalized gradient descent
3. **Lower bound**: enforced exactly by the dual (w_i = max(l, a_i . mu)), not by clipping
4. **Normalization**: Weights normalized to sum to n_reweighted
5. **Infeasibility**: refused with a message suggesting approximate balance

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
