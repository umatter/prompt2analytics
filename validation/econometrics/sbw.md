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
matrix is formed. The bound w >= l holds exactly and exact balance is reached to rounding;
the residual r is accumulated with compensated (Neumaier) sums. `lambda` is reported as -mu.

The solve converges when ||r||_inf <= 1e-14 (1 + ||b||_inf), or <= `tolerance` (1 + ||b||_inf)
after a full step that kept S. If no ascent direction exists, the line search fails or
`max_iter` steps (mu_0 counts as one) are used up while ||r||_inf <= `tolerance` (1 + ||b||_inf),
the solve stops there as converged. Only outside that tolerance do these cases mean an error.

Infeasible exact balance is an error (`InvalidSpecification`, suggesting
`balance_tol > 0`), as R `sbw` refuses it too: with `min_weight >= 0` a target mean outside
the reweighted group's range is refused up front, and otherwise no ascent direction or a
failed line search outside the tolerance, or diverging multipliers, mean that no feasible
weights exist. Running out of `max_iter` outside the tolerance is a `ConvergenceFailure`
(`iterations = max_iter`, `last_change` the balance residual); it says nothing about
feasibility. A target mean that overflows is refused up front, and a non-finite Gram matrix
or multiplier (covariates too large for f64) is a `Computation` error, never a converged
result.

Before October 2026 the exact path inverted the dense (n+k+1) x (n+k+1) system of the
equality-constrained problem, then clipped negative weights at `min_weight` and renormalised.
That is O(n^3) time and O(n^2) memory, and whenever a bound was active the clipped weights
were neither the QP optimum nor exactly balanced (on Test Case 1 below: control 1 weight
3.3271 instead of 6.0244, ATT 0.5825 instead of 0.5553). When no bound binds, the new solver
returns the old output to floating-point noise, `lambda` included.

### Approximate Balance Solution

With `balance_tol > 0` each weighted covariate mean must lie within delta_j of its target,
where delta_j = `balance_tol` (`balance_tol_scale = Raw`, the default) or `balance_tol` times
the covariate's full-sample standard deviation (`StdDev`, n - 1 denominator, Zubizarreta
2015). The band QP, in the n scale (m reweighted units),

```
minimise   (1/2) ||w||^2
subject to (1/m) sum_i w_i = 1,   |(1/m) sum_i w_i x_ij - t_j| <= delta_j,   w_i >= l
```

is solved exactly through its dual in k + 1 variables mu = (y, nu). With covariates centred
on the band and scaled by the group SD (a_i = [xs_i, 1]/m, half-widths r_j), the dual
objective is

```
F(mu) = sum_i h(a_i . mu) - sum_j r_j |y_j| + nu,   h(z) = -z^2/2 (z >= l), l^2/2 - l z (z < l)
w_i(mu) = max(l, a_i . mu)
```

-F is minimised by proximal Newton (Lee, Sun & Saunders 2014): each step minimises the
quadratic model plus the l1 term exactly by the feature-sign method (Lee, Battle, Raina & Ng
2007) and takes an Armijo step. The (k+1) x (k+1) Hessian `sum_{w_i > l} a_i a_i'` is
regularised only when singular; memory is O(n k). The solve stops when the KKT residual is
below 1e-14 (1 + max r_j), or below `tolerance` (1 + max r_j) after an unregularised full step
that kept the set of units off the bound; running out of `max_iter` returns an unconverged
result with a note.

Infeasibility is reported, never returned as weights:
- **Range check** (with `min_weight >= 0`): a band [t_j - delta_j, t_j + delta_j] that misses
  the reweighted group's range of covariate j cannot be met by non-negative weights.
- **Dual certificate**: a dual value above U = ((m - (m-1) l)^2 + (m-1) l^2)/2, the largest
  primal objective on {w >= l, sum w = m}, proves by weak duality that no feasible weights
  exist (R `solve.QP`: "constraints are inconsistent").
Both are `InvalidSpecification` errors asking for a larger `balance_tol`.

`balance_penalty` is not used: the QP has no penalty parameter. Only the default 1000 is
accepted; any other value is an `InvalidSpecification` error that says to omit it (decision
WP23-10).

### Estimands

ATT reweights the controls to the treated means and ATC the treated to the control means; the
other group keeps weight 1. ATE solves one problem per group, each reweighted to the pooled
means; `objective_treated` and `objective_control` report the two objectives
(`objective_value` is their sum), `lambda_treated` the treated problem's multipliers, and the
balance tables compare the weighted treated with the weighted controls. With
`normalize_to_n` each reweighted group sums to its own size, otherwise to 1.

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
- `test_sbw_exact_max_iter_exhaustion_is_not_infeasibility`: Test Case 1 with `max_iter = 2`
  and `tolerance = 1e-3` is a `ConvergenceFailure` with `iterations = 2` (not the
  infeasibility verdict) whose message shows the threshold tolerance (1 + ||b||_inf) and the
  configured tolerance; with `max_iter = 4` and `tolerance = 1e-2` the fourth step is within
  tolerance and the solve stops as converged.
- `test_sbw_exact_overflow_is_an_error_not_a_converged_fallback`: an overflowing target mean,
  a target that overflows the multipliers and covariates of order 1e200 are errors. The
  first (with `min_weight < 0`, which skips the range pre-check) used to return uniform
  weights reported as converged; the last (default configuration) used to be a
  singular-matrix error that blamed the constraints.

### Test Case 2: weighting_sim (ATT, ATC, ATE; exact and approximate)

R code: `validation/r/cases/causal_weighting.R`, case `weights_sbw`, on
`validation/datasets/generated/weighting_sim.csv` (300 units, 151 treated, covariates x1, x2,
x3). Each problem is the SBW QP solved with `quadprog::solve.QP` (R `sbw` and `optweight` are
not installed); approximate balance uses delta = 0.1 full-sample SD.

| Quantity | R value |
|----------|---------|
| ATT exact: objective, zero weights, effect | 0.003840102375577727, 20, 1.5617368270324565 |
| ATC exact: objective, zero weights, effect | 0.004486478085953354, 24, 1.3128019372669482 |
| ATE exact: objective treated / control, effect | 0.0010394911183649218 / 0.0009247533745617431, 1.4307248209844157 |
| ATT approximate: objective, zero weights, effect | 0.0023680772629313743, 7, 1.5582772197757029 |
| ATC approximate: objective, zero weights, effect | 0.0027409649018704484, 13, 1.3451918221996249 |
| ATE approximate: objective treated / control, effect | 0.0003886807347007872 / 0.0003511886754797501, 1.4663089969510448 |

Tests (registry `validation/methods/causal_weighting.json`, entry `sbw`):
`test_validate_sbw_exact`, `test_validate_sbw_approx`, `test_validate_sbw_atc`,
`test_validate_sbw_ate`. Weights (heads of 12 or 8) at 1e-8 relative + 1e-12 absolute,
objectives, weighted means and effects at `Tol::coef(n)`, zero-weight counts exact.
`test_validate_sbw_approx` also checks every weighted mean within 0.1 SD of its target.

Property tests (no case): `test_sbw_estimands_differ`, `test_sbw_approx_scale_invariant`
(data x 1e-6 and x 1e6 give the same weights in similar steps), `test_sbw_approx_infeasible_is_error`
(the range check and the dual certificate), `test_sbw_balance_penalty_is_refused` and
`test_sbw_approx_matches_quadprog_on_test_data` (Test Case 1 data, ATT, `balance_tol = 0.05`
raw, against `quadprog::solve.QP` to 1e-10; the x2 lower band bound is active).

## Validation Status

| Test Case | Status | Tolerance |
|-----------|--------|-----------|
| Exact Balance ATT | Pinned to quadprog / R sbw | see Test Case 1 |
| Exact balance, no binding bound | Equals old dense solution | weights 1e-12 |
| Exact balance, binding bounds | KKT certificate + quadprog | 1e-9 |
| Infeasible exact balance | Error, as in R sbw | |
| Approximate balance (ATT, ATC, ATE) | Pinned to quadprog (weights_sbw) | 1e-8 relative |
| Exact ATC, ATE | Pinned to quadprog (weights_sbw) | 1e-8 relative |
| Infeasible approximate balance | Error (range check, dual certificate) | |
| ESS Computation | Implemented | exact formula |
| Balance Statistics | Implemented | std_diff, var_ratio |

## Rust Implementation Notes

1. **Exact Balance**: dual semismooth Newton in the k + 1 constraint dimensions (Schur-complement first step), O(n k^2) per step
2. **Approximate Balance**: the band QP through its dual by proximal Newton with an exact feature-sign subproblem, O(n k) memory
3. **Lower bound**: enforced exactly by the dual (w_i = max(l, a_i . mu)), not by clipping
4. **Normalization**: each reweighted group sums to its own size (`normalize_to_n`) or to 1
5. **Infeasibility**: exact balance is refused with a message suggesting approximate balance; approximate balance with one asking for a larger `balance_tol`

## Missing values and column types

Missing values (null or NaN) in the treatment or covariates are refused, with
"Column 'x' contains k missing value(s) (null or NaN). Drop or impute them first."
(`EconError::NullValues`, naming the column and the count). A text column gives
`NonNumericColumn`, infinite values `NonFiniteValues`, an absent column `ColumnNotFound`; a
column that exists is never reported as not found. Int64 and Boolean 0/1 treatments are
read as numbers and give the same result as Float64. Results on complete data are
unchanged.

R: sbw is not installed here; not verified.

Tests: `test_validate_na_sbw_columns` (case `causal_mediation_na`, registry
`validation/methods/causal_na.json`, id `weighting_na_refusal`).

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
