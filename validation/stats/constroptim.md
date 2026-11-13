# Constrained Optimization Validation

## Method Overview

`constrOptim` performs minimization of a function subject to linear inequality constraints using an adaptive barrier algorithm. It wraps unconstrained optimization methods (Nelder-Mead, BFGS) with a logarithmic barrier function.

**Key Parameters:**
- `theta`: Initial parameter values
- `f`: Objective function to minimize
- `grad`: Optional gradient function
- `ui`: Constraint matrix (k x p, where k = number of constraints)
- `ci`: Constraint vector (length k)
- `method`: Optimization method (Nelder-Mead or BFGS)

Constraints: `ui %*% theta >= ci`

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `constrOptim()` | Reference implementation |
| scipy | `scipy.optimize.minimize(method='SLSQP')` | Alternative |

## Algorithm

R's adaptive barrier (Lange), with R's defaults `mu = 1e-4`, `outer.iterations = 100`,
`outer.eps = 1e-5`:

1. At the outer iterate θₖ minimise `R(θ, θₖ) = f(θ) - mu Σᵢ [(uᵢθₖ - cᵢ) log(uᵢθ - cᵢ) - uᵢθ]`
   (NaN outside the region) with BFGS (gradient given) or Nelder-Mead; mu stays fixed.
2. Stop when `|R - R_old| < (1e-3 + |R|) outer.eps` (the inner result is returned); otherwise move to
   the inner solution and stop with code 11 if f increased.
3. Code 7 when the outer iterations run out; code 1 when the last inner optimisation hit its limit.
   `value` is f at `par`, `barrier_value` the last R minus `value`. No point is projected.

Inner methods: BFGS without a gradient runs Nelder-Mead (R's default for a missing gradient; R's
explicit `method = "BFGS"` without one stops), and L-BFGS-B runs BFGS; both say so in
`ConstrOptimResult::notes`.

## Test Cases

### Test Case 1: Simple 2D Quadratic

**R Code:**
```r
# Minimize (x-2)^2 + (y-3)^2 subject to x + y >= 1
f <- function(x) (x[1] - 2)^2 + (x[2] - 3)^2
grad <- function(x) c(2 * (x[1] - 2), 2 * (x[2] - 3))
ui <- matrix(c(1, 1), nrow = 1)
ci <- 1

result <- constrOptim(c(0.5, 0.5), f, grad, ui, ci)
print(result$par)
# [1] 2 3  (unconstrained minimum, constraint is not active)
print(result$value)
# [1] 0
```

**Rust tests:**
- `stats::constroptim::tests::test_validate_constr_optim_quadratic` (`crates/p2a-core/src/stats/constroptim.rs`): x² + y² with x + y ≥ 1 from (1, 1); `quadratic_par` (0.500000000151549 twice) and `quadratic_value` (0.500000000303098), R-pinned (case `stat_constroptim_legacy`, `Tol::iterative(1e-4)`; the convergence code is pinned exactly, the outer iterations are a path quantity).
- `stats::constroptim::tests::test_validate_constr_optim_nelder_mead`: Nelder-Mead on (x - 1)² + (y - 2)² with x, y ≥ 0 from (0.5, 0.5); `nelder_mead_par` (1.0000008855605316, 2.0000000235991697) at `Tol::iterative(1e-4)` and the convergence code; the value (R 7.8e-13) is zero up to the inner tolerance, so it is checked below 1e-9, not against R's digits.
- `test_constr_optim_codes_seven_and_eleven`, `test_constr_optim_method_substitutions_are_noted` (properties): codes 7 and 11, the notes for BFGS without a gradient and for L-BFGS-B, R's defaults.

### Test Case 2: Active Constraint

**R Code:**
```r
# Minimize (x-0)^2 + (y-0)^2 subject to x + y >= 5
f <- function(x) x[1]^2 + x[2]^2
grad <- function(x) c(2 * x[1], 2 * x[2])
ui <- matrix(c(1, 1), nrow = 1)
ci <- 5

result <- constrOptim(c(3, 3), f, grad, ui, ci)
print(result$par)
# Approximately [2.5, 2.5] - on the constraint boundary
print(result$value)
# Approximately 12.5
```

**Rust tests:**
- `stats::constroptim::tests::test_validate_constr_optim_binding_constraint` (`crates/p2a-core/src/stats/constroptim.rs`): minimise x² subject to x ≥ 1 from 2; `binding_par` (1.0000000025631852) and `binding_value` (1.0000000051263704), R-pinned (case `stat_constroptim_legacy`, `Tol::iterative(1e-4)`; the convergence code is pinned exactly, the outer iterations are a path quantity).

### Test Case 3: Multiple Constraints (Box Constraints)

**R Code:**
```r
# Minimize (x-3)^2 + (y-4)^2 subject to 0 <= x <= 2, 0 <= y <= 2
f <- function(x) (x[1] - 3)^2 + (x[2] - 4)^2
ui <- rbind(diag(2), -diag(2))  # x >= 0, y >= 0, -x >= -2, -y >= -2
ci <- c(0, 0, -2, -2)

result <- constrOptim(c(1, 1), f, NULL, ui, ci)
print(result$par)
# [1] 2 2  (corner of box closest to (3,4))
```

**Rust tests:**
- `stats::constroptim::tests::test_validate_constr_optim_linear_three_constraints` (`crates/p2a-core/src/stats/constroptim.rs`): -x - y with x, y ≥ 0 and x + y ≤ 1 from (0.3, 0.3); `lp3_par` (0.49999999831081493 twice) and `lp3_value` (-0.9999999966216299), R-pinned (case `stat_constroptim_legacy`, `Tol::iterative(1e-4)`; the convergence code is pinned exactly, the outer iterations are a path quantity).

### Test Case 4: Higher Dimensional

**R Code:**
```r
# 10D: minimize sum((x_i - i)^2) subject to x_i >= 0
f <- function(x) sum((x - (0:9))^2)
ui <- diag(10)
ci <- rep(0, 10)

result <- constrOptim(rep(5, 10), f, NULL, ui, ci)
print(result$par)
# Should be approximately 0, 1, 2, ..., 9
```

**Rust tests**: no Rust test runs the 10-dimensional case (not R-pinned; the R-pinned case comes with STAT-62).

## Numerical Precision Summary

| Problem Size | Parameter Tolerance | Value Tolerance |
|-------------|---------------------|-----------------|
| p < 5 | 1e-4 | 1e-6 |
| p = 5-20 | 1e-3 | 1e-5 |
| p > 20 | 1e-2 | 1e-4 |

## Known Differences

1. **Inner optimisers**: p2a's own BFGS (backtracking line search) and Nelder-Mead, not R's `optim`; the
   solutions agree at `Tol::iterative(1e-4)`, function and gradient counts are p2a's.
2. **L-BFGS-B**: not available; BFGS runs, with a note.


## Timing against the previous implementation (Tier 4)

Release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), median of five interleaved runs of a scratch harness at f80b0131 and with this change, on a shared machine (load average 20-60), so differences under about 10% are noise. Preliminary: to be repeated at integration.

| Method | n | f80b0131 (µs) | this change (µs) | ratio |
|---|---|---|---|---|
| x² + y², x + y ≥ 1, BFGS | 2 | 235,006.2 | 1,026.5 | < 0.01 |
| (x - 1)² + (y - 2)², x, y ≥ 0, Nelder-Mead | 2 | 616.1 | 48.7 | 0.08 |

n is the number of parameters. The previous barrier ran 100 outer iterations; R's adaptive barrier stops after a few.

## Performance Notes

- Complexity depends on inner optimizer and number of constraints
- BFGS typically faster with gradient
- Rust implementation 5-20x faster than R for medium-sized problems

## References

1. Lange, K. (2010). Numerical Analysis for Statisticians (2nd ed.). Springer.
2. Boyd, S., & Vandenberghe, L. (2004). Convex Optimization. Cambridge University Press.
3. R Core Team. constrOptim() documentation.
