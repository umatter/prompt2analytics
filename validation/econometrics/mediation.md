# Causal Mediation Analysis Validation

## Method Overview

IPW-based causal mediation analysis for decomposing treatment effects into natural direct and indirect effects, following Huber (2014).

**p2a Function**: `run_mediation_analysis()`

## What is pinned to R

- **Total effect and rows kept after trimming:** the Hajek IPW formula on `glm(binomial)` propensity
  scores with p2a's trimming rule, written out in R (causalweight is not installed): case
  `medflex_scales` (outcome `yo`, trim 0.05: total effect -0.10767853082261691, 500 rows kept;
  registry `validation/methods/causal_medflex.json`, id `mediation_ipw_t3`) and T2's
  `causal_mediation_na` (below).
- **Bootstrap percentile rule:** SD (n - 1) and the 95% type-7 percentile interval of a fixed
  199-draw vector (`medflex_scales`, `pct_ci` [1.8825599511963476, 20.7409494752895], `pct_sd`
  5.773974522368977; `Tol::coef(199)`).
- **Huber's decomposition (Tier 4):** total, θ(0) (NDE, `direct_effect`), δ(1) (NIE,
  `indirect_effect`), θ(1) (`direct_effect_treated`), δ(0) (`indirect_effect_control`) and the
  rows kept, at trim 0 and 0.05: case `mediation_huber_sim` (n 800, a treatment-mediator
  interaction; registry `validation/methods/causal_effects_t4.json`, id `mediation_huber_t4`;
  `Tol::iterative(1e-8)`, rows exact). At trim 0: total 1.693693779178835, θ(0)
  0.5739670558020316, δ(1) 1.1197267233768033, θ(1) 1.2305951663827461, δ(0)
  0.46309861279608877. Nothing in this page was produced by causalweight or mediation; the
  formulas are written out in R.

Tests (p2a-core, `econometrics::mediation`): `test_validate_mediation_total_effect_medflex_scales`,
`test_validate_mediation_percentile_rule_medflex_scales`, `test_validate_mediation_huber_decomposition` and the T2 tests listed under "R reference
(T2-WP13)". Property tests with no R value: `test_mediation_recovers_dgp` (total effect near the
DGP's 0.7), `test_mediation_decomposition_identity`, `test_mediation_proportion_mediated_identity`,
`test_mediation_bootstrap_ses_positive`, `test_mediation_percentile_cis_ordered`,
`test_mediation_p_values_in_unit_interval`, `test_mediation_seeded_bootstrap_reproducible`,
`test_mediation_propensity_separation_warns`, `test_mediation_default_seed_is_fixed`,
`test_mediation_rejects_bad_trim`, `test_mediation_both_decompositions_add_up`,
`test_mediation_bootstrap_carries_all_four`.

## Estimator

With p = P(D = 1 | X) and pM = P(D = 1 | M, X) (logistic GLMs), each potential-outcome mean is
the Hajek-normalised IPW mean hm(w) = Σ w Y / Σ w over the kept units, with weights (Huber 2014):

| Mean | Weight |
|---|---|
| E[Y(1, M(1))] | D / p |
| E[Y(0, M(0))] | (1 - D) / (1 - p) |
| E[Y(1, M(0))] | D (1 - pM) / (pM (1 - p)) |
| E[Y(0, M(1))] | (1 - D) pM / ((1 - pM) p) |

Total = E[Y(1,M(1))] - E[Y(0,M(0))] decomposes two ways: θ(0) + δ(1), with θ(0) =
E[Y(1,M(0))] - E[Y(0,M(0))] (the NDE) and δ(1) = E[Y(1,M(1))] - E[Y(1,M(0))] (the NIE), and
θ(1) + δ(0), with θ(1) = E[Y(1,M(1))] - E[Y(0,M(1))] and δ(0) = E[Y(0,M(1))] - E[Y(0,M(0))]. The
two differ when treatment and mediator interact. A unit is kept when both scores lie in
(trim, 1 - trim); trimming on pM conditions on a post-treatment variable, as Huber's estimator
does. The bootstrap resamples the kept units and carries all five effects.

## Tier 4 changes

- **Direct and indirect effects (CAUS-26).** The direct effect weighted the controls by
  pM / ((1 - p)(1 - pM)), which identifies neither θ(0) nor θ(1), and the indirect effect was
  total minus that. `direct_effect` is now θ(0) and `indirect_effect` δ(1), and θ(1) and δ(0)
  are added with SEs, intervals and p-values (Display rows "Direct θ(1)", "Indirect δ(0)").
  On `mediation_huber_sim` at trim 0 the old direct effect was 0.773059; θ(0) is 0.573967.
- **Proportion mediated** on `medflex_scales`' `yo` is now δ(1) / total = -2.993.

## Tier 3 changes

- **Propensity models** `p(D = 1 | X)` and `p(D = 1 | X, M)` are logistic GLMs fitted by the
  `glm.fit` IRLS, epsilon 1e-10, replacing a local Newton-Raphson with a Gauss-Jordan inverse
  (CAUS-113 part). They go through `econometrics::glm::fit_propensity_lean`, the lean logit path,
  and a fit that leaves it (separation, a rank-deficient or ill-conditioned weighted design, no
  convergence, a row with `y < 1` above `eta = 15` or `y > 0` below -15, a last step that no longer
  meets the rule once refined) is `fit_glm`'s own fit. Point estimates on complete data
  move by at most about 1e-9. Non-convergence, fitted probabilities numerically 0 or 1 and
  aliased covariates are reported in `MediationResult::warnings` (Display `Warning:` lines).
- **Percentile intervals** follow R's `quantile(type = 7)` (they used a floor rule); p-values are
  two-sided normal (`normal_p_value_two_sided`), no longer 0 below about 1e-16.
- **Seed:** `seed = None` means seed 42, so repeated calls give identical SEs (they drew OS
  entropy before).
- **Failed bootstrap replicates** (e.g. a resample with one arm) are counted and reported with
  the first error; fewer than 50 usable replicates (or, when fewer than 50 are
  requested, any failed one) is an error, as in medflex.
- **Proportion mediated** `NIE / ATE` is never clamped (it was clamped to [-1, 1]; on
  `medflex_scales`' `yo` it was -7.891 with the old direct-effect weight, shown as -1 before),
  with a warning outside [0, 1].
- **`trim`** must lie in [0, 0.5); a treated or control arm emptied by trimming is an error.

**Estimands**:
- Total Effect (ATE)
- Natural Direct Effect (NDE) θ(0), and θ(1)
- Natural Indirect Effect (NIE) δ(1), and δ(0)
- Proportion Mediated (NIE/ATE)

## Reference Implementations

| Package | Function | Language | Notes |
|---------|----------|----------|-------|
| causalweight | `medweight()` | R | **Primary reference** |
| mediation | `mediate()` | R | Simulation-based |
| medflex | `neWeight()` | R | Natural effect models |

## Test Case 1: Synthetic Data with Known Mediation Structure

### Data Generating Process

```
X ~ Uniform(-1, 1)
D ~ Bernoulli(0.5)  # Random treatment
M = 0.5*D + 0.3*X + epsilon_m, epsilon_m ~ N(0, 0.3)
Y = 0.4*D + 0.6*M + 0.2*X + epsilon_y, epsilon_y ~ N(0, 0.5)

True effects:
- Direct effect (NDE): 0.4
- Indirect effect (NIE): 0.5 * 0.6 = 0.3
- Total effect (ATE): 0.4 + 0.3 = 0.7
- Proportion mediated: 0.3 / 0.7 ≈ 42.9%
```

### R Code (causalweight::medweight)

```r
library(causalweight)

set.seed(42)
n <- 1000

# Covariates
x <- runif(n, -1, 1)

# Treatment (random assignment)
d <- rbinom(n, 1, 0.5)

# Mediator: M = 0.5*D + 0.3*X + noise
m <- 0.5*d + 0.3*x + rnorm(n, 0, 0.3)

# Outcome: Y = 0.4*D + 0.6*M + 0.2*X + noise
y <- 0.4*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

# Run mediation analysis
result <- medweight(
  y = y,
  d = d,
  m = m,
  x = as.matrix(x),
  boot = 999,
  trim = 0.05
)

cat("=== R causalweight Results ===\n")
cat("Total Effect:", result$total, "\n")
cat("Direct Effect (NDE):", result$dir0, "\n")  # dir0 is NDE
cat("Indirect Effect (NIE):", result$indir0, "\n")  # indir0 is NIE
cat("Proportion Mediated:", result$indir0 / result$total, "\n")
```

### Expected Results

| Statistic | True Value | R (causalweight) | p2a (Rust) | Tolerance |
|-----------|------------|------------------|------------|-----------|
| Total Effect | 0.7 | not computed in this repository | 0.65-0.75 | 0.1 |
| Direct Effect (NDE) | 0.4 | not computed in this repository | 0.35-0.45 | 0.1 |
| Indirect Effect (NIE) | 0.3 | not computed in this repository | 0.25-0.35 | 0.1 |
| Proportion Mediated | 0.43 | not computed in this repository | 0.35-0.50 | 0.1 |

The R code above is a recipe for a comparison that has not been run here; the ranges are
expectations from the DGP, not R output.

## Test Case 2: Confounded Setting

### DGP with Confounding

```
X ~ Uniform(-1, 1)
D ~ Bernoulli(logit(0.3 + 0.5*X))  # Treatment depends on X
M = 0.5*D + 0.3*X + epsilon_m
Y = 0.4*D + 0.6*M + 0.2*X + epsilon_y
```

### R Code

```r
set.seed(42)
n <- 1000

x <- runif(n, -1, 1)

# Treatment assignment depends on X (confounding)
ps <- plogis(0.3 + 0.5*x)
d <- rbinom(n, 1, ps)

m <- 0.5*d + 0.3*x + rnorm(n, 0, 0.3)
y <- 0.4*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

# Must control for X to get unbiased estimates
result <- medweight(
  y = y,
  d = d,
  m = m,
  x = as.matrix(x),
  boot = 999,
  trim = 0.05
)

print(result)
```

## Test Case 3: No Mediation (Zero Indirect Effect)

### DGP

```
D does not affect M
M = 0.3*X + epsilon_m
Y = 0.7*D + 0.6*M + 0.2*X + epsilon_y

True: NIE = 0, NDE = Total = 0.7
```

### R Code

```r
set.seed(42)
n <- 1000

x <- runif(n, -1, 1)
d <- rbinom(n, 1, 0.5)

# M does NOT depend on D
m <- 0.3*x + rnorm(n, 0, 0.3)

# Y depends on both D and M
y <- 0.7*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

result <- medweight(
  y = y, d = d, m = m,
  x = as.matrix(x),
  boot = 499
)

cat("NIE should be ~0:", result$indir0, "\n")
cat("NDE should be ~0.7:", result$dir0, "\n")
```

### Expected Results

| Statistic | True Value | Expected Range |
|-----------|------------|----------------|
| NIE | 0.0 | -0.05 to 0.05 |
| NDE | 0.7 | 0.65 to 0.75 |
| Total | 0.7 | 0.65 to 0.75 |

## Test Case 4: Full Mediation (Zero Direct Effect)

### DGP

```
D only affects Y through M
M = 0.8*D + 0.3*X + epsilon_m
Y = 0.0*D + 0.6*M + 0.2*X + epsilon_y

True: NDE = 0, NIE = Total = 0.48
```

### R Code

```r
set.seed(42)
n <- 1000

x <- runif(n, -1, 1)
d <- rbinom(n, 1, 0.5)

m <- 0.8*d + 0.3*x + rnorm(n, 0, 0.3)

# No direct effect of D on Y
y <- 0.0*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

result <- medweight(
  y = y, d = d, m = m,
  x = as.matrix(x),
  boot = 499
)

cat("NDE should be ~0:", result$dir0, "\n")
cat("NIE should be ~0.48:", result$indir0, "\n")
```

## Numerical Precision

| Sample Size | Effect Tolerance | SE Tolerance |
|-------------|------------------|--------------|
| n < 500 | 0.15 | 0.05 |
| n = 500-2000 | 0.10 | 0.03 |
| n > 2000 | 0.05 | 0.015 |

## Known Differences from causalweight

1. **Effect definitions**: causalweight returns `dir0`, `dir1`, `indir0`, `indir1`. p2a returns all four: `direct_effect` = θ(0) (`dir0`), `indirect_effect` = δ(1) (`indir1`), `direct_effect_treated` = θ(1) (`dir1`), `indirect_effect_control` = δ(0) (`indir0`).

2. **Bootstrap**: causalweight may use different resampling strategy.

3. **Propensity score model**: Both use logistic regression; p2a's is `glm.fit`'s IRLS.

4. **Trimming**: Both use symmetric trimming on propensity scores.

## Comparing p2a to causalweight

To run direct comparison:

```r
# R code
library(causalweight)

# Generate data
set.seed(42)
n <- 1000
x <- runif(n, -1, 1)
d <- rbinom(n, 1, 0.5)
m <- 0.5*d + 0.3*x + rnorm(n, 0, 0.3)
y <- 0.4*d + 0.6*m + 0.2*x + rnorm(n, 0, 0.5)

# Save for Rust
write.csv(data.frame(y=y, d=d, m=m, x=x), "mediation_test_data.csv")

# R results
result <- medweight(y=y, d=d, m=m, x=as.matrix(x), boot=999, trim=0.05)
cat("Total:", result$total, "\n")
cat("NDE (dir0):", result$dir0, "\n")
cat("NIE (indir0):", result$indir0, "\n")
```

```rust
// Rust code (pseudo)
let ds = Dataset::from_csv("mediation_test_data.csv")?;
let result = run_mediation_analysis(&ds, "y", "d", "m", &["x"], config)?;
println!("Total: {}", result.total_effect);
println!("NDE: {}", result.direct_effect);
println!("NIE: {}", result.indirect_effect);
```

## Missing values and column types

Rows with a missing value (null or NaN) in the outcome, treatment, mediator or a covariate
are dropped jointly before either propensity model is fitted, and the drop is reported in
`MediationResult.missing` and in a last Display line (also in the MCP text of
`mediation_analysis`):
`Missing values: 50 of 400 rows dropped because of missing values (y: 15, m: 25, x1: 10)`.
`N` counts the complete rows kept after trimming; `n_obs + n_trimmed` is the number of
complete rows. Int64 and Boolean columns are read as numbers (a 0/1 Int64 treatment used to
be refused as non-numeric); a text column gives `NonNumericColumn`, an absent one
`ColumnNotFound`.

R: `mediation::mediate` (mediation 4.5.1) refuses a mediator and an outcome model fitted on
different rows ("number of observations do not match between mediator and outcome
models"); with both models fitted on the complete rows it runs. p2a therefore fits both
propensity models on the same complete rows.

### R reference (T2-WP13)

Case `causal_mediation_na` (`validation/r/cases/causal_mediation_na.R`, generated dataset
`validation/datasets/generated/causal_na.csv`, registry `validation/methods/causal_na.json`,
id `mediation_na`): 400 simulated rows with a confounded 0/1 treatment `d` (Int64), 15
missing outcomes, 25 missing mediators and 10 missing `x1` values; 350 rows are complete.

| Quantity | R | Tolerance |
|---|---|---|
| complete rows, rows dropped | 350, 50 | exact |
| missing cells `y`, `m`, `x1` | 15, 25, 10 | exact |
| total effect, trim 0.05 (rows kept) | 0.7351820850179664 (280) | `Tol::iterative(1e-8)` (`glm.fit` IRLS to epsilon 1e-10 against R at 1e-14); rows exact |
| total effect, trim 0 (rows kept) | 1.0005110336992182 (350) | `Tol::iterative(1e-8)`; rows exact |
| total effect in the MCP text | 0.7352 | `Tol::printed(4)` |

The total effect is the Hajek IPW formula on `glm(binomial, epsilon = 1e-14)` propensity
scores of `d` on `(x1, x2)` and `(x1, x2, m)`, with p2a's trimming rule; it is a formula
cross-check, because causalweight is not installed. Tests (p2a-core,
`econometrics::mediation`): `test_validate_mediation_na_omit_causal_na`,
`test_validate_mediation_total_effect_causal_na`, `test_validate_na_mediation_int64_treatment`,
`test_validate_na_mediation_reports_missing`, `test_validate_na_mediation_errors_name_the_column`;
MCP (`crates/p2a-mcp/tests/tier2/wp13_causal.rs`): `test_validate_mcp_mediation_analysis_causal_na`.

Until T2-WP13 the covariate matrix was scrambled (CAUS-25): it was filled column by column
and reshaped row-major, so the estimates depended on the row and covariate order (total
effect 1.3683 on this case). The direct and indirect effects are pinned on
`mediation_huber_sim` (above).

## Wall-clock against ff8e7559

Medians over three interleaved rounds, one P core, one rayon thread, `release`, at 003af73a; ratio is
this tree over ff8e7559. The first rows are the timing harness (`wp26`), the last the Criterion group
`MediationAnalysis` (99 replicates, trim 0.05). Both `econometrics_benchmarks` binaries were built with
the default features (`full`, which includes `database`).

| Row | n | ff8e7559 | this tree | ratio |
|---|---|---:|---:|---:|
| harness, 199 replicates (ms) | 100 | 3.98 | 2.33 | 0.59 |
| harness, 199 replicates (ms) | 1,000 | 32.29 | 16.68 | 0.52 |
| harness, 199 replicates (ms) | 10,000 | 333.09 | 165.45 | 0.50 |
| Criterion, 99 replicates (ms) | 200 | 1.64 | 1.00 | 0.61 |
| Criterion, 99 replicates (ms) | 500 | 2.97 | 1.81 | 0.61 |
| Criterion, 99 replicates (ms) | 1,000 | 5.57 | 3.22 | 0.58 |

The gains depend on the design's conditioning, and the two data sets differ there. In the
Criterion data the mediator nearly separates the arms, so trimming at 0.05 drops 77% of the rows
and leaves a mediator that is close to a function of the covariate: the bootstrap resamples of
those rows give the p(D = 1 | X, M) design a median condition number of 273 at its first step.
The harness data drop 26% and stay at 11. Those fits take the refinement on their last steps.
Past a condition number of 1e4, or with a row its probability does not explain beyond
`|eta| = 15`, a fit is `fit_glm`'s and pays for both loops.

Against `fit_glm` the shared lean logit fit differs by at most 4.5e-9 on a coefficient, 9.1e-8 on an SE and
2.1e-6 on a covariance entry at the default tolerance (quasi-separated fits of n = 15 to 80 with a rare binary
covariate; 8.2e-11 and 2.7e-11 on heavy-tailed fits). These gaps are accepted.

## References

- Huber, M. (2014). "Identifying Causal Mechanisms (Primarily) Based on Inverse Probability Weighting." *Journal of Applied Econometrics*, 29, 920-943.
- Imai, K., Keele, L., & Tingley, D. (2010). "A General Approach to Causal Mediation Analysis." *Psychological Methods*, 15(4), 309-334.
- Pearl, J. (2001). "Direct and Indirect Effects." *Proceedings of the 17th Conference on Uncertainty in Artificial Intelligence*.
- Bodory, H. & Huber, M. (2018). "causalweight: Estimation Methods for Causal Inference Based on Inverse Probability Weighting." R package.
