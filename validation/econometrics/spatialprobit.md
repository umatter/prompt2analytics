# Spatial Probit Models Validation

## Method Overview

The `spatialprobit` module implements Bayesian spatial probit models for binary dependent variables with spatial dependence. They follow the model of R's `spatialprobit` package (Wilhelm & de Matos, 2013). They are **not R-pinned**: that package is not installed, and the validation is by the property tests below.

### Models Implemented

1. **SAR Probit (Spatial Autoregressive Probit)**
   - Model: y* = rho * W * y* + X * beta + epsilon, epsilon ~ N(0, I)
   - y = 1 if y* > 0, else y = 0
   - Captures spatial spillover effects in binary outcomes

2. **SEM Probit (Spatial Error Probit)**
   - Model: y* = X * beta + u, u = lambda * W * u + epsilon, epsilon ~ N(0, I)
   - y = 1 if y* > 0, else y = 0
   - Accounts for spatial correlation in errors (nuisance dependence)

### Data requirements

- One dataset row per unit of W: row i of the data is unit i of the weights, so no row can
  be dropped.
- Missing values (null or NaN) in y or X are refused with `NullValues` naming the column,
  as in spdep/spatialreg (`na.fail`); remove the incomplete rows and rebuild W first.
- Integer and Boolean (0/1) columns are accepted. The outcome is exactly 0/1 and may be an
  Int64, Float64 or Boolean column with both values present; any other value is refused with
  `InvalidSpecification` naming the column and the first offending row.
- `n_draws` must be at least 2 (a posterior SD needs two retained draws), `thin` at least 1.
- `compute_impacts` needs at least one regressor besides the intercept.
- With no `seed`, a run uses the documented default seed 42, so every run is reproducible.
- Tests (T2-WP14): `test_na_spatial_probit_int64_and_boolean_outcome` and
  `test_na_spatial_probit_missing_outcome` (`econometrics::spatialprobit::tests`).

## Estimation Method

Uses Bayesian MCMC with data augmentation following LeSage & Pace (2009):

1. **Data Augmentation**: Albert & Chib (1993) method for latent y*
2. **Sampling Steps**:
   - Sample y* from the exact one-sided truncated normal given (beta, rho) and observed y: normal rejection when the bound is on the near side of the mean, Robert's (1995) exponential rejection when it is on the far side, so both tails are exact
   - Sample beta from conjugate multivariate normal posterior
   - Sample rho using Griddy Gibbs (Ritter & Tanner, 1992)
3. **Posterior Inference**: Means, standard deviations, and credible intervals (R type-7 quantiles of the retained draws) from MCMC draws

## Validation

There is no R reference: `spatialprobit` is not installed, and the seeded chains could not be
compared anyway (`StdRng` is ChaCha12). What is checked:

**The latent draws**, against the exact distribution. `test_truncated_normal_sampler_ks` and
`test_truncated_normal_sampler_tail_6_sigma` draw 20,000 values from N(mu, 1) truncated at 0 and
compare them with the exact one-sided truncated-normal CDF (through upper tails, so that it is
accurate far from the mean):

- every draw is on the right side of the bound;
- the sample mean is within 5 SD / sqrt(N) of the exact mean;
- the Kolmogorov-Smirnov distance is below 0.019045116000253333 (`sqrt(-log(0.5e-6) / 2) / sqrt(N)`,
  P(D > critical) = 1e-6).

| mu | Truncation | Exact mean | Exact SD |
|---|---|---|---|
| 1 | y = 1, above 0 | 1.28759997 | 0.79352775 |
| -2.5 | y = 1, above 0 | 0.32274480 | 0.29828477 |
| -6.2 | y = 1, above 0 | 0.15382038 | 0.15050886 |
| -8 | y = 1, above 0 | 0.12136811 | 0.11968661 |
| 7 | y = 0, below 0 | -0.13754561 | 0.13513664 |
| -0.5 | y = 0, below 0 | -1.00916043 | 0.69726282 |

The means and SDs are computed in R from `lam = dnorm(a) / pnorm(a, lower.tail = FALSE)`. The
inverse-CDF sampler this replaced failed in both tails: at mu = 7 every draw was above the bound,
and at mu = -8 every draw was the constant 0.01.

**The impacts of one draw**: `probit_draw_impacts_match_r_reference` pins direct and total effects
against the Neumann series transcribed in R.

**Credible intervals**: `test_spatial_probit_credible_intervals_are_type7` checks R's
`quantile(type = 7)` formula on the sorted draws, to 1e-12.

**Seeded chains**: `seeded_probit_runs_are_unchanged` pins seeded posterior summaries as a
regression test. These are this implementation's own values, not a reference.

Exact replication of an R run would require the same random stream. Posterior summaries agree
with another implementation only up to Monte Carlo error.

### Differences from R spatialprobit

1. **Approximation method**: a truncated Neumann series for (I - rho*W)^{-1}.
2. **Griddy Gibbs**: a 50-point grid for rho.

## MCMC Diagnostics

The result includes:
- `acceptance_rate`: Should be between 0.2-0.8 for good mixing
- `beta_draws`: Full chain for convergence diagnostics
- `rho_draws`: Full chain for spatial parameter
- `credible_interval_*`: 95% credible intervals

### Convergence Checks

1. **Trace plots**: Examine beta_draws and rho_draws
2. **Autocorrelation**: Should decay quickly
3. **Effective sample size**: Should be substantial fraction of n_draws

## Implementation Details

### File Location
`crates/p2a-core/src/econometrics/spatialprobit.rs`

### Key Functions
- `run_sar_probit()`: SAR probit estimation
- `run_sem_probit()`: SEM probit estimation
- `sample_latent_y()`: Truncated normal sampling for y*
- `standard_normal_truncated_below()`: Exact draw from N(0, 1) truncated to [a, inf) (Robert 1995)
- `sample_beta()`: Conjugate normal posterior for beta
- `sample_rho_griddy_gibbs()`: Griddy Gibbs for spatial parameter
- `compute_spatial_probit_impacts()`: Direct/indirect/total effects

### Dependencies
- `ndarray`: Matrix operations
- `rand`, `rand_distr`: Random number generation
- `statrs`: Normal distribution functions

## References

### Primary Sources

1. LeSage, J.P. & Pace, R.K. (2009). "Introduction to Spatial Econometrics". CRC Press, Chapter 10.
   ISBN: 978-1420064247.

2. Albert, J.H. & Chib, S. (1993). "Bayesian analysis of binary and polychotomous response data".
   *Journal of the American Statistical Association*, 88(422), 669-679.
   https://doi.org/10.1080/01621459.1993.10476321

3. Geweke, J. (1991). "Efficient simulation from the multivariate normal and Student-t distributions
   subject to linear constraints". *Computing Science and Statistics: Proc. 23rd Symposium on the
   Interface*, 571-578.

4. Beron, K.J. & Vijverberg, W.P.M. (2004). "Probit in a Spatial Context: A Monte Carlo Analysis".
   In Anselin, L., Florax, R.J.G.M. & Rey, S.J. (Eds.), *Advances in Spatial Econometrics*
   (pp. 169-195). Springer.

5. Robert, C.P. (1995). "Simulation of truncated normal variables". *Statistics and Computing*,
   5(2), 121-125. https://doi.org/10.1007/BF00143942

### R Package Reference

6. Wilhelm, S. & de Matos, M.G. (2013). "Estimating Spatial Probit Models in R".
   *The R Journal*, 5(1), 130-143.
   https://cran.r-project.org/package=spatialprobit

### Related Implementations

- R package `spatialprobit`: https://cran.r-project.org/package=spatialprobit
- R package `spdep`: Spatial dependence and weights
- R package `spatialreg`: Spatial regression models (continuous outcomes)

## Tests

All in `econometrics::spatialprobit::tests`:

- Estimation: `test_sar_probit_basic`, `test_sem_probit_basic`, `test_spatial_probit_impacts`,
  `test_credible_intervals`, `test_truncated_normal_sampling`.
- Sampler: `test_truncated_normal_sampler_ks`, `test_truncated_normal_sampler_tail_6_sigma`.
- Inputs: `test_spatial_probit_rejects_n_draws_0`, `test_spatial_probit_rejects_thin_0`,
  `test_spatial_probit_rejects_non_binary`, `test_spatial_probit_rejects_impacts_without_regressors`,
  `probit_rejects_zero_draws_and_zero_thinning`,
  `probit_rejects_draw_counts_that_overflow_or_cannot_be_stored`, `probit_sampler_honours_cancellation`.
- Output: `test_spatial_probit_display_lists_every_impact_se`,
  `test_spatial_probit_credible_intervals_are_type7`, `test_spatial_probit_unseeded_runs_are_reproducible`.
- Pins: `probit_draw_impacts_match_r_reference`, `seeded_probit_runs_are_unchanged`.
- Missing values: `test_na_spatial_probit_int64_and_boolean_outcome`,
  `test_na_spatial_probit_missing_outcome`.

## MCP Tool Integration

Tools exposed:
- `sar_probit_model`: SAR probit for binary spatial outcomes
- `sem_probit_model`: SEM probit for binary outcomes with spatial error

Both tools require spatial weights to be created first using `spatial_neighbors`.

`SpatialProbitResult` and `SpatialProbitImpacts` implement `Display`, with one impact row per
variable carrying its own posterior SDs.
