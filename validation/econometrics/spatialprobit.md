# Spatial Probit Models Validation

## Method Overview

The `spatialprobit` module implements Bayesian spatial probit models for binary dependent variables with spatial dependence, by the Gibbs sampler of LeSage & Pace (2009, ch. 10), the scheme of R's `spatialprobit` package (Wilhelm & de Matos, 2013). The **posterior is not R-pinned**: that package is not installed, and an MCMC run could not reproduce its draws anyway. The deterministic pieces of the sampler and of the summaries (the full conditionals, the marginal probabilities, the composite log-likelihood and the impacts at given parameters) are R-pinned against base R (registry entry `spatial_probit_pieces` in `validation/methods/spatial_diag_t4.json`, case `spatial_probit_pieces_t4` of `validation/r/cases/spatial_diag_t4.R`), and the posterior is checked by recovering simulated parameters.

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

Gibbs sampling with data augmentation (Albert & Chib 1993), A = I - rho W, one sweep of each step per
iteration:

1. **Latent outcomes**: one single-site sweep. z_i given the others is N(m_i, 1 / H_ii) with H = A'A,
   m_i = z_i - (A'(A z - c))_i / H_ii (c = X beta for SAR, A X beta for SEM), truncated to (0, inf) where
   y_i = 1 and (-inf, 0] where y_i = 0. Each draw is exact: normal rejection when the bound is on the near
   side of the mean, Robert's (1995) exponential rejection on the far side. The residual A z - c is
   updated along column i of A after each draw, so a sweep costs O(links) with no solve. spatialprobit
   runs m = 10 inner sweeps of its truncated multivariate normal sampler per iteration; both are valid
   Gibbs schemes for the same posterior.
2. **beta | z, rho**: conjugate normal, prior N(beta0, I / 0.01).
3. **rho | z, beta**: griddy Gibbs on 2,000 cells of the prior range intersected with the interval where
   A is invertible, with the log kernel ln|A| - ||A z - c||^2 / 2 (ln|A| from the eigenvalues of W, once
   per fit); the draw inverts the trapezoid CDF within its cell. There is no rejection step, so
   `acceptance_rate` is 1.

**Summaries at the posterior mean** use the marginal distribution of each latent outcome,
z_i ~ N(mu_i, sigma_i^2) with mu = A^-1 X beta (SAR) or X beta (SEM) and sigma_i^2 = [(A'A)^-1]_ii, from one
dense inverse: `fitted_prob` = Phi(mu_i / sigma_i); `log_likelihood` is the composite (marginal)
log-likelihood sum_i ln Phi(+-mu_i / sigma_i); McFadden's pseudo R^2 against the intercept-only probit; the
percent correctly predicted. The joint likelihood is an n-dimensional orthant probability and is not
computed, so `dic` and `log_marginal_likelihood` are NaN (spatialprobit reports neither).

**Impacts** (per draw, then posterior mean and SD; intercept excluded):

- SAR, spatialprobit's and LeSage & Pace's convention: phi_i = dnorm(mu_i), direct_k =
  beta_k mean_i(phi_i [A^-1]_ii), total_k = beta_k mean_i(phi_i [A^-1 1]_i), indirect = total - direct.
- SEM: the derivative of the reported probability Phi(mu_i / sigma_i), mu = X beta: direct_k =
  beta_k mean_i(phi(mu_i / sigma_i) / sigma_i); no spillover (indirect 0, total = direct). The MCP tool
  `sem_probit_model` does not request them.

A^-1 is exact, not a truncated Neumann series: dense inverses at Chebyshev points of the second kind over
the range of the rho draws (9 nodes, doubled to 17 or 33 until nested interpolants agree to 1e-10),
interpolated barycentrically to each draw. When 33 nodes miss the target (rho draws reaching close to
1 / lambda_max, where the multiplier is singular), the interpolant is measured against the dense inverse
in its end cells and a warning reports the attained accuracy (about 3e-7 for draws spanning -0.3 to 0.92
on the test weights).

## Validation

**R-pinned pieces** (`spatial_probit_pieces_t4`: kNN-4 row-standardised W of `spatial_sim`, n = 80,
X = [1, x1, x2], beta = (-0.3, 0.8, -0.5); closed-form quantities, 1e-8 relative):

- `test_validate_probit_rho_kernel_spatial_sim`: ln|I - rho W| and the SAR and SEM log kernels of rho on
  the grid -0.9, ..., 0.9;
- `test_validate_probit_latent_conditional_spatial_sim`: the full-conditional means of z_i (SAR and SEM)
  and variances 1 / H_ii at rho = 0.6;
- `test_validate_probit_marginal_probabilities_spatial_sim`: sigma_i, Phi(mu_i / sigma_i) and the composite
  log-likelihood, SAR at rho = 0.5 and 0.9 and SEM at lambda = 0.6;
- `test_validate_probit_impacts_exact_spatial_sim`: SAR direct, indirect and total effects at rho = 0.5
  and 0.9 from the exact inverse;
- `test_validate_sem_probit_direct_effects_spatial_sim`: SEM direct effects at lambda = 0.6;
- `test_validate_dense_inverse_parts_spatial_sim` (`spatial::weights`): diag(A^-1), [(A'A)^-1]_ii and
  A^-1 X.

| Quantity (rho = 0.5) | x1 | x2 |
|---|---|---|
| direct | 0.260576 | -0.162860 |
| indirect | 0.227208 | -0.142005 |
| total | 0.487784 | -0.304865 |

At rho = 0.9 the total effect of x1 is 1.115893; the ten-term Neumann series the impacts used before
gave 1.036181 (7% short).

**The posterior**: `sar_probit_recovers_dgp` and `sem_probit_recovers_dgp` simulate a 20 x 20 rook grid
(row-standardised), beta = (0, 1), SAR rho = 0.5 and 0.8 and SEM lambda = 0.6, three seeds each, 2,000
draws after 500 burn-in: the posterior mean is within 0.15 of the truth and the 95% credible interval
covers it. `probit_rho_step_draws_from_the_kernel` draws 20,000 values of rho at fixed z and beta and
compares their mean and SD with the kernel's by quadrature (within 4 Monte Carlo SEs).

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

**The impacts interpolation**: `probit_impacts_interpolation_matches_dense` compares interpolated impacts
with those of the dense inverse at off-node rho (1e-9 relative); `probit_impacts_interpolation_warns_near_the_singular_rho`
checks the warning; `sem_probit_impacts_have_no_spillover`.

**Credible intervals**: `test_spatial_probit_credible_intervals_are_type7` checks R's
`quantile(type = 7)` formula on the sorted draws, to 1e-12.

**Seeded chains**: `seeded_probit_runs_are_unchanged` pins seeded posterior summaries as a
regression test. These are this implementation's own values at seed 42, not a reference.

Exact replication of an R run would require the same random stream. Posterior summaries agree
with another implementation only up to Monte Carlo error.

### Differences from R spatialprobit

1. **Latent sweeps**: one single-site Gibbs sweep per iteration; spatialprobit runs m = 10 sweeps of
   its sparse truncated-multivariate-normal sampler. Both target the same posterior; mixing per
   iteration differs.
2. **Impacts**: the exact multiplier (interpolated in rho to 1e-10), where spatialprobit approximates it by
   a power series in W; the SAR convention is the same. SEM impacts have no spillover.
3. **Fit statistics**: the composite log-likelihood at the posterior mean; no DIC or marginal
   likelihood.

## Timing against the previous implementation (Tier 4)

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. Data: a binary outcome on
a jittered grid with 4-nearest-neighbour weights, two regressors,
the default configuration (seed 1); five calls per process at n = 100, one at n = 1,000 and 2,000.

| Model | n | f80b0131 (ms) | this change (ms) | ratio [min] |
|---|---|---|---|---|
| SAR probit, impacts | 100 | 74.56 | 87.10 | 1.17 [1.10] |
| SAR probit, impacts | 1,000 | 1,554 | 3,652 | 2.35 [2.35] |
| SAR probit, impacts | 2,000 | 6,368 | 21,318 | 3.35 [3.35] |
| SAR probit, no impacts | 100 | 63.13 | 61.21 | 0.97 [0.96] |
| SAR probit, no impacts | 1,000 | 1,439 | 1,224 | 0.85 [0.85] |
| SAR probit, no impacts | 2,000 | 6,132 | 6,263 | 1.02 [1.02] |
| SEM probit, no impacts | 100 | 48.52 | 63.18 | 1.30 [1.30] |
| SEM probit, no impacts | 1,000 | 1,217 | 1,234 | 1.01 [1.01] |
| SEM probit, no impacts | 2,000 | 5,695 | 6,300 | 1.11 [1.11] |

The impacts cost 17 to 37 dense LU inverses of I - rho W, 19 typical (17 interpolation nodes and two checks
against the dense inverse in the end cells; a failed check adds the 16 nodes of the next refinement and two
more checks), O(n^3) each, where the old code summed a truncated power series in
W; above about n = 1,000 they dominate the run. The sampler itself (no impacts) costs what the old one
did within about 10% from n = 1,000; at n = 100 the 2,000-cell rho grid evaluated per iteration
dominates the SEM run. The SAR rows with impacts at n = 1,000 and 2,000 and the SEM row at n = 100 are
signed off 2026-11-16 (quiet 2.35 / 3.35x and 1.30x against the signed 2.36 / 3.43x and 1.27x); the SAR row with
impacts at n = 100 (1.17x) and the SEM row at n = 2,000 (1.11x) are within the estimates signed off 2026-11-15
(0.8-1.2x). Peak resident memory of one process at n = 2,000 (median of three): SAR with impacts 112.4 -> 201.5 MiB
(1.79x), SEM 112.5 -> 170.8 MiB (1.52x), the dense n x n inverses; signed off 2026-11-17.
On a hosted server `probit_impact_rows` holds `sar_probit_model` with impacts to 1,500 rows.

## MCMC Diagnostics

The result includes:
- `acceptance_rate`: always 1 (the griddy-Gibbs step has no rejection)
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
- `sample_latent_sweep()`: one single-site Gibbs sweep over the latent outcomes
- `standard_normal_truncated_below()`: Exact draw from N(0, 1) truncated to [a, inf) (Robert 1995)
- `sample_beta()`: Conjugate normal posterior for beta
- `rho_log_kernel()`, `draw_from_grid()`: griddy Gibbs for the spatial parameter
- `marginal_summary()`: marginal probabilities and composite log-likelihood at given parameters
- `compute_spatial_probit_impacts()`, `InverseInterpolant`: Direct/indirect/total effects

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
- R-pinned pieces: `test_validate_probit_rho_kernel_spatial_sim`,
  `test_validate_probit_latent_conditional_spatial_sim`,
  `test_validate_probit_marginal_probabilities_spatial_sim`,
  `test_validate_probit_impacts_exact_spatial_sim`, `test_validate_sem_probit_direct_effects_spatial_sim`.
- Posterior and sampler: `sar_probit_recovers_dgp`, `sem_probit_recovers_dgp`,
  `probit_rho_step_draws_from_the_kernel`, `truncated_normal_far_bound_returns_the_bound`,
  `ln_normal_cdf_is_accurate_in_both_tails`.
- Impacts: `probit_impacts_interpolation_matches_dense`,
  `probit_impacts_interpolation_warns_near_the_singular_rho`, `sem_probit_impacts_have_no_spillover`.
- Pins: `seeded_probit_runs_are_unchanged`.
- Missing values: `test_na_spatial_probit_int64_and_boolean_outcome`,
  `test_na_spatial_probit_missing_outcome`.

## MCP Tool Integration

Tools exposed:
- `sar_probit_model`: SAR probit for binary spatial outcomes
- `sem_probit_model`: SEM probit for binary outcomes with spatial error

Both tools require spatial weights to be created first using `spatial_neighbors`.

`SpatialProbitResult` and `SpatialProbitImpacts` implement `Display`, with one impact row per
variable carrying its own posterior SDs.
