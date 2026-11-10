# BART (regression and probit)

## Method

`p2a_core::ml::bart` samples a sum of trees (Chipman, George & McCulloch 2010) with grow and prune proposals
and conjugate leaf draws.

With `classification = true` it runs probit BART as `BART::pbart` (`y` coded 0/1, both classes present):

- `offset = qnorm(mean(y))`; trees start at 0;
- the latent SD is fixed at 1 and the leaf prior SD at `3 / (k sqrt(n_trees))`;
- each sweep draws `z_i ~ N(offset + f(x_i), 1)` truncated by `y_i` (Albert & Chib 1993), by inversion, or by
  Robert's (1995) exponential sampler when the side's probability is below 1e-100, and fits the trees to
  `z - offset`;
- each kept draw stores `Phi(offset + f)`; `predictions` are the posterior mean probabilities (pbart's
  `prob.train.mean`).

`Phi` is R's `pnorm` (r_mathlib; statrs' normal CDF errs near the 11th digit). Prediction intervals in both
modes are percentiles of the draws, R's `quantile(type = 7)`.

## R references

`validation/r/cases/ml_boosting_t3.R`. The BART package is not a case dependency, and neither reference below is a
BART-package pin.

- `ml_bart_probit_single_leaf` (`generated/bart_probit_small.csv`, n = 40): with one tree that cannot split the
  posterior of `Phi(offset + f)`, `f ~ N(0, 9 / k^2)`, is computable; `stats::integrate` gives its mean and SD.
- `ml_bart_probit_glm` (`generated/bart_probit_sim.csv`): `glm(binomial(link = "probit"))` probabilities on a
  linear-index DGP, a property target.

## Decisive values

| Quantity | Value |
|---|---|
| Offset | -0.3853205 |
| Posterior mean probability | 0.3515426 |
| Posterior SD of the probability | 0.0735486 |

## Tests

Registered as `ml_bart_probit` (`ml/bart.rs`): `test_validate_bart_probit_single_leaf_posterior`,
`test_validate_bart_probit_tracks_glm_probit`. Properties: `test_truncated_normal_draws_stay_on_their_side`
(with the tail mean against the closed form), `bart_probit_rejects_non_binary_and_single_class`,
`bart_intervals_are_type7_percentiles`.

## Tolerances

Offset `Tol::coef(40)`. Posterior mean 0.009 and SD 0.006 absolute: 5 SDs of 200 R replications of this chain (0.001745
and 0.001045). glm property: mean absolute difference below 0.085 and correlation above 0.90.

## Wall-clock

`release`, `ci` features, against `ff8e7559`: one fit per process, nine rounds, base and this tree
interleaved per row with the order alternating, one P core (`taskset`), `RAYON_NUM_THREADS=1`, 2026-11-08,
load average 1.5-4.2 (falling from a finished build; the fits run on their own core). Data: n x 5 uniform
design, y = 2 x1 + 0.5 x2 + noise. Cells are medians; "ratio" is this tree over the base, "min ratio" the
ratio of the minima.

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| BART regression, 50 trees, 20 burn-in, 50 draws | 100 | 13.45 | 13.49 | 1.00 | 1.00 |
|  | 1,000 | 145.1 | 143.4 | 0.99 | 0.98 |
|  | 10,000 | 1,508.8 | 1,504.8 | 1.00 | 1.00 |

Within 1.1x. Probit BART is new (the base ignored `classification`) and is not timed.

## Not pinned

The regression sampler: not R-pinned. Its grow/prune kernel, shared with probit, has open findings (ML-15, ML-16, ML-64,
ML-120, ML-121, ML-122). The single-leaf pin excludes the kernel, and the glm property is insensitive to it.
