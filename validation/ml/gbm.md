# Gradient boosting machine (gbm)

## Method

`p2a_core::ml::gbm` is Friedman's (2001) gradient boosting with least-squares regression trees. Each iteration
fits a tree to the pseudo-residuals of the in-bag rows (the split search, its gain and the relative influence use
the pseudo-residuals), then re-estimates each terminal node by the family's own rule over the leaf's in-bag rows
(R gbm's `FitBestConstant`), and adds `learning_rate` times the tree to the fit.

| `GbmFamily` | Pseudo-residual | F0 | Terminal node over the leaf's rows |
|---|---|---|---|
| `Gaussian` | `y - F` | `mean(y)` | mean pseudo-residual |
| `Laplace` | `sign(y - F)` | `median(y)` | `median(y - F)` (Friedman, Algorithm 3) |
| `Binomial` | `y - p`, `p = plogis(F)` | `qlogis(mean(y))` | `sum(y - p) / sum(p (1 - p))`, 0 when the denominator is 0 (Algorithm 5) |
| `Huber` | `clip(y - F, delta)` | `median(y)` | `med + mean(sign(r - med) min(delta, abs(r - med)))`, `r = y - F`, `med = median(r)` (Algorithm 4) |

Medians are R's: the mean of the two middle values for an even count.

**Huber threshold.** With `huber_alpha = Some(a)` (default 0.9, `a` in (0, 1)), each iteration's `delta` is the
type-7 `a`-quantile of `abs(y - F)` over that iteration's in-bag rows, as Friedman's M-TreeBoost and scikit-learn's
`HuberLoss`; the fit is then scale-equivariant (a fit on `1000 y` is 1000 times the fit on `y`). With
`huber_alpha = None`, the fixed `huber_delta` is used (in the units of `y`). The training loss of an iteration uses
that iteration's `delta`; the initial loss uses the `delta` of `y - F0` over all rows. gbm has no Huber regression,
so this family is property-tested only (`test_gbm_huber_leaf_is_friedman_algorithm_4`,
`test_gbm_huber_and_laplace_are_scale_equivariant`, `test_gbm_huber_fits_a_large_scale_target`).

Before T4 (ML-94, WT-1) every family's leaf was the mean pseudo-residual: Laplace and Huber leaves were bounded by 1
and `delta`, so 50 trees at rate 0.1 could not move the fit more than 5 (or `5 delta`) from F0, and Bernoulli leaves
were a gradient step instead of a Newton step.

Trees stop at depth 512 whatever `max_depth` says (as the other tree builders).

## R references

`validation/r/cases/ml_boosting_t4.R`, data `generated/gbm_leaf_t4.csv` (n = 200; `y_lad = 100 x0 + 400 1{x1 >
0.5} + 5 t3`, `y_bin` Bernoulli), registry `validation/methods/ml_boosting_t4.json` (`ml_gbm_leaves`), gbm 2.3.1:
`gbm(distribution = 'laplace' | 'bernoulli', n.trees = 50, interaction.depth = 1, n.minobsinnode = 1,
shrinkage = 0.1, bag.fraction = 1, train.fraction = 1)`.

| Case | Decisive values |
|---|---|
| `gbm_laplace_stumps` | `initF` 704.8775263259499; fitted range [198.18419631849926, 1139.487334411909], sum 137300.9174633154; 26 stumps on x0, 24 on x1; relative influence 1568.6574629631994, 1484.9487285621392, 0; MAE 52.96535637047575 |
| `gbm_bernoulli_stumps` | `initF` 0.6410908186038922; probability range [0.19000883719411826, 0.9108185012773216], sum 130.98737288833993; 31 / 15 / 4 stumps; relative influence 68.77672212941899, 7.34185530651337, 1.093445751345184 |

Tests (R-pinned, `Tol::coef(200)`, split variables exact): `test_validate_gbm_laplace_stumps`,
`test_validate_gbm_bernoulli_stumps`. The Gaussian stumps are pinned in `validation/methods/ml_cart_cp.json`
(`test_validate_gbm_stump_influence`; see `tree_importance.md`). Before the fix the Laplace fit ranged over
[700.17, 707.14] (MAE 276.94) and the Bernoulli probabilities over [0.442, 0.789].

## Wall-clock

`release`, `ci` features, the T4 boosting/forest fixes against the code before them (`34a4186a`), nine processes per
side interleaved, one core (`taskset`), one rayon thread, 2026-11-16, load average 22-42 on 12 cores (a loaded
machine: read ratios near 1 as within noise). Cells are the median of the process medians; "min ratio" is the ratio
of the minima. Design: n rows, 5 features uniform on [0, 10], `y = 2 x0 - x4 + (x1 - 5)^2 / 5` plus noise.

| `gbm()` defaults | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| Gaussian | 1,000 | 136.3 | 137.8 | 1.01 | 0.99 |
|  | 10,000 | 1,681 | 1,662 | 0.99 | 0.99 |
| Laplace | 1,000 | 133.3 | 132.5 | 0.99 | 1.00 |
|  | 10,000 | 1,734 | 1,719 | 0.99 | 0.97 |
| Bernoulli (`y > 5`) | 1,000 | 145.5 | 148.0 | 1.02 | 1.04 |
|  | 10,000 | 1,873 | 1,906 | 1.02 | 1.01 |
| Huber | 1,000 | 138.8 | 143.7 | 1.04 | 1.03 |
|  | 10,000 | 1,792 | 1,839 | 1.03 | 0.99 |

The leaf re-estimates are O(n) per tree against the split search's O(p n log n) per level: within 1.05x.

## What does not match R

- **Trees deeper than stumps are not R-pinned.** gbm's `interaction.depth` counts splits (a tree grows by its best
  split anywhere), p2a's `max_depth` counts levels, so the trees differ beyond depth 1; nothing is compared there.
- Subsampled fits (`subsample < 1`) draw rows with p2a's generator, not R's.
- The Laplace pseudo-residual of a zero residual is 0 (gbm: -1); it changed nothing on the pinned design.
