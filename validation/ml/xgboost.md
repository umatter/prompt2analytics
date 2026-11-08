# XGBoost (exact and histogram)

## Method

`p2a_core::ml::xgboost` is the exact greedy algorithm of Chen & Guestrin (2016, Algorithm 1);
`p2a_core::ml::xgboost_fast` is a histogram variant. Both use XGBoost's L1 rule (`src/tree/param.h`,
`CalcWeight` and `CalcGain`): the leaf weight is `-ThresholdL1(G, alpha) / (H + lambda)` and each gain term
`ThresholdL1(G, alpha)^2 / (H + lambda)`.

- **Histogram bins.** With `bins = min(max_bin, n)`, the edges are the sorted values at 0-based positions
  `floor(i n / bins)`, `i = 1, ..., bins - 1` (repeated values once), and a value goes to the first edge it
  does not exceed, so bin 0 holds at least the `floor(n / bins) + 1` smallest values (the two smallest when
  `max_bin >= n`). The binning is p2a's, not the package's sketch.
- **`colsample_bylevel`.** Each tree level samples `ceil(fraction * columns)` columns, at least 1, out of
  the tree's columns. XGBoost uses floor; the field's doc records p2a's ceil as a deviation.
- **`n_jobs`.** A positive value runs the fit in an own rayon pool of that size; 0 uses the global pool.
- **Histogram subtraction.** A node builds the smaller child's histogram and subtracts it from its own for
  the sibling, as the fast module's header says.

## R references

`validation/r/cases/ml_boosting_t3.R`:

- `ml_xgboost_pkg`: the installed package, `xgboost::xgb.train(tree_method = "exact")` (3.2), on
  `generated/xgb_small.csv` (n = 16), alpha 0 and 3, 3 rounds, depth 2, base score `mean(y)`. The predictions,
  the root split and the gain are compared; xgboost reports twice the gain of the paper's equation 7 and
  computes in single precision.
- `ml_xgboost_l1`: the published rule written out in R, for the exact trees and for p2a's histogram trees. It
  agrees with the package on the exact trees up to single precision (largest difference 1.2e-7).

## Decisive values

| Quantity | Value |
|---|---|
| Exact root, alpha 0 | feature x2, threshold -0.6982305, gain 1.8465346 |
| Exact root, alpha 3 | feature x1, threshold 0.586179, gain 0.1070430: alpha moves the split |
| Histogram stump leaves, alpha 0 | -0.3975445, 0.4969306 |
| Histogram stump leaves, alpha 3 | -0.0975445, 0.1219306 |

## Tests

Registered under `ml_xgboost_l1` (cases `ml_xgboost_l1` and `ml_xgboost_pkg`):
`test_validate_xgboost_exact_l1`, `test_validate_xgboost_exact_package` (`ml/xgboost.rs`),
`test_validate_fast_xgboost_l1_stump`, `test_validate_fast_xgboost_l1_four_rounds` (`ml/xgboost_fast.rs`).

Properties: `colsample_bylevel` sampling and its ceil rule, `n_jobs` and subtraction give the same trees up to
last-bit differences, strided and column-major input views.

`crates/p2a-core/tests/validation_boosting_methods.rs` holds DGP-recovery checks, not R comparisons.

## Tolerances

Written-out rule: `Tol::coef(16)` (1e-6 relative for n = 16), split features and bins exact. Package:
1e-6 absolute, the resolution of its single-precision arithmetic.

## Wall-clock

`release`, `ci` features, against `fe13b889`: one fit per process, nine rounds, base and this tree
interleaved per row with the order alternating, one P core (`taskset`), `RAYON_NUM_THREADS=1`, 2026-11-08,
load average 1.5-4.2 (falling from a finished build; the fits run on their own core). Data: n x 5 uniform
design, y = 2 x1 + 0.5 x2 + noise. Cells are medians; "ratio" is this tree over the base, "min ratio" the
ratio of the minima.

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| exact XGBoost, defaults | 100 | 6.68 | 8.11 | 1.21 | 1.21 |
|  | 1,000 | 76.98 | 86.94 | 1.13 | 1.13 |
|  | 10,000 | 972.6 | 1,039.7 | 1.07 | 1.07 |
| fast (histogram) XGBoost, defaults | 100 | 34.97 | 35.81 | 1.02 | 1.05 |
|  | 1,000 | 77.83 | 83.45 | 1.07 | 1.06 |
|  | 10,000 | 201.3 | 182.4 | 0.91 | 0.92 |

Exact XGBoost at n = 100 and 1,000 (1.21x, 1.13x) has not been signed off. The per-node sort compares with
`total_cmp` and the split gain soft-thresholds its three sums; reverting the sort comparator alone leaves
1.19x / 1.10x. With `alpha = 1` the split gain itself changes (alpha now enters it), so the trees differ and the
row is not like-for-like:

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| exact XGBoost, alpha = 1 | 1,000 | 78.58 | 32.08 | 0.41 | 0.41 |

LightGBM has no page of its own; its rows, same conditions:

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| LightGBM, L2, defaults | 100 | 4.77 | 6.29 | 1.32 | 1.31 |
|  | 1,000 | 277.6 | 319.7 | 1.15 | 1.15 |
|  | 10,000 | 2,934.2 | 3,388.7 | 1.16 | 1.15 |
| LightGBM, Huber | 1,000 | 279.7 | 320.6 | 1.15 | 1.14 |

The LightGBM rows (1.15-1.32x) have not been signed off. A build of this tree with the base's `lightgbm.rs`
runs at 1.00x, and one with only the `total_cmp` comparisons reverted still at 1.31x, so the cost is in
`lightgbm.rs` but not in a comparison; the L2 path does no new work per iteration (the remaining edits are the
Huber unit hessians and the up-front finite check). The Huber row is not like-for-like: the base's loss
oscillated with zero hessians.

## Not pinned

The histogram trees against the package (the bins differ); `colsample_*` draws (p2a's own generator). Fast
XGBoost's seeding (ML-42) and exact XGBoost's early stopping on the training loss (ML-143) are open findings.
For gbm, see `tree_importance.md`.
