# XGBoost (exact and histogram)

## Method

`p2a_core::ml::xgboost` is the exact greedy algorithm of Chen & Guestrin (2016, Algorithm 1);
`p2a_core::ml::xgboost_fast` is a histogram variant. Both use XGBoost's L1 rule (`src/tree/param.h`,
`CalcWeight` and `CalcGain`): the leaf weight is `-ThresholdL1(G, alpha) / (H + lambda)` and each gain term
`ThresholdL1(G, alpha)^2 / (H + lambda)`.

- **Histogram bins.** Edges are the sorted values at `floor(i n / bins)` and a value goes to the first edge
  it does not exceed, so the two smallest values share bin 0. The binning is p2a's, not the package's
  sketch.
- **`colsample_bylevel`.** Each tree level samples `ceil(fraction * columns)` columns, at least 1, out of
  the tree's columns. XGBoost uses floor; p2a's ceil is a recorded deviation on both field docs.
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

Registered as `ml_xgboost_l1`: `test_validate_xgboost_exact_l1`, `test_validate_xgboost_exact_package`
(`ml/xgboost.rs`), `test_validate_fast_xgboost_l1_stump`, `test_validate_fast_xgboost_l1_four_rounds`
(`ml/xgboost_fast.rs`).

Properties: `colsample_bylevel` sampling and its ceil rule, `n_jobs` and subtraction give the same trees up to
last-bit differences, strided and column-major input views.

`crates/p2a-core/tests/validation_boosting_methods.rs` holds DGP-recovery checks, not R comparisons.

## Tolerances

Written-out rule: `Tol::coef(16)` (1e-6 relative for n = 16), split features and bins exact. Package:
1e-6 absolute, the resolution of its single-precision arithmetic.

## Not pinned

The histogram trees against the package (the bins differ); `colsample_*` draws (p2a's own generator). Fast
XGBoost's seeding (ML-42) and exact XGBoost's early stopping on the training loss (ML-143) are open findings.
For gbm, see `tree_importance.md`.
