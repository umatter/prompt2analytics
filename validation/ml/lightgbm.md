# LightGBM-style histogram boosting

## Method

`p2a_core::ml::lightgbm` is leaf-wise gradient boosting on histogram bins (Ke et al. 2017), written in Rust. It
is **not R-pinned**: no case compares it with the `lightgbm` R package, because the binning and the tie rules
differ. The rules below follow LightGBM 4.7.0 and are checked as properties.

- **Leaf-wise growth.** A tree grows by splitting, among its current leaves, the one whose best split gains most,
  until it has `num_leaves` leaves (2 to 131,072, LightGBM's range), no leaf has a positive-gain split that keeps
  `min_data_in_leaf` rows on each side, or `max_depth` is reached (-1: no limit; every tree also stops at depth
  512). `test_lightgbm_num_leaves_caps_every_tree`, `test_lightgbm_num_leaves_two_is_the_root_split`.
- **Bagging.** Rows are bagged only when `bagging_freq > 0` and `bagging_fraction < 1`: a bag of
  `ceil(n bagging_fraction)` rows is drawn every `bagging_freq` iterations (iterations 0, k, 2k, ...) and kept in
  between. `bagging_freq = 0` (the default) never bags, whatever `bagging_fraction` says.
  `test_lightgbm_bagging_freq_zero_does_not_bag`, `test_lightgbm_bag_is_kept_between_rebags`.
- **Leaves.** Each leaf is the Newton step `-G / (H + lambda_l2)`, with `lambda_l1` soft-thresholding `G`. For
  `regression_l1` the leaves are then renewed with the median of `y - F` over the leaf's in-bag rows, before the
  learning rate (LightGBM's `RenewTreeOutput`). `huber` leaves are not renewed: LightGBM 4.7.0 does not renew them
  either (a leaf of its `huber` objective is the mean clipped residual). `test_lightgbm_l1_leaves_are_residual_medians`.
- **Parameters.** `num_leaves` outside [2, 131072], `max_bin` below 2, `bagging_fraction` or `feature_fraction`
  outside (0, 1], `num_iterations` 0 and a learning rate that is not finite and positive are refused
  (`test_lightgbm_rejects_bad_parameters`).

Behaviour quoted from LightGBM 4.7.0 (R package, `exp3.R`/`exp4.R` of the T4 plan; not a pinned case):
`regression_l1`, one round, learning rate 1, `num_leaves = 4`, `min_data_in_leaf = 5`, `min_data_in_bin = 1` gives
exactly 4 distinct leaf outputs, each `median(y) + median(y - median(y))` over the leaf's rows (a leaf of 90 rows:
60.888763, where the mean would give 62.445831); `bagging_fraction = 0.5, bagging_freq = 0` gives the unbagged fit
for any seed (MSE 15.32438), `bagging_freq = 5` bags (37.56 / 38.94 for seeds 1 / 2).

Before T4: the leaf cap counted only finalised leaves, so trees grew past `num_leaves` (153 leaves at the default 31
on 4,000 rows; ML-31); `bagging_freq = 0` bagged every iteration and `freq = k` used all rows between the bags
(ML-95); `regression_l1` leaves were the mean sign of the residuals, so the fit moved at most `learning_rate` per
iteration (WT-1); `num_leaves` and `max_bin` below 2 panicked (ML-75).

## Wall-clock

`release`, `ci` features, the T4 boosting/forest fixes against the code before them (`34a4186a`), nine processes per
side interleaved, one core (`taskset`), one rayon thread, 2026-11-16, load average 22-42 on 12 cores (a loaded
machine: read ratios near 1 as within noise). Cells are the median of the process medians; "min ratio" is the ratio
of the minima. Design: n rows, 5 features uniform on [0, 10], `y = 2 x0 - x4 + (x1 - 5)^2 / 5` plus noise.

| `lightgbm()` | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| defaults (regression) | 1,000 | 175.3 | 146.4 | 0.84 | 0.87 |
|  | 10,000 | 2,218 | 414.8 | 0.19 | 0.19 |
| `regression_l1` | 1,000 | 142.4 | 148.0 | 1.04 | 1.00 |
|  | 10,000 | 1,662 | 406.1 | 0.24 | 0.25 |
| bagging 0.8, `bagging_freq` 1 | 1,000 | 153.0 | 154.1 | 1.01 | 0.97 |

Not like-for-like: the base grew trees past `num_leaves` (the speed-up at 10,000 is the leaf cap).

## What does not match LightGBM

- Bins are quantile midpoints of the distinct values (LightGBM's bin finder also merges sparse bins and keeps a
  zero bin).
- The bagging and feature draws use p2a's generator.
- No `min_sum_hessian_in_leaf`, `min_gain_to_split`, monotone constraints or categorical splits.
