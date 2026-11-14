# Quantile regression forest

`crates/p2a-core/src/ml/quantile_rf.rs` (`quantile_rf`, `predict_quantiles`). **Not R-pinned**: `quantregForest`
is not installed on the validation machine, so no R reference case exists and the method has no registry entry.
Its tree growth and impurity importance are pinned against rpart and randomForest in
[`tree_importance.md`](tree_importance.md).

## Method (Meinshausen 2006)

The trees are grown as in a random forest, each on a bootstrap sample. After growing, every one of the `n`
training rows is dropped down each tree and the rows of each leaf are recorded. For a point `x` and `k` trees,
training row `i` gets the weight

    w_i(x) = (1/k) sum_t 1{i in L_t(x)} / |L_t(x)|

with `L_t(x)` the training rows in the leaf of tree `t` that `x` falls in; the weights sum to one. The
conditional distribution is `F(y | x) = sum_i w_i(x) 1{y_i <= y}`, and the alpha-quantile is eq. (6),

    Q_alpha(x) = inf{y : F(y | x) >= alpha}

a training response, with no interpolation; at a cumulative weight equal to alpha the smaller response is taken.

- **In-sample predictions** (`QuantileRfResult::predictions`): every tree and every training row, as
  `quantregForest`'s `predict` on the training data.
- **Out-of-bag predictions** (`oob_predictions`, `oob_error`): for training row `i`, the trees where `i` is out
  of bag, with row `i` left out of their leaves; a tree whose leaf holds only row `i` contributes nothing, and
  the weights are renormalised over the contributing trees. NaN when no tree contributes.
- **New data**: `predict_quantiles`/`predict_quantiles_at` use the training responses the model keeps;
  `predict_quantiles_with_y`/`_with_y_at` take them as an argument.

Before this was fixed, every tree's leaf members (bootstrap draws, with duplicates) were pooled into one sample
and a type-7 quantile was interpolated from it, so trees with large leaves dominated, and `predict_quantiles`
always returned an error because the model did not keep `y`.

## Tests

| Test | What it checks |
|---|---|
| `test_qrf_meinshausen_two_stumps` | Two stumps on 12 rows (splits at 4.5 and 8.5): weights and eq. (6) quantiles at x = 3, 6, 10, alpha 0.1-0.9, equal to the base-R computation exactly (dyadic weights); OOB through one tree |
| `test_qrf_weights_sum_to_one` | 50-tree forest, five points: the weights sum to one, and each tree adds `1/k` |
| `test_qrf_one_tree_is_its_leaf_ecdf` | One tree: each prediction is eq. (6) on the training rows of its leaf |
| `test_qrf_oob_excludes_the_row` | OOB prediction equals a brute-force reference over the out-of-bag trees without the row |
| `test_predict_quantiles_uses_stored_y` | `predict_quantiles` equals `predict_quantiles_with_y` with the training `y`, bit for bit |
| `test_quantile_rf_seeded_output_is_unchanged` | The seeded forest's impurity and leaf counts are those of the earlier builder, bit for bit |
| `test_qrf_refuses_overflowing_target` | A target whose sum of squares overflows is refused |

Reference values (base R, `qrf.R`): at x = 3 the weights are 3/16 on rows 1-4 and 1/16 on rows 5-8, and the
quantiles at alpha = 0.1, 0.25, 0.5, 0.75, 0.9 are 1.4, 1.5, 3.1, 4.1, 8.9; at x = 6, 1.5, 3.2, 4.1, 7.9, 9.2; at
x = 10, 3.2, 3.5, 4.6, 7.9, 8.9.

## Not compared with quantregForest

`quantregForest`'s defaults and options (node size, `mtry`, `nthreads`, `what`) are not checked here, and its
random draws cannot be reproduced. The method and its OOB rule follow the paper and the package's documentation.

## Wall-clock

`release`, p2a-core with the `ci` features, against `f80b0131` (v0.5.0), five rounds with the sides interleaved per
row (order alternating per round), one core (`taskset -c 2`), one rayon thread, 2026-11-14. **Preliminary**: the load
average was 33-70 during the runs (other builds on the machine), so ratios within about 0.05 of 1 are noise. Cells are
the median of the round medians (each the median of 1-21 calls after a warm-up); "ratio" is this tree over the base,
"min ratio" the ratio of the minima. Friedman-1 data, p = 5, noise sd 1. No R timing (quantregForest is not installed).

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| fit with in-sample and OOB quantiles, 100 trees | 100 | 16.3 | 15.0 | 0.92 | 0.92 |
|  | 1,000 | 302 | 304 | 1.01 | 1.02 |
|  | 10,000 | 4,865 | 5,330 | 1.10 | 1.09 |
| the same, 500 trees (default) | 1,000 | 1,770 | 1,666 | 0.94 | 0.93 |
| quantiles of 1,000 new rows (base: `predict_quantiles_with_y`) | 100 | 50.2 | 34.4 | 0.68 | 0.67 |
|  | 1,000 | 98.9 | 83.8 | 0.85 | 0.88 |
|  | 10,000 | 192 | 188 | 0.98 | 1.03 |

Peak RSS of one 500-tree fit at n = 10,000 (`/usr/bin/time -v`, whole process): 205 MB base, 103 MB this tree (each
tree records its leaf members once, as `u32` offsets, and its in-bag rows as a bit set, instead of leaf lists of
bootstrap draws).
