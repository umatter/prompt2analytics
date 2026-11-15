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
| `test_qrf_meinshausen_two_stumps` | Two stumps on 12 rows (splits at 4.5 and 8.5): weights and eq. (6) quantiles at x = 3, 6, 10, alpha 0.1-0.9, equal to the base-R computation exactly (dyadic weights); one tree alone (tree B) |
| `test_qrf_weights_sum_to_one` | 50-tree forest, five points: the weights sum to one, and each tree alone gives `1/|L|` over its leaf, whose sum over the trees divided by `k` is the forest's weight at every rank |
| `test_qrf_one_tree_is_its_leaf_ecdf` | One tree: each prediction is eq. (6) on the training rows of its leaf |
| `test_qrf_oob_excludes_the_row` | OOB prediction equals a brute-force reference over the out-of-bag trees without the row |
| `test_predict_quantiles_uses_stored_y` | `predict_quantiles` equals `predict_quantiles_with_y` with the training `y`, bit for bit |
| `test_quantile_rf_seeded_output_is_unchanged` | The seeded forest's impurity and leaf counts are those of the earlier builder, bit for bit |
| `test_qrf_refuses_overflowing_target` | A target whose sum of squares overflows is refused |

Reference values, computed in base R: each stump puts weight `1/|L|` on the training rows of the leaf `x` falls in,
the forest averages the two, and eq. (6) is the smallest `y` whose cumulative weight in ascending `y` reaches alpha:

```r
x <- 1:12; y <- c(3.1, 1.4, 4.1, 1.5, 9.2, 6.5, 3.5, 8.9, 7.9, 3.2, 3.8, 4.6)
leaf_w <- function(s, x0) { m <- if (x0 <= s) x <= s else x > s; m / sum(m) }
w <- function(x0) (leaf_w(4.5, x0) + leaf_w(8.5, x0)) / 2
q <- function(x0, a) { o <- order(y); y[o][which(cumsum(w(x0)[o]) >= a)[1]] }
sapply(c(0.1, 0.25, 0.5, 0.75, 0.9), function(a) q(3, a))
```

At x = 3 the weights are 3/16 on rows 1-4 and 1/16 on rows 5-8, and the
quantiles at alpha = 0.1, 0.25, 0.5, 0.75, 0.9 are 1.4, 1.5, 3.1, 4.1, 8.9; at x = 6, 1.5, 3.2, 4.1, 7.9, 9.2; at
x = 10, 3.2, 3.5, 4.6, 7.9, 8.9.

## Not compared with quantregForest

`quantregForest`'s defaults and options (node size, `mtry`, `nthreads`, `what`) are not checked here, and its
random draws cannot be reproduced. The method and its OOB rule follow the paper and the package's documentation.

## Wall-clock

`release`, p2a-core with the `ci` features, against `f80b0131` (v0.5.0); quiet re-time, 2026-11-15 (this tree at
`2d0bb8b6`): one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; base and this tree interleaved per row
with the order rotated per round, each process the median of its calls after a warm-up (calls calibrated to about 0.8
s per process); cells are the median of the process medians over five rounds (nine or ten for rows within 1.05-1.15x);
1-minute load average 1.0-3.0. "ratio" is this tree over the base, "min ratio" the ratio of the minima.
Friedman-1 data, p = 5, noise sd 1. No R timing (quantregForest is not installed).

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| fit with in-sample and OOB quantiles, 100 trees | 100 | 11 | 9.3 | 0.85 | 0.84 |
|  | 1,000 | 173 | 163 | 0.94 | 0.94 |
|  | 10,000 | 2,909 | 2,821 | 0.97 | 0.97 |
| the same, 500 trees (default) | 1,000 | 959 | 859 | 0.90 | 0.91 |
| quantiles of 1,000 new rows (base: `predict_quantiles_with_y`) | 100 | 31 | 14.9 | 0.48 | 0.48 |
|  | 1,000 | 51.9 | 32.8 | 0.63 | 0.63 |
|  | 10,000 | 106 | 79.6 | 0.75 | 0.74 |

Peak RSS of one 500-tree fit at n = 10,000 (`/usr/bin/time -v`, whole process): 205 MB base, 103 MB this tree (each
tree records its leaf members once, as `u32` offsets, and its in-bag rows as a bit set, instead of leaf lists of
bootstrap draws). The 100-tree fit at n = 10,000 (1.10x on the loaded run) was signed off on 2026-11-14 (rule 5); the quiet re-time
puts it at 0.97x, so it drops off. Quiet peak RSS (median of three): 205 MB base, 103 MB this tree.
