# CART (`ml::cart`)

## Method

`cart()` and `run_cart()` fit a regression (`anova`) or classification (`gini`, `information`) tree as
`rpart` 4.1.23 does for continuous predictors, with unit weights, the default priors, 0/1 loss and no
surrogate or competitor splits. `cart_prune()` is `prune.rpart`.

## R reference

`rpart::rpart(xval = <fold vector>, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)` and `rpart::prune`.
Case file: `validation/r/cases/ml_cart_cp.R`. Registry: `validation/methods/ml_cart_cp.json`.

**Status: R-pinned** (see the registry for the quantities).

## How p2a computes it

- **Growth.** Rows are presorted once per feature and the sorted lists are partitioned stably at each split, so a
  node costs O(p n_node). The anova criterion is the SSE decrease. The class criteria are computed from integer
  class counts: Gini `t - (sum c_j^2) / t` and information `sum -c_j ln(c_j / t)`. A split is kept when it improves on
  the best so far strictly, so an exact tie goes to the first position and the first feature. The direction
  of a split and the node ids (`2k`, `2k + 1`) are rpart's.
- **Complexities.** The weakest-link pass gives each node its complexity. The cp list and the table of nested
  subtrees (CP, nsplit, rel error) are read off the tree, relative to the root risk. The class risk is the
  misclassification count.
- **Cross-validation.** Each fold refits on its training rows without sorting again and is run down on the left-out
  rows at the geometric means of consecutive cp values, with the fold weights t/n: `alpha_g = alpha * (t_g / n)`
  and `cpx_g[j] = cpx[j] * (t_g / n)`. `xerror` is the summed risk over the
  root risk and `xstd` its standard error. `min_xerror_cp` and `one_se_cp` are read off the table. With `xval = 10`
  (the default, as rpart) a call does 11 fits, the same work as rpart.
- **Prune.** `cart_prune(cp)` removes every subtree whose complexity is at or below `cp`, and recomputes the
  predictions, the importance and the cp table.
- **Importance.** The sum of the improvements of each feature's primary splits, in the nodes that remain.
- **Ties.** A class tie in a node takes the lowest class label.
- **Limits.** `max_depth` is at most 30 and 0 means 30, as rpart.
  A classification tree accepts at most max(2, min(1000, n/2)) classes (C5.0's cap; rpart has none). A target whose
  sum of squares lies outside `sqrt(f64::MIN_POSITIVE)` to `sqrt(f64::MAX)`, about 1.5e-154 to 1.3e154, is refused
  with a request to rescale it. Inside that range the table does not depend on the target's scale.

## Cases, values and tests

| Case | Call | Decisive values |
|---|---|---|
| `cart_cp_anova` | 300 rows, 4 predictors, `cp = 0.001`, a fold vector | 18 cp rows; `CP[1]` 0.21357711; `rel error[2]` 0.57284578; `xerror[1]` 1.00922464; root risk 336.33641957; min-xerror cp 0.01144455; 1-SE cp 0.02282230; pruned at 0.02: 5 rows |
| `cart_cp_class` | 300 rows, 3 predictors, classes 3/17, Gini and information | Gini: `CP` 0.35294118, 0.02941176, 0.01960784, 0.00980392, 0.001; `xerror` 1, 0.67647059, 0.74509804, 0.71568627, 0.78431373; root risk 102; 17 nodes. Information: `xerror[3]` 0.72549020 |
| `cart_class_ties` | 30 rows, tied classes, `maxdepth = 1`, no CV | root label 4 (the lowest); root deviance 20; row 30 has probabilities 0.5, 0.5, 0 |
| `cart_prune_xor` | 40 rows, XOR, `cp = 0.001`, pruned at 0.05 and 0.5 | `CP` 0.32988576, 0.001; `nsplit` 0, 3 (the root's improvement is small, the weakest-link pass keeps it); importance x0 1032.74142273, x1 0.36130987 |

Tests, in `crates/p2a-core/src/ml/cart.rs`:

- `test_validate_cart_cp_anova`, `test_validate_cart_cp_class`, `test_validate_cart_info_split`: cp table, frame,
  splits, importance, predictions, probabilities, root risk, min-xerror and 1-SE cp.
- `test_validate_cart_class_ties`, `test_validate_cart_class_codes_do_not_change_the_tree` (the class table under each
  recoding of the labels).
- `test_validate_cart_prune_recomputes`, `test_validate_cart_prune_xor`.

Tolerances (`validation/README.md`, coefficient row): n = 300 1e-8 relative, n = 30-40 1e-6 relative; counts, node
ids, split variables, directions and class labels exact.

Property tests that load no case: `test_cart_exact_tie_takes_the_first_candidate`,
`test_cart_rundown_stretches_equal_the_direct_loop` (the fast rundown against the direct loop),
`test_cart_rejects_bad_arguments`, `test_cart_random_folds_are_seeded_and_balanced`,
`test_cart_constant_target_is_a_single_leaf`, `test_cart_unsplittable_root_has_one_cp_row`,
`test_cart_method_parses_aliases_and_rejects_unknown`.

## Deviations and what is not implemented

- Surrogate and competitor splits and missing feature values (ML-125). `max_surrogate` (default 5) and
  `use_surrogate` (default true) are accepted and have no effect, and non-finite features are refused. With complete
  data surrogates do not change rpart's tree, but rpart's default `variable.importance` adds the surrogate splits'
  improvements. p2a's importance is rpart's with `maxsurrogate = 0`, so a default-to-default comparison with rpart
  differs in the importance only.
- Factor predictors. Integer columns are treated as continuous, as rpart does for numeric columns.
- Case weights, priors and a loss matrix.
- Random folds use p2a's LCG, not R's RNG. Pass `folds` to reproduce rpart.
- For a tree without a split, `xerror` and `xstd` are `None` where rpart prints 0.
- A constant target gives an empty table and a warning where rpart prints NaN.
- rpart's altered priors (`aprior`, 1 up to an ulp) are not reproduced; exact gini ties break to the first feature
  and position (rpart: by rounding). On random designs with heavily tied predictors, the cross-validated columns can
  differ from rpart where a fold tree has a tie that rounding decides; the frames were identical on all 40 designs
  checked, and 4 of them differed in `xerror`/`xstd` by up to 3.3%. These are tie-order differences, and no pinned
  case is affected.

## Wall-clock

`release`, `ci` features, against `fe13b889`, five rounds with base and this tree interleaved per row (order
alternating), one P core (`taskset`), one rayon and polars thread, 2026-11-08, load average 1.3-2.8. Cells are
the median of the round medians; "ratio" is this tree over the base, "min ratio" the ratio of the minima. R
(4.3, one BLAS thread) runs once per round on the same CSV, timed with `proc.time()`.

CART: the `noisy` design of the tests with p = 10 (gini: y > 2.5), `cart()` defaults otherwise; R is
`rpart(y ~ ., d, control = rpart.control(xval = ., maxsurrogate = 0, maxcompete = 0))`. The `ml_benchmark`
rows use that example's data and settings (depth 5, `min_split` 20, `min_bucket` 7, cp 0.01, default `xval`).

| Row | n | base (ms) | this tree (ms) | ratio | min ratio | R (ms) | this tree / R |
|---|---:|---:|---:|---:|---:|---:|---:|
| anova, `xval = 0` | 100 | 0.054 | 0.026 | 0.49 | 0.55 | 1.22 | 0.02 |
|  | 1,000 | 1.11 | 0.475 | 0.43 | 0.42 | 2.06 | 0.23 |
|  | 10,000 | 14.39 | 5.14 | 0.36 | 0.36 | 11.40 | 0.45 |
| anova, `xval = 10` (default) | 100 | 0.062 | 0.269 | 4.33 | 5.09 | 1.60 | 0.17 |
|  | 1,000 | 1.13 | 2.99 | 2.66 | 2.68 | 5.27 | 0.57 |
|  | 10,000 | 14.41 | 28.22 | 1.96 | 1.97 | 43.50 | 0.65 |
| gini, `xval = 0` | 100 | 0.104 | 0.028 | 0.27 | 0.28 | 1.36 | 0.02 |
|  | 1,000 | 1.97 | 0.497 | 0.25 | 0.25 | 2.53 | 0.20 |
|  | 10,000 | 25.79 | 5.55 | 0.21 | 0.21 | 16.00 | 0.35 |
| gini, `xval = 10` (default) | 100 | 0.108 | 0.228 | 2.12 | 2.17 | 1.82 | 0.12 |
|  | 1,000 | 1.96 | 3.19 | 1.63 | 1.65 | 8.00 | 0.40 |
|  | 10,000 | 25.59 | 31.81 | 1.24 | 1.25 | 73.00 | 0.44 |

| Row | n | base (ms) | this tree (ms) | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| `ml_benchmark` CART regression | 1,000 | 1.22 | 3.59 | 2.95 | 2.98 |
|  | 5,000 | 7.30 | 18.10 | 2.48 | 2.49 |
|  | 10,000 | 15.06 | 35.37 | 2.35 | 2.36 |
|  | 50,000 | 87.28 | 196.1 | 2.25 | 2.25 |
| `ml_benchmark` CART classification | 1,000 | 2.15 | 3.37 | 1.56 | 1.57 |
|  | 5,000 | 13.23 | 16.89 | 1.28 | 1.28 |
|  | 10,000 | 26.88 | 34.41 | 1.28 | 1.27 |
|  | 50,000 | 156.9 | 212.7 | 1.36 | 1.33 |

Without cross-validation a fit is 2-5x faster than the base. The base ran no cross-validation (its `xerror` and
`xstd` columns were not cross-validated, and its `xval = 10` time equals its `xval = 0` time); the default
`xval = 10` now runs the 11 fits rpart runs, the default kept by the maintainer (2026-10-30) against an estimate of
1.5-5x. Measured: 1.24-4.33x of the base and 0.13-0.65x of rpart; the `ml_benchmark` rows, which run the default
10 folds, 1.28-2.95x. These ratios were signed off by the maintainer on 2026-11-08.

## Running

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::cart::tests::test_validate
Rscript validation/r/run.R --check
```
