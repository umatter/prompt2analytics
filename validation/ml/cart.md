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
  and `cpx_g[j] = cpx[j] * (t_g / n)`. A fold with no left-out row is skipped. `xerror` is the summed risk over the
  root risk and `xstd` its standard error. `min_xerror_cp` and `one_se_cp` are read off the table. With `xval = 10`
  (the default, as rpart) a call does 11 fits, the same work as rpart.
- **Prune.** `cart_prune(cp)` removes every subtree whose complexity is at or below `cp`, and recomputes the
  predictions, the importance and the cp table.
- **Importance.** The sum of the improvements of each feature's primary splits, in the nodes that remain.
- **Ties.** A class tie in a node takes the lowest class label.
- **Limits.** `max_depth` is at most 30 and 0 means 30, as rpart.

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

## Running

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::cart::tests::test_validate
Rscript validation/r/run.R --check
```
