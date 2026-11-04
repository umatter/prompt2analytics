# SHAP (TreeSHAP and Kernel SHAP)

## Method

SHAP values are the Shapley values of a game `v(S)` defined by a model `f`, a point `x` and a
coalition `S` of features:
`phi_i = sum_{S not containing i} |S|! (M - |S| - 1)! / M! (v(S + i) - v(S))`. They add up:
`v(empty) + sum_i phi_i = f(x)`. `p2a_core::ml::shap` computes two games.

- **TreeSHAP** (`ShapMethod::Tree`; `shap_values_model`, `shap_tree_ensemble`,
  `shap_values_random_forest`): the tree path-dependent game `v(S) = E[f(x) | x_S]` (Lundberg et
  al. 2020, Algorithm 1: follow `x` at splits on features in `S`, otherwise weight both children
  by their training cover). Algorithm 2 (EXTEND/UNWIND over the unique-feature path, as shap's
  `tree_shap_recursive`) gives its Shapley values exactly, in O(L D^2) per tree and row. A
  forest's values are the tree average. The base value is `E[f]`, the mean over trees of
  `sum_leaf value * cover / root cover`, and the additivity check compares `base + sum(phi)` with
  the forest's predictions of the rows explained (tolerance `1e-8 * max(1, max |f|)`; `None` when
  the check is off). With `compute_interactions` the result also holds the Shapley interaction
  index (Lundberg et al. 2020, eqs. 8-9; shap's conditioning recursion): one symmetric M x M
  matrix per row whose rows sum to the SHAP values.
- **Kernel SHAP** (`ShapMethod::Kernel`; `kernel_shap`, `shap_kernel`, and for a forest
  `shap_values_model_with`): the interventional game `v(S) = mean_b f(x_S, b_notS)` over a
  background sample (Lundberg & Lee 2017; Janzing, Minorics & Blöbaum 2020), by
  shap.KernelExplainer's weighted regression on coalitions with the Shapley kernel applied once
  and the efficiency constraint `sum(phi) = f(x) - E f` imposed by eliminating the last feature;
  the regression goes through the rank-revealing weighted least squares of `linalg::solve`.
  `n_samples` is shap's `nsamples` (default `2 M + 2048`): every coalition is evaluated, and the
  values are exact, when `2^M - 2 <= n_samples`, which the default does for M <= 11. Otherwise
  shap's design with `n_samples` coalitions is used (complete subset sizes first, then sizes
  drawn in proportion to the kernel mass and members without replacement), from ChaCha8 seeded
  with `seed` (default 42) on stream i for row i, so each row is reproducible on its own. A
  sampled design that does not identify every feature is an error. M = 1 gives
  `phi = f(x) - E f`; M = 0 is an error.
- **Options.** `ShapMethod` (tree, kernel) and `FeaturePerturbation` (interventional,
  tree_path_dependent, observational) parse with aliases through the shared option tables. Each
  method computes its own game: TreeSHAP with `interventional` is an error that names the Kernel
  path, Kernel SHAP with `tree_path_dependent` or `compute_interactions` is an error, and
  `observational` (true conditional expectations) is not implemented and is an error. A Kernel
  setting (`n_samples`) given to TreeSHAP is reported as a warning.

## References

- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
  NeurIPS 30.
- Lundberg, S. M., Erion, G. G., & Lee, S. I. (2018). Consistent individualized feature
  attribution for tree ensembles. arXiv:1802.03888.
- Lundberg, S. M., Erion, G., Chen, H., et al. (2020). From local explanations to global
  understanding with explainable AI for trees. Nature Machine Intelligence 2, 56-67
  (Algorithms 1-2, eqs. 8-9).
- Janzing, D., Minorics, L., & Blöbaum, P. (2020). Feature relevance quantification in
  explainable AI: a causal problem. AISTATS 2020.
- Python `shap`: `KernelExplainer` (coalition design and `solve`) and `TreeExplainer`
  (`tree_shap_recursive`).

## R reference

No SHAP package is a case dependency (treeshap, fastshap, kernelshap, shapr are not installed),
so the references are the definitions, written out and enumerated over every coalition in
`validation/r/cases/ml_shap.R`. The R-pinned values are therefore the Shapley values of the
defined games, not the output of a SHAP package; no agreement with fastshap, treeshap or
kernelshap is claimed.

- **`shap_tree_rpart`** (dataset `validation/datasets/generated/shap_tree_sim.csv`, n = 80,
  `set.seed(4801)`, `y = 2 x0 + 3 x1 x2 + sin(3 x3) + N(0, 0.3^2)`): a tree grown by
  `rpart::rpart` 4.1.23 (anova, `cp = 0`, `minsplit = 8`, `minbucket = 3`, `maxdepth = 4`, no
  competitors or surrogates), exported as a node table (21 nodes, 11 leaves; x1 repeats on one
  path, which exercises UNWIND). Values: the leaf predictions, `E[f] = 2.4428755663159962`, the
  SHAP values of the 80 rows (row 1: `-0.23378818082422309, 0.025321160945115901,
  0.056711180654526704, -0.044476795688267304`) and of 5 points outside the data, and the
  interaction matrices of rows 1-10.
- **`shap_kernel_sim`** (dataset `validation/datasets/generated/shap_kernel_sim.csv`, 20 x 12
  standard normal, `set.seed(4802)`; rows 1-4 explained, rows 5-20 background): the
  interventional values of four fixed models, written out identically in the Rust tests:
  `f1 = exp(x0 / 2) + x0^2` (M = 1), `f4 = x0 x1 x2 + 2 x3 + 0.5 x0^2` (M = 4),
  `f11 = x0 x1 x2 + sin(x3) x4 + x5 x6 x7 x8 + sum_{j=5..10} (j - 4)/4 x_j` (M = 11) and
  `f12 = f11 + 0.5 x11 x0` (M = 12), with their base values (`m4_base = 0.73790812391210658`) and
  predictions.

Before this package, TreeSHAP was a heuristic (case tree row 1: `-0.965, -0.006, 0.996, -1.004`;
maximum error 2.85 over 80 rows, additivity error up to 1.82), the additivity check compared the
explained rows with the training predictions, and Kernel SHAP weighted its coalitions twice
(bias 0.339 on f4, 0.353 on f12; one feature divided by zero).

## Rust tests

R-pinned (registry `validation/methods/ml_shap.json`):

- `p2a-core::ml::shap::tests::test_validate_treeshap_rpart_tree`: leaf predictions exact,
  `E[f]`, SHAP values of the 80 rows and of the 5 new points.
- `p2a-core::ml::shap::tests::test_validate_treeshap_interactions_rpart_tree`: interaction
  matrices of rows 1-10.
- `p2a-core::ml::shap::tests::test_validate_kernel_shap_sim_m1`,
  `p2a-core::ml::shap::tests::test_validate_kernel_shap_sim_m4` and
  `p2a-core::ml::shap::tests::test_validate_kernel_shap_sim_m11_enumerated`: exact Kernel SHAP
  (one feature; every coalition, 14 and 2,046), base values and predictions.
- `p2a-core::ml::shap::tests::test_validate_kernel_shap_sim_m12_sampled`: shap's sampled design
  at the default budget (2,072 of 4,094 coalitions; sizes 1-3 complete), efficiency exact,
  values within the Monte Carlo bound, base and predictions exact, seeds reproducible.

Property tests against exact enumeration in Rust (not registered: they load no case):

- `p2a-core::ml::shap::tests::test_treeshap_matches_bruteforce_shapley` (random CART trees of
  depth 6, training rows and points outside the data range),
  `p2a-core::ml::shap::tests::test_treeshap_interactions_match_bruteforce`,
  `p2a-core::ml::shap::tests::test_tree_interactions_sum_to_shap_values`,
  `p2a-core::ml::shap::tests::test_treeshap_additivity_new_rows` (more rows than the training
  set), and through the public tree API
  `p2a-core::validation_ml_advanced::test_shap_forest_matches_bruteforce_shapley`;
- `p2a-core::ml::shap::tests::test_kernel_shap_exact_coalitions_match_bruteforce` (the
  `x0 x1 x2 + x3` example: `(1/3, 1/3, 1/3, 1)`),
  `p2a-core::ml::shap::tests::test_kernel_shap_single_feature`,
  `p2a-core::ml::shap::tests::test_kernel_design_matches_shap_explainer` (rows, complete sizes
  and fixed weight of shap's design),
  `p2a-core::ml::shap::tests::test_kernel_shap_small_budget_uses_the_sampled_design` and
  `p2a-core::ml::shap::tests::test_shap_values_model_with_kernel_matches_interventional_enumeration`
  (the forest's interventional game).

## Tolerance

- Exact algorithms against enumeration (`EXACT_SHAP`): 1e-10 relative, 1e-12 absolute; observed
  1.5e-14 or better (TreeSHAP about 7e-16).
- Sampled Kernel SHAP at M = 12 (`KERNEL_MC_TOL`): 0.15 absolute, twice the maximum of 100 R
  replications of the design (quantiles 0.047 / 0.061 / 0.071, maximum 0.073); efficiency 1e-10.
- Additivity: `1e-8 * max(1, max_i |f(x_i)|)` in the library (`ADDITIVITY_TOLERANCE`), 1e-10 or
  better in the tests.

## Not validated / differences

- Interventional TreeSHAP (the background-sample recursion) is not implemented; the Kernel path
  (`shap_values_model_with` with `ShapMethod::Kernel`) gives the forest's interventional values,
  exactly whenever every coalition fits the budget.
- Observational (conditional) SHAP is not implemented.
- The sampled Kernel path is a Monte Carlo estimate. p2a draws from ChaCha8 (seed, row stream),
  not numpy, so shap's samples are not reproduced, only its design (registry deviation under
  ML-100).
- shap's `l1_reg` feature selection for large M is not implemented (all features are kept).

## Run

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- ml::shap
Rscript validation/r/run.R --check
```
