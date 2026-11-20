# Conditional Inference Trees (CTree)

## Method

`ctree` follows `partykit::ctree` (Hothorn, Hornik & Zeileis 2006; partykit 1.3.1, libcoin 1.0.13) for numeric
inputs. At each node of `n` rows:

1. **Variable tests.** Each input is tested for independence from the response with the conditional linear
   statistic of Strasser and Weber (numeric input untransformed; for the response `h(y) = y`, or class indicators
   for a classification). `quadratic`: `(n - 1) r^2` for a regression (1 df) and `(n - 1) SSB / Sxx` for a
   classification (df = classes present in the node - 1), with asymptotic chi-squared p-values. `maximum`
   (regression only): `|z|` with a two-sided normal p-value.
2. **Untestable inputs.** An input whose statistic has variance below `sqrt(eps)` (partykit's absolute tolerance,
   copied as is) is not tested: its statistic and p-values are NaN (`null` in JSON), and it does not count in `m`.
3. **P-values.** `bonferroni`: partykit's adjustment on the log scale, `1 - (1 - p)^m` over the `m` tested
   inputs (`min(1, m p)` differs from it). `univariate`: no adjustment. `montecarlo`: the share of `nresample`
   (default 9,999) permutations of the response whose statistic reaches the observed one, unadjusted, as partykit
   does when `testtype = "MonteCarlo"` alone. A permuted statistic counts when it is at least the observed one
   minus `sqrt(eps) max(|observed|, 1)`, so a permutation that ties mathematically is not lost to summation order.
   Each permutation draws from a ChaCha8 stream derived from the seed (42 when none is given), the node and the
   permutation index.
4. **Selection and stopping.** The input with the smallest adjusted p-value (on an exact tie the larger statistic,
   then the first). A node is a leaf when no adjusted p-value is below `1 - mincriterion`, when it has fewer than
   `minsplit` rows, or at `maxdepth`. There is no `n <= 2 minbucket` stop.
5. **Node minimum.** Each child keeps at least `max(minbucket, ceil(0.01 n))` rows (`minprob = 0.01`).
6. **Split point.** The observed value `u` of the selected input (left: `x <= u`, not a midpoint) that maximises
   the two-sample statistic of `I(x <= u)`: the variance-reduction cut for a regression, Pearson's X^2 for a
   classification; the first cut on ties. When the selected input has no admissible cut, the next significant
   input is tried (`splittry = 2`).
7. **Leaves.** The mean, or the class with the largest share (the first on ties, as `which.max`).

## R Package Reference

- **Package**: `partykit` 1.3.1 (libcoin 1.0.13)
- **Function**: `ctree(formula, data, control = ctree_control())`
- **Cases**: `validation/r/cases/ml_trees_t4.R` (`ctree_reg_t4`, `ctree_cls_t4`, `ctree_null_root_t4`,
  `ctree_montecarlo_t4`) and `validation/r/cases/numeric_ctree.R` (`tail_ctree_root`)

## What is pinned

R-pinned (registry `validation/methods/ml_trees_t4.json`):

- `ctree_partykit_trees`: `test_validate_ctree_regression_tree_t4`, `test_validate_ctree_classification_tree_t4`
  and `test_validate_ctree_null_root_t4` pin the tree shape, node sizes, split variables and breaks, split
  statistics and adjusted p-values, terminal predictions and class probabilities, the root statistics with raw and
  adjusted p-values, and the fitted values and classes.
- `ctree_montecarlo_pvalues`: `test_validate_ctree_montecarlo_t4` pins the root statistics, and the Monte Carlo
  p-values within Monte Carlo error (two independent permutation draws).
- `ctree_root_statistics` (`validation/methods/numeric_hygiene.json`, case `tail_ctree_root`): root p-values
  below 1e-16 and the choice between them, `test_validate_ctree_root_tail_tie_break`.

Properties: `test_ctree_classification_statistic_is_the_anova_quadratic_form`, `test_ctree_null_root_split_rate`,
`test_ctree_constant_input_is_not_tested`, `test_ctree_splits_a_node_of_twice_minbucket`,
`test_ctree_minprob_raises_the_node_minimum`, `test_ctree_class_ties_take_the_first_class`,
`test_ctree_split_matches_naive_reference`, `test_ctree_montecarlo_is_reproducible_and_seeded`,
`test_ctree_montecarlo_counts_tied_permutations`, `test_ctree_montecarlo_honours_cancellation`, and parameter validation
(`test_ctree_rejects_minsplit_below_twice_minbucket`, `test_ctree_minbucket_zero_acts_as_one`,
`test_ctree_options_parse_aliases_and_reject_unknown`, `test_ctree_maximum_teststat_is_not_the_quadratic_one`).
`minsplit` below `2 * minbucket` is refused, because no node of that size can be split into two children of
`minbucket` rows.

## What is not R parity

- Variable importance: `Sum(1 - adjusted p)` over the splits on each input, not partykit's permutation `varimp`.
- Refused: `testtype = "Teststatistic"`, and `teststat = "maximum"` for a classification (it needs `pmvnorm`).
- Not supported: categorical inputs, surrogate splits, missing values in the inputs, and partykit's MIA.

## DGP property tests

`test_ctree_regression_properties`, `test_ctree_classification_properties` and `test_ctree_predict_properties`
(`crates/p2a-core/tests/validation_ml_advanced.rs`) check DGP recovery by thresholds; they are property tests, not
R parity (X-10).

## Target types and missing values

- **Target.** The task comes from the target, as in partykit. A numeric target (integer and decimal columns) fits a
  regression tree unless `classification = true` (`CtreeConfig::classification = Some(true)`; R: `factor(y)`),
  when its distinct values are the classes. A text, categorical or boolean target always fits a classification
  tree, with the class codes 0..K-1 in R `factor()` order reported in `CtreeResult::class_names` (JSON
  `class_names`); `classification = false` is refused for it. A classification has at most
  `max(2, min(1000, n / 2))` classes, `n / 2` rounded down (the cap of C5.0 and CART).
- **Missing values.** A missing value (null or NaN) in the target or a feature is an error naming the column (Fail;
  no surrogate splits). `±Inf` is refused as well.
- **Tests.** `test_ctree_numeric_target_is_regression`, `test_ctree_large_values_are_regression`,
  `test_ctree_classification_on_request`, `test_ctree_class_cap`,
  `test_ctree_class_rule_message_states_the_rounded_bound` (`crates/p2a-core/src/ml/ctree.rs`),
  `test_na_ctree_class_labels`, `test_na_ml_wrappers_int64_target_equals_float64` and
  `test_na_ml_wrappers_refuse_missing_values` (`crates/p2a-core/src/ml/extract.rs`),
  `test_na_mcp_tree_tools_integer_and_missing_targets`, `test_mcp_c50_ctree_text_targets` and
  `test_mcp_ctree_numeric_target_and_classification_flag` (`crates/p2a-mcp/src/tools/handlers/ml.rs`).

## Timing

Against v0.5.0 (`f80b0131`). Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the
integrated branch, `--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread
for polars and BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating
(nine for rows within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the
processes, the ratio of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. Fits
per process 201 / 51 / 21 at n = 100 / 1,000 / 10,000. Inputs: p = 5 uniform on [0, 10); regression
`y = 3 (x1 > 5) + 0.5 x2 + U(-1, 1)`; classification K = 3 from x1 and x2 with 15% label noise. Defaults
(Bonferroni, quadratic). Nodes are the nodes of the fitted tree; per node is the fit time divided by them.

| Fit | n | v0.5.0 | nodes | now | nodes | per fit | per node | R `partykit::ctree` (nodes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regression | 100 | 80.4 µs | 13 | 30.8 µs | 13 | 0.38x | 0.38x | 38 ms (15) |
| regression | 1,000 | 1.7 ms | 33 | 424 µs | 35 | 0.25x | 0.24x | 108 ms (31) |
| regression | 10,000 | 21.1 ms | 67 | 5.53 ms | 67 | 0.26x | 0.26x | 364 ms (75) |
| classification | 100 | 93.7 µs | 7 | 32.3 µs | 5 | 0.34x | 0.48x | 29 ms (5) |
| classification | 1,000 | 1.05 ms | 5 | 234 µs | 5 | 0.22x | 0.22x | 40 ms (5) |
| classification | 10,000 | 14.8 ms | 9 | 3.33 ms | 5 | 0.22x | 0.40x | 179 ms (7) |

v0.5.0 fitted the classification rows as classification trees too (integer class codes), with its own tests and
splits; the R column uses R's generator on the same design, so its trees differ.

Monte Carlo p-values (`testtype = montecarlo`, new, no base; regression design; median of 3-5 processes; lane timing
on the loaded machine, not repeated in the quiet re-time):

| n | nresample = 999 | nodes | nresample = 9,999 (default) | nodes | R, nresample = 999 |
|---:|---:|---:|---:|---:|---:|
| 100 | 10.2 ms | 13 | 98 ms | 13 | 79 ms |
| 1,000 | 147 ms | 39 | 1.24 s | 39 | 0.78 s |
| 10,000 | 1.90 s | 109 | 18.6 s | 111 | 4.58 s |

At n = 10,000 the default 9,999 permutations take 18.6 s, inside the hosted 60 s deadline; the call is
interruptible at every permutation. Peak RSS of a process fitting twice at n = 10,000: regression 4.0 MB (v0.5.0)
and 3.7 MB (now), classification 3.9 MB and 3.9 MB, Monte Carlo (999) 4.2 MB (lane). Quiet re-time, regression at
n = 10,000, one process per side, median of three: 3.9 MB (v0.5.0) and 3.8 MB (now).

## Release notes

- The node statistics, p-values, split variables and break points now follow partykit: the conditional linear
  statistic replaces the rank statistic, the Bonferroni adjustment is `1 - (1 - p)^m`, breaks are observed values
  instead of midpoints, the node minimum includes `minprob`, a node of exactly `2 minbucket` rows can split, and
  class ties go to the first class. Trees fitted before differ.
- A numeric target is now a regression tree; `classification: true` restores a classification on class codes.
  Text targets with up to `max(2, min(1000, n / 2))` classes are accepted (before: at most 10).
- `testtype = "montecarlo"` is available (`n_permutations` through MCP).
- The root tests of every input are reported (`root_statistics`, `root_p_values`, `root_p_values_adjusted`; JSON
  `root_tests`).
