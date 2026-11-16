# ROC curve and AUC (`ml::evaluation::roc_auc`, `ml::evaluation_fast`)

## Method

`roc_auc(predictions, actual, n_thresholds)` returns the ROC curve, the area under it and Youden's optimal
threshold with the classification metrics there. `fast_roc_auc(predictions, actual)` returns the same curve, AUC
and optimum; `fast_roc_auc_parallel(predictions, actual)` returns the AUC alone, sorting in parallel. All three run
on one tie-aware sweep (`evaluation::roc_sweep`) and agree field by field.

## Convention (pROC)

- **Rule.** An observation is classified positive when `score >= threshold`. The positive class is the larger of
  the two label values; labels are matched exactly.
- **Curve.** A first point at `+inf` (nothing positive, the point (0, 0)), then one point per distinct score in
  decreasing order, each after its whole tie group is counted. Ties are exact equality (`-0.0 == 0.0`). k distinct
  scores give k + 1 points: the (fpr, tpr) pairs of `pROC::roc()`, whose k + 1 thresholds are `-Inf`, the midpoints
  and `Inf`. The last point is (1, 1).
- **AUC.** `sum_g fp_g (2 tp_above + tp_g) / (2 n_pos n_neg)` over the tie groups, in integers: the Mann-Whitney
  statistic with half credit for ties, equal to `pROC::auc()` and to `wilcox.test`'s W / (n1 n0).
- **Youden's optimum.** J = sensitivity + specificity - 1 is compared in integers (`tp n_neg - fp n_pos`), so equal
  values tie exactly; among tied optima the lowest threshold is reported (pROC lists tied optima in increasing
  threshold order). `optimal_threshold` is the observed score that starts the optimal rule; pROC reports the
  midpoint below it (registry deviation, ML-92).
- **`n_thresholds`** (`roc_auc` only) caps the returned curve: `None` returns every point; `Some(m)`, m >= 2,
  returns at most m + 1 points, always the `+inf` point, the last point and the optimum, the rest evenly spaced by
  index. The AUC and the optimum always use every point. `Some(0)` and `Some(1)` are refused.
- **Refused:** unequal lengths, an empty input, NaN scores or labels (missing values), infinite scores (pROC's
  `roc()` stops on them), infinite labels, and anything but exactly two label values.
- **Cost.** One sort, O(n log n), and an O(n) pass.

## R reference

`pROC::roc(label, score, levels = c(neg, pos), direction = "<")`, `pROC::auc`, `pROC::coords(r, "best",
best.method = "youden")`, pROC 1.19.1, R 4.3.3. Case file: `validation/r/cases/ml_evaluation_t4.R`. Registry:
`validation/methods/ml_evaluation_t4.json`.

**Status: R-pinned** (pROC; see the registry for the quantities).

## Cases, values and tests

| Case | Data | Decisive values |
|---|---|---|
| `ml_roc_small` | the findings' four score vectors (`roc_small.csv`, set 1-4) | ml91 (ties at 1.0 across classes): AUC 0.72222222, optimum 0.6 (sensitivity 1, specificity 0.66666667); ml91u (the same ranking on an unbounded scale): the same, optimum 3; ml92 (perfect separation): AUC 1, optimum 0.7; ml93 (a tie group with both classes): AUC 0.77777778, two tied optima, 0.5 reported |
| `ml_roc_ties` | 240 rows, classes 3/7, seed 20261115 (`roc_ties.csv`) | probabilities on a 0.1 grid (54 rows tie at 1.0): AUC 0.78635635, 12 points, optimum 0.6 (sensitivity 0.86725664, specificity 0.55905512); integer scores: AUC 0.73649920, 15 points, optimum 1 |

R-pinned tests: `test_validate_roc_auc_small`, `test_validate_roc_auc_ties` (`evaluation.rs`),
`test_validate_fast_roc_auc_small`, `test_validate_fast_roc_auc_ties` (`evaluation_fast.rs`; the parallel AUC too).
Tolerances (`validation/README.md`, coefficient row): n = 6 1e-6 relative, n = 240 1e-8 relative; thresholds exact.

Property tests: `roc_auc_equals_the_pair_count_on_tied_scores` (50 seeded tied draws: the AUC against the naive
pair count, every point against a direct count), `roc_auc_refuses_infinite_scores`, `roc_auc_groups_signed_zeros`,
`roc_auc_reports_the_lowest_of_tied_optima`, `roc_auc_caps_the_returned_points`,
`fast_roc_auc_parallel_does_not_depend_on_the_order_of_ties`, `fast_roc_auc_parallel_refuses_three_classes`,
`fast_and_exact_agree_on_tied_scores`.

## Timing

Preliminary (lane timing, 2026-11-16): release builds, `--no-default-features --features ci`, base `f80b0131`
(v0.5.0) against this tree; one core (`taskset -c 8`), `RAYON_NUM_THREADS=1`, 9 processes per side interleaved,
each the median of 2,001 / 401 / 41 calls (11 for the base `roc_auc` at n = 10,000) after 10 warm-ups; cells are
the median of the process medians, in µs. Machine shared with other builds (load average 3-18). Scores: logistic
of a normal with a class shift ("continuous"), and the same rounded to 0.1 ("ties"); 40% positives;
`roc_auc(.., Some(100))`.

| Function | Scores | n | base (µs) | this tree (µs) | ratio |
|---|---|---:|---:|---:|---:|
| `roc_auc` | continuous | 100 | 74.1 | 6.9 | 0.09 |
|  |  | 1,000 | 3,647 | 57.9 | 0.02 |
|  |  | 10,000 | 747,648 | 1,173 | 0.002 |
|  | ties | 100 | 38.6 | 4.3 | 0.11 |
|  |  | 1,000 | 363.0 | 22.8 | 0.06 |
|  |  | 10,000 | 10,184 | 264.9 | 0.03 |
| `fast_roc_auc` | continuous | 100 | 7.4 | 7.3 | 0.98 |
|  |  | 1,000 | 83.1 | 65.3 | 0.79 |
|  |  | 10,000 | 1,968 | 1,328 | 0.68 |
|  | ties | 100 | 7.2 | 4.3 | 0.59 |
|  |  | 1,000 | 64.5 | 23.2 | 0.36 |
|  |  | 10,000 | 1,045 | 415.0 | 0.40 |
| `fast_roc_auc_parallel` | continuous | 100 | 62.8 | 24.8 | 0.39 |
|  |  | 1,000 | 123.8 | 91.7 | 0.74 |
|  |  | 10,000 | 1,375 | 1,402 | 1.02 |
|  | ties | 100 | 59.7 | 23.6 | 0.40 |
|  |  | 1,000 | 122.1 | 87.7 | 0.72 |
|  |  | 10,000 | 1,319 | 1,330 | 1.01 |
