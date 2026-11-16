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

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. Calls per process 2,001 /
401 / 41 (11 for `roc_auc` at n = 10,000), in µs. With four threads (cores 0-3) `fast_roc_auc_parallel` at n = 10,000
takes 745.7 µs at v0.5.0 and 782.4 µs here (1.05 [1.09]). Scores: logistic
of a normal with a class shift ("continuous"), and the same rounded to 0.1 ("ties"); 40% positives;
`roc_auc(.., Some(100))`.

| Function | Scores | n | base (µs) | this tree (µs) | ratio [min] |
|---|---|---:|---:|---:|---|
| `roc_auc` | continuous | 100 | 54.29 | 5.509 | 0.10 [0.10] |
|  |  | 1,000 | 2,699 | 43.46 | 0.02 [0.02] |
|  |  | 10,000 | 723,407 | 873.6 | 0.00 [0.00] |
|  | ties | 100 | 28.50 | 3.517 | 0.12 [0.12] |
|  |  | 1,000 | 281.8 | 17.45 | 0.06 [0.07] |
|  |  | 10,000 | 8,733 | 190.8 | 0.02 [0.02] |
| `fast_roc_auc` | continuous | 100 | 5.415 | 5.022 | 0.93 [0.92] |
|  |  | 1,000 | 59.16 | 44.60 | 0.75 [0.79] |
|  |  | 10,000 | 1,365 | 942.2 | 0.69 [0.69] |
|  | ties | 100 | 5.958 | 3.403 | 0.57 [0.57] |
|  |  | 1,000 | 52.19 | 19.49 | 0.37 [0.34] |
|  |  | 10,000 | 724.0 | 271.9 | 0.38 [0.36] |
| `fast_roc_auc_parallel` | continuous | 100 | 58.64 | 22.96 | 0.39 [0.45] |
|  |  | 1,000 | 122.3 | 89.01 | 0.73 [0.79] |
|  |  | 10,000 | 1,280 | 1,322 | 1.03 [1.00] |
|  | ties | 100 | 58.39 | 22.80 | 0.39 [0.30] |
|  |  | 1,000 | 121.3 | 87.12 | 0.72 [0.78] |
|  |  | 10,000 | 1,292 | 1,321 | 1.02 [0.97] |
