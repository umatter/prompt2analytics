# Support vector classification

Two solvers live in `crates/p2a-core/src/ml/`. `fast_svm` (`svm_fast.rs`) is LIBSVM's C-SVC solver
and is compared with `e1071::svm`. `linear_svm` (`svm.rs`, the MCP tool `ml_svm` and `p2a ml svm`) is its
linear-kernel front and is pinned to e1071 too. `kernel_svm` (`svm.rs`, a simplified SMO, library only) is not
pinned.

## `fast_svm`: LIBSVM C-SVC

Second-order working-set selection, the least-recently-used kernel-column cache bounded by
`cache_size_mb`, shrinking with gradient reconstruction, `calculate_rho`, and `min_working_set`, a
floor on the active set (0 is LIBSVM's behaviour; at `n` every variable stays active). The larger
label is +1, and predictions come back in the original labels.

The cache holds single-precision entries, as LIBSVM's `Qfloat`, so `cache_size_mb` holds as many
columns as in e1071. The solution is that of the rounded problem, about 1e-7 relative from the
exact one: far below the default tolerance 1e-3, and visible only at a tolerance such as this
comparison's 1e-10.

### R reference

`validation/r/cases/ml_ppr_svm.R`, case `svm_libsvm`, dataset `validation/datasets/generated/svm_sim.csv`
(160 rows; a noisy ring `cls` for the radial kernel, a noisy half-plane `cls_lin` for the linear
kernel, labels 1 and 2). `e1071::svm(type = "C-classification", gamma = 0.5, scale = FALSE,
tolerance = 1e-10, cachesize = 40)`, oriented so that label 2 is positive. Five runs: radial and
linear at cost 0.1 and 10, and radial at cost 10 without shrinking.

### Tests and tolerances

In `ml::svm_fast::tests`, p2a at the same tolerance 1e-10:

| Test | Compared with e1071 |
|---|---|
| `test_validate_fast_svm_libsvm_radial_c01` | decision values, bias, dual coefficients at `Tol::iterative(4e-7)`; support-vector count and indices, predictions exact |
| `test_validate_fast_svm_libsvm_radial_c10` | the same, with and without shrinking; the two runs agree at the same tolerance |
| `test_validate_fast_svm_libsvm_linear_c01`, `test_validate_fast_svm_libsvm_linear_c10` | decision values, bias, `w` at `Tol::iterative(4e-7)`; predictions exact |

`4e-7` is 10x the largest difference measured in the five runs (3.3e-8, a radial dual coefficient; the
largest on decision values is 3.8e-9). The rows stay in their original order and e1071 groups them by
class, so the two take different paths to one optimum: they are not equal digit by digit.

Every run also checks, on the exact kernel, `0 <= alpha <= C`, `sum alpha y = 0` to 1e-12, the
optimality condition (up to the single-precision rounding above), and that the dual objective of p2a's
alphas is at least that of e1071's, less 1e-9 relative.

Without R: `test_fast_svm_labels_are_mapped_and_returned` (any two-valued coding, a reversed coding
negates the decision values), `test_kernel_cache_evicts_least_recently_used`,
`test_fast_svm_kernel_cache_is_bounded` (the cache size changes neither the iterations nor the solution),
`test_fast_svm_min_working_set_floor` (a floor of `n` equals `shrinking = false` exactly),
`test_fast_svm_reports_iteration_limit`, `test_fast_svm_polynomial_is_feasible`,
`test_fast_svm_rejects_bad_arguments`.

### Deviations from R

- Linear kernels: the dual need not be unique, so alphas and indices are not compared, only `w`.
- Polynomial kernels: no R case. They run through the same solver.

## `linear_svm`: LIBSVM with a linear kernel

`linear_svm` runs `fast_svm` with `FastKernel::Linear`, a kernel cache of at most 40 MB (e1071's `cachesize`),
shrinking on, and `max_iter = max_iterations * n` (saturating): `max_iterations` (default 1000) is a budget of
passes of n pair updates, and `tolerance` is LIBSVM's `eps` (default 1e-3, e1071's `tolerance`). `weights` is
`sum_k alpha_k y_k x_k`, `bias` LIBSVM's `-rho`; a decision value `> 0` is the larger label, as LIBSVM
(`svm_predict` follows the same rule; it used `>= 0`). `n_iterations` counts LIBSVM pair updates, and the
result carries the solver's convergence record and warnings (`ResultNotes`). R equivalent:
`e1071::svm(kernel = 'linear', type = 'C-classification', scale = FALSE)`.

Before ML-40 (Tier 4) `linear_svm` ran its own simplified SMO, which searched non-bound multipliers below a
hard-coded 1 and stopped after one sweep without a change: at C = 0.1 on `cls_lin` it returned
`w = (0.0259, 0.0385)` against e1071's `(1.1909, 0.7867)`, at C = 10 `w = (2.4035, 3.5791)` against
`(2.0528, 1.2152)`, both reported as converged.

| Test (`crates/p2a-core/src/ml/svm.rs`) | Compared with | Tolerance |
|---|---|---|
| `test_validate_linear_svm_libsvm_c01`, `test_validate_linear_svm_libsvm_c10` | e1071 `linear_c01_*`, `linear_c10_*`: `w`, bias, decision values `x w + b`, predictions; called at tolerance 1e-10 with a budget of 1,000,000 passes | `Tol::iterative(4e-7)`; predictions exact |
| `test_linear_svm_equals_fast_svm` | `fast_svm` with the same settings at C = 0.1, 1, 10: weights, bias, support vectors, iterations, predictions | exact |
| `test_linear_svm_iteration_budget` | one pass on a problem that needs more stops with `converged = false` and a convergence warning | property |

MCP: `test_validate_mcp_svm_linear_matches_e1071` (`crates/p2a-mcp/tests/tier3/wp57_ml_spatial.rs`) runs
`ml_svm` on `svm_sim` at cost 10 and tolerance 1e-10: the written prediction column equals `linear_c10_pred`.

## `kernel_svm`: not R-pinned

`kernel_svm` (library only; no MCP tool or CLI command) keeps a simplified SMO: it skips alpha steps below 1e-5
and is accurate only to its KKT tolerance. It checks its inputs as `linear_svm` does (C and the tolerance finite
and positive, `max_iter >= 1`, finite features) plus the kernel's (`gamma`, when given, finite and positive;
`degree >= 1` for the polynomial kernel; finite `coef0`); before ML-140 C = 0 returned an untrained model
labelled converged. The n x n kernel matrix is allocated fallibly, and the budget is checked per row of it and
per SMO sweep. The result keeps the resolved `gamma` and the training width (`n_features`):
`kernel_svm_predict` refuses data of another width and uses that `gamma` (before ML-141 it truncated wider or
narrower data silently and recomputed a defaulted `gamma` from them). Tests:
`test_kernel_svm_rejects_bad_arguments`, `test_kernel_svm_honours_cancellation_and_ceiling`,
`test_kernel_svm_predict_checks_width_and_keeps_gamma`.

## Class labels

`linear_svm` and `kernel_svm` keep the two class labels as `i32`: the target must hold two whole numbers in the
`i32` range ("SVM: class labels must be whole numbers ... (found 0.2 and 0.7); recode the target, for example
as 0/1"). Before ML-102 two fractional labels were truncated (0.2 and 0.7 both became 0), and `kernel_svm`'s
training accuracy compared the truncated predictions with the original values; it is now computed on the
internal -1/+1 labels. The class-distribution line of `linear_svm`'s summary counts the predictions of each
label ("45 predicted 0 (negative), 55 predicted 1 (positive)"); it counted predictions below zero as negative
before (ML-101). Tests: `test_svm_refuses_fractional_labels`, `test_svm_display_counts_predicted_labels`,
`test_kernel_svm_accuracy_uses_internal_labels`.

## Missing values

`fast_svm` and `linear_svm` refuse non-finite input. The MCP tool `ml_svm` drops rows with a missing
value (`NaPolicy::Omit`).

## Timing

Quiet integration re-time, 2026-11-16: release builds of v0.5.0 (`f80b0131`) and of the integrated branch,
`--no-default-features --features ci`; one core (`taskset -c 2`), `RAYON_NUM_THREADS=1` (one thread for polars and
BLAS too), cwd `/dev/shm`; per row five processes per side, interleaved with the order alternating (nine for rows
within 1.05-1.15x), each the median of its calls after a warm-up; cells are the median over the processes, the ratio
of the minima in brackets. 1-minute load average 0.7-2.7 on 12 cores, nothing else building. Peak RSS: one process per
side, the median of three (`/usr/bin/time`). Intel Core 7 150U.

`linear_svm` against the v0.5.0 SMO. "Benchmark" is `unified_benchmarks`' SVM row (cluster data, p = 5, label 1
when `i % 3 == 0`, else -1, C = 1, defaults); "overlapping" is two overlapping classes (p = 5, the label a noisy
linear rule) at C = 10 and default settings, where v0.5.0 stops short of the optimum, so its time is not a
like-for-like row.

| Data | n | v0.5.0 | head | head / v0.5.0 [min] |
|---|---|---|---|---|
| benchmark | 100 | 29.5 µs | 42.9 µs | 1.45 [1.00] |
| benchmark | 1,000 | 408 µs | 212 µs | 0.52 [0.50] |
| benchmark | 10,000 | 4.84 ms | 2.45 ms | 0.51 [0.50] |
| overlapping, C = 10 | 100 | 1.36 ms | 329 µs | 0.24 [0.23] |
| overlapping, C = 10 | 1,000 | 7.92 ms | 13.2 ms | 1.67 [1.67] |
| overlapping, C = 10 | 10,000 | 518.2 ms | 1.09 s | 2.10 [2.10] |

Peak RSS: benchmark at n = 10,000 3.8 MB against 5.0 MB; overlapping at n = 1,000 3.2 MB against 4.0 MB, at
n = 10,000 3.8 MB against 46.4 MB (the kernel cache of at most 40 MB, e1071's default). The memory increase on hard
problems was signed off with the choice of LIBSVM's solver. The overlapping time rows at n >= 1,000 compare an
exact solver with one that stopped early: n = 1,000 (1.67x) is signed off 2026-11-16 (1.70x); n = 10,000 runs
2.10x in the quiet re-time against the 1.71x signed (v0.5.0 was faster on the quiet machine, the head time
is the same) and has gone back to the user, as has the benchmark row at n = 100 (median 1.45x, minima 1.00x: most
calls of this tree take about 43 µs, the fastest 29 µs, as v0.5.0's).

Benchmark outputs on the paper's `cluster_n*.csv` data (`n_support_vectors`, bias; all converged):

| n | v0.5.0 | head | e1071 (`unified_r_20261110_173651.json`) |
|---|---|---|---|
| 100 | 4, 1.2445459858 | 4, 1.2444725584 | 4, 1.244473 |
| 1,000 | 2, 1.3118137256 | 2, 1.3118137486 | 2, 1.311814 |
| 10,000 | 2, 1.3444078174 | 2, 1.3443436295 | 2, 1.344344 |

The v0.5.0 column equals the paper run of the same commit; `a0f5cf14^` (before `fast_svm`'s rewrite) was not
rebuilt, v0.5.0 stands in for it.
