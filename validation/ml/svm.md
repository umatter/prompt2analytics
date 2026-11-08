# Support vector classification

Two solvers live in `crates/p2a-core/src/ml/`. `fast_svm` (`svm_fast.rs`) is LIBSVM's C-SVC solver
and is compared with `e1071::svm`. `linear_svm` and `kernel_svm` (`svm.rs`) are not pinned to R.

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

## `linear_svm` and `kernel_svm`: not R-pinned

`linear_svm` hard-codes the bound 1.0 in `select_second_alpha` (finding ML-40), so it is not optimal at
`C != 1`; `kernel_svm` skips alpha steps below 1e-5 and is accurate only to its KKT tolerance.

## Missing values

`fast_svm` and `linear_svm` refuse non-finite input. The MCP tool `ml_svm` drops rows with a missing
value (`NaPolicy::Omit`).
