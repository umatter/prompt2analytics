# Validation: Quantile regression (rq, summary.rq)

## Scope

`p2a_core::regression::quantreg` (`quantreg`, `run_quantreg`, `quantreg_multi`,
`quantreg_multi_with`, and the array API `rq_fit`, `rq_se`, `hall_sheather_bandwidth`):

| Option | Values | R |
|---|---|---|
| `QuantRegConfig::algorithm` | `Auto` (default: `br` for n <= 5000, `fn` above), `BarrodaleRoberts` (`br`), `FrischNewton` (`fn`), `Irls` | `rq(method = "br" \| "fn")`; IRLS has no R method |
| `QuantRegConfig::se` | `Nid` (default), `Iid`, `Ker`, `Boot` | `summary.rq(se = "nid" \| "iid" \| "ker" \| "boot")` |
| `bootstrap_samples`, `seed` | replications for `Boot` (0: 200), resampling seed (default 42) | `summary.rq(se = "boot", R = )` |

Every coefficient row reports the estimate, the SE, t = estimate / SE, the two-sided p-value on
t(n - k) and the 95% interval estimate -/+ qt(0.975, n - k) SE. The result also carries the
SE type and how it was computed (`se_method`, the bandwidths, the sparsity), the simplex's
nonunique flag, a convergence record and the warnings; `Display` ends with them.

## R code

`validation/r/cases/reg_quantreg_br.R` writes three cases (R 4.3.3, quantreg 6.1):

- `rq_br_se`: `rq(y ~ x, tau, method = "br")` on T1's `validation/datasets/generated/quantreg_hetero.csv`
  (100 rows, columns `y`, `x`) at tau 0.02, 0.25, 0.5 and 0.75, with `summary.rq(se = "iid" | "nid" | "ker",
  covariance = TRUE)`, the bandwidths, `rq.fit.fnb` on the dispersion-standardised outcome, y x 1e-6 and 1e6,
  the nonunique location models and the n = 5 subsample;
- `rq_br_mtcars`: `rq(mpg ~ wt + hp + qsec)`, `rq(mpg ~ wt + hp - 1)` and `rq(mpg ~ 1)` on T1's
  `validation/datasets/generated/mtcars.csv` (n = 32) with the same SE types;
- `rq_boot_xy`: `boot.rq(cbind(1, x), y, tau, R = 50, U = U)` (the xy-pair bootstrap of `summary.rq(se =
  "boot")`) on a committed resample index matrix, `validation/datasets/generated/rq_boot_indices.csv` (100 rows x 50
  integer columns `b01`-`b50`, 1-based row indices; sha256
  `5449244ce3a51ab04ab384afe658922a5cde7311d22fc0ff25249cf0d811ffa7`).

Rebuild with `Rscript validation/r/run.R --write rq_br_se rq_br_mtcars rq_boot_xy`; check with
`Rscript validation/r/run.R --check rq_br_se rq_br_mtcars rq_boot_xy`. R warns while the cases run
("Solution may be nonunique" from the location models and the tau 0.75 iid sparsity fit, "non-positive
fis"); the cases count those warnings. Some keys are R facts that no test pins: `fn_*_coef` and
`fn_*_nit` (rq.fit.fnb's iterates), `fn_r_*` and `scale_*_fn_r_*` (R's own `fn` with its absolute
duality gap), `loc_*_coef` (one vertex of a nonunique solution), `scale_*_se_*` (R's scale-dependent
iid and nid SEs) and `*_ci_rank` (rank-inversion intervals, not computed by p2a).

## Values

Decisive rows (R; the committed JSON is authoritative):

| Quantity | R |
|---|---|
| `rq_br_se` coefficients, tau 0.25 / 0.5 / 0.75 | (1.380407426551238, 2.7682582131912987) / (1.3487349841266367, 3.1699511987566726) / (1.7406414250496087, 3.4670761595702593) |
| tau 0.5 SE nid / iid / ker | (0.17987111706382683, 0.107953087584705) / (0.4807247768355065, 0.07953329682514744) / (0.9774375872023762, 0.19853639127827996) |
| Hall-Sheather bandwidth, n = 100, tau 0.25 / 0.5 / 0.75 | 0.14496914586992257 / 0.20931604694700326 / 0.14496914586992257 |
| tau 0.02: bandwidth before / after halving | 0.024281494906297038 / 0.012140747453148519 |
| Powell bandwidth, tau 0.5 | 3.3711555303829153 |
| `rq_br_mtcars` tau 0.25 nid SE (Intercept, wt, hp, qsec) | 7.030569777571573, 0.4940325260350767, 0.016953413406396757, 0.2964210734105201, with "1 non-positive fis" |
| Frisch-Newton standardisation s = mean \|y - median(y)\| | 8.611609324583537 |
| `rq_boot_xy` SE, tau 0.5 | 0.24570002521529088, 0.09922200972323754 |

Before this package p2a reported at tau 0.5 coefficients 1.348696693 / 3.169975618 (IRLS, REG-52),
SEs 0.05028920546 / 0.008320075224 (sqrt(n) too small, REG-26) and intervals with 1.96 (REG-64).

## Tests and tolerances

`validation/methods/reg_quantreg_br.json` (methods `quantreg_br`, `quantreg_summary_se`,
`quantreg_fn`, `quantreg_boot`) lists the R-pinned tests: `test_validate_rq_br_coefficients`,
`test_validate_rq_br_mtcars`, `test_validate_rq_br_nonunique_location`,
`test_validate_rq_scale_equivariance`, `test_validate_rq_median_unchanged_by_outlying_responses`,
`test_validate_rq_irls_scale_free_and_honest`, `test_validate_rq_summary_se_iid`,
`test_validate_rq_summary_se_nid`, `test_validate_rq_summary_se_ker`,
`test_validate_rq_bandwidth_halving`, `test_validate_rq_small_sample_se`,
`test_validate_rq_multi_hetero`, `test_validate_rq_fn_coefficients` and
`test_validate_rq_boot_xy_indices`. T1's `test_validate_quantreg_hetero`
(`regression_diagnostics.json`) pins the coefficients, objectives and nid/iid SEs on the same data.

Tolerances follow validation/README.md: n = 100 coefficients 1e-8, SEs, t, intervals and bandwidths
1e-6 relative, p-values 1e-6 relative or 1e-12 absolute; n = 32 coefficients 1e-6, the rest 1e-5;
n = 5 coefficients 1e-6, SEs 1e-5. Frisch-Newton is compared with the exact simplex solution at
`Tol::iterative(1e-6)` (an interior-point method stops at a duality gap, not at a vertex); it is
not R-pinned to rq.fit.fnb's iterates.

Properties checked without an R value: the nonunique location models (the solution lies between
the order statistics y(k) and y(k+1), k = tau n, with the objective of y(k));
`test_rq_se_scale_equivariance` and `test_rq_fn_scale_free` (y x 1e-6 and 1e6 give the scaled
coefficients and SEs, the same iterations and the same warnings); `test_rq_ci_uses_t_quantile`
(qt(0.975, 98) and qt(0.975, 28); p < 0.05 exactly when 0 is outside the interval);
`test_rq_auto_selects_by_n`; `test_rq_boot_seeded_reproducible`;
`test_rq_bootstrap_samples_requires_boot_se`; `test_rq_rejects_singular_design`,
`test_rq_rejects_invalid_parameters` and `test_rq_fn_rejects_extreme_tau`.

## Differences from R (documented)

1. **The default SE is nid at every n.** R's `summary.rq` uses rank inversion (`se = "rank"`)
   for n < 1001; p2a does not compute rank-inversion intervals and refuses `se = "rank"`.
   `se_method` names the type that was computed.
2. **The 95% interval** is `coef -/+ qt(0.975, n - p) se`; summary.rq prints none for
   iid/nid/ker.
3. **Frisch-Newton runs on the standardised outcome** `y / mean |y - median(y)|`, so its
   duality-gap rule (`tol`, quantreg's `eps = 1e-6`) is scale-free. R's absolute rule makes
   `rq(method = "fn")` stop after 3 iterations, 0.93% off, at y x 1e-6
   (`scale_m6_fn_r_tau50_coef`).
4. **summary.rq's fuzz is scale-free.** R's `eps = sqrt(.Machine$double.eps)` is absolute, so its
   iid and nid SEs move by 7.7% and 4.0% at y x 1e-6. p2a uses `sqrt(.Machine$double.eps) *
   mean |y - median(y)|` (1 when that is 0) in the iid zero-residual count and the nid density,
   and every SE is scale-equivariant. At ordinary scale this moves the nid SEs by at most 3.0e-7
   relative on `quantreg_hetero` (iid is unchanged), except at tau 0.02, where tau - h = 0.0079
   and the intercept's nid SE moves 1.2e-6; the tau 0.25 intercept's nid p-value moves 3.4e-6.
   Both are registry deviations (REG-26), with p2a's values equal to summary.rq's formula with the
   scaled eps.
5. **nid density** `f_i = 2h / (dyhat_i - eps)` is 0 when `dyhat_i - eps <= 0`; R's `pmax(0,
   ..)` gives Inf at exactly 0.
6. **A premature simplex end is an error** (`SingularMatrix`). R returns zero coefficients with
   "Premature end - possible conditioning problem in x". A singular design is refused before
   the fit, as rq does ("Singular design matrix").
7. **A pivot cap** of 50 (n + p) gives `ConvergenceFailure`; R has none. Frisch-Newton stops
   after 500 iterations, as quantreg, and reports non-convergence.
8. **Standard errors that cannot be computed** (iid with fewer than h + pz + 1 observations, a
   singular density-weighted design, a ker bandwidth of 0 because at least half the residuals are
   tied) leave the fit in place with NaN SEs, t, p and intervals and a warning; R's `rq` succeeds
   and `summary.rq` stops.
9. **Automatic method** (the default): the simplex for n <= 5000 and Frisch-Newton above,
   quantreg's own advice for large problems; R's `rq` default is the simplex at every n. On unique
   problems the two agree within `Tol::iterative(1e-6)`. The nonunique flag is reported by the
   simplex only. The nid auxiliary fits and the bootstrap replicates use the algorithm that ran
   (the simplex for an IRLS fit), except that an auxiliary fit at tau -/+ h outside (tol, 1 - tol),
   where Frisch-Newton cannot run, uses the simplex (R's nid would stop there); the iid sparsity
   fit is always the simplex.
10. **Bootstrap**: replicates are refitted with p2a's simplex (or Frisch-Newton), while
    `boot.rq.xy` uses `rq0` with tolerance 1e-4 (they agree exactly on the pinned indices);
    resampling is seeded ChaCha8 (stream = replicate index), so seeded p2a results do not equal
    R's `sample()`; resamples with a singular design are dropped with a warning.
11. **IRLS** is not an R method: an approximation whose objective is within about 1e-7 relative of
    the LP minimum here.
12. **Not implemented** from summary.rq: `se = "rank"`, `"BLB"`, `"extreme"`, `"conquer"`, other
    `bsmethod`s, `hs = FALSE` (Bofinger), weights, `rq(tau)` outside (0, 1) (the rq process).

## Performance

The simplex's cost grows about quadratically in n (each pivot updates the O(n p) tableau, and the
Barrodale-Roberts pass-through rescans the candidate rows), as in R; the default nid SEs add two
fits. That is why `Auto` switches to Frisch-Newton, which is linear in n (O(n p^2) per iteration),
above 5000 observations. The long loops (pivots, interior-point and IRLS iterations, bootstrap
replicates) are cancellable through resource checkpoints.

**Timing, like for like (T3-PERF3).** The wave-2A timing compared the old default, IRLS with
one sandwich standard error built on an Epanechnikov kernel density of the residuals at zero,
against the new default, `Auto` with `nid` SEs, which fits two extra quantiles. Release build,
one P core (`taskset -c 2`), `RAYON_NUM_THREADS=1`, tau = 0.5 with three regressors and an
intercept, the scratch timing binaries, three interleaved rounds on a machine loaded by other
builds; head / base against `fe13b889`'s default (base 3.2 ms at n = 1,000, 26.0 ms at
n = 10,000):

| Head fit | n = 1,000 | n = 10,000 |
|---|---|---|
| `Auto` (br up to 5,000, fn above), se = ker | 0.44 | 1.03 |
| `Auto`, se = iid | 0.43 | 1.04-1.09 |
| `Auto`, se = nid (the default) | 1.20 | 2.84 |
| br, se = ker | 0.44 | 2.31 |
| br, se = nid | 1.20 | 7.1 |
| fn, se = ker | 0.48 | 1.05 |
| fn, se = nid | 1.44 | 2.90 |

With the same kind of standard error (a kernel or difference-quotient sparsity estimate and
one fit), the default path is within 1.1x of the old fit (faster at n = 1,000). The rest is the
nid SE's cost, 2.75 times the ker fit at both sizes: it refits at tau ± h to estimate each
observation's density, which is what `summary.rq(se = "nid")` does. The simplex at
n = 10,000 is not on the default path. No code change.
