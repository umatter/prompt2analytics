# Friedman's SuperSmoother Validation

## Method

`regression::supsmu::supsmu` is a port of R's `stats::supsmu` (Friedman's Fortran
`supsmu`/`smooth` in R's `ppr.f`): running lines with uniform weights in windows of
`2 ibw + 1` points, `ibw = int(0.5 span n + 0.5)` (at least 2), shifted at the ends or
wrapped around [0, 1] with `periodic = TRUE`. With `span = "cv"` (None) it smooths with the
tweeter, midrange and woofer spans (0.05, 0.2, 0.5, R's single-precision constants
widened to double), smooths each span's exact leave-one-out absolute residuals with the
midrange span, picks the best span at each point, moves it toward the woofer by `bass`
(Friedman's rule), smooths, clamps and interpolates the spans, and ends with a tweeter
smooth. Tied x enter the windows one by one and their fits are averaged afterwards; the
result has one value per distinct x (R's `!duplicated(xo)`).

Before this, `supsmu` was a different smoother (tricube windows of rounded sizes, ties merged
first, approximate CV residuals; REG-31) and ignored `periodic` (REG-55).

Inputs as R: `periodic` needs x in [0, 1]; `span` must be in [0, 1]; `bass` in [0, 10] (R
ignores values outside (0, 10]; p2a refuses them); weights non-negative. A periodic smooth
whose widest window exceeds n is an error (R reads past its array there).

## R reference

Cases in `validation/r/cases/regression_smoothers_t3.R`:

- `smoothers_gcv_periodic_loess`: x = sorted `runif(100)`, y = sin(2 pi x) + N(0, 0.3^2)
  (`set.seed(20261127)`):
  ```r
  supsmu(x, y, periodic = TRUE); supsmu(x, y, periodic = FALSE)
  supsmu(x, y, span = 0.3, periodic = TRUE); supsmu(x, y, span = 0.3)
  supsmu(x, y, bass = 5); supsmu(x, y, bass = 5, periodic = TRUE)
  ```
- `supsmu_sine`: the legacy `validate_regression_diag.R` DGP (`set.seed(42)`, x = sorted
  `runif(100, 0, 2 pi)`, y = sin(x) + N(0, 0.3^2)); supersedes `supsmu_test.csv`:
  ```r
  supsmu(x, y); supsmu(x, y, span = 0.3); supsmu(x, y, bass = 5); supsmu(x, y, wt = w); supsmu(round(x, 1), y)
  ```

## Values

| Key | R (first values, ..., last) |
|---|---|
| `supsmu_periodic_y` | 0.05406098939563001, 0.05089869608987187, 0.07043700555743775, ..., -0.0036871704595962396 |
| `supsmu_plain_y` | 0.3721017324147564, 0.37179504877741937, 0.3658049366798011, ..., -0.2448389761845025 |
| `supsmu_span03_periodic_y` | 0.020128144899512465, 0.018171832200386623, 0.05037921783266596, ..., -0.03187916957288198 |
| `supsmu_span03_plain_y` | 0.19730074531476421, 0.19807899986174776, 0.2132797846750213, ..., -0.36707986702824796 |
| `supsmu_bass5_y` | 0.3901069225946856, 0.3901178289723678, 0.390330851178126, ..., -0.2486110828717627 |
| `supsmu_bass5_periodic_y` | 0.06590949818723782, 0.06223518419001231, 0.0812102175071758, ..., 0.010015485926487058 |
| `supsmu_sine` `cv_y` | 0.08836614520859382, 0.0945472103290684, 0.10558401206586905, ... |
| `span03_y`, `bass5_y`, `weighted_y` | 0.23794559748436644, ...; 0.08976977516918519, ...; 0.07547843596009472, ... |
| `ties_x`, `ties_y` | 0, 0.2, 0.3, 0.5, ... (54 values); 0.11071908799385004, 0.24979831134634112, 0.312820578487983, ... |
| `reg_smoothers_na` `supsmu_y` | 0.15651909282843657, 0.26487134495896175, 0.50685449350390188, ... (56) |

## Tolerances

`validation/README.md` coefficient row: n = 100: 1e-8, n = 56: 1e-6. The port agrees with
R to about 1e-15.

## Tests

Registered (`supsmu` in `validation/methods/regression_smoothers_t3.json`):

- `test_validate_supsmu_periodic`: cv, span 0.3 and bass 5, periodic and not; periodic and
  plain fits differ.
- `test_validate_supsmu_sine`: the x grid (exact), cv, span 0.3, bass 5, weights and ties.

Also `test_validate_supsmu_na` (below) now pins `supsmu_y`. Property tests (not registered):
`test_supsmu_periodic_is_rotation_invariant` (a periodic smooth does not depend on where
[0, 1) is cut, 1e-10), `test_supsmu_rejects_invalid_inputs` (periodic x outside [0, 1],
span, bass, negative weights, a periodic window wider than n).

## Missing and Non-finite Values (REG-13)

`supsmu()` deletes the observations with a missing (NaN) or infinite x, y or weight before
smoothing, as R's `supsmu` does (`is.finite(x + y + wt)`, with the warning "observations
with NAs, NaNs and/or Infs deleted"), and reports them in `SupsmuResult::missing`
(`by_column` names `x`, `y` and `wt`). With fewer than 4 complete observations it is an
error. Sorting uses `f64::total_cmp`. Before, a NaN never left the running sums, so one
missing y made the whole smooth NaN. MCP `regression_supsmu` reads x, y and the weights
column jointly with the incomplete rows dropped (a missing weight drops its row instead of
becoming 1.0, and a text weights column is an error instead of being ignored); `p2a reg
supsmu` does the same for x and y. Both report the drop (`missing` in JSON, a `Note:` line
in the CLI text). With fewer than 4 complete rows they fail before `supsmu()` is reached,
with `InsufficientData` naming the method and the columns and counting the dataset's rows
("SuperSmoother of 'y' on 'x': 2 of 6 rows have no missing value").

R: case `reg_smoothers_na` of `validation/r/cases/regression_na.R`
([source](../r/cases/regression_na.R)): `supsmu(x, y)` on 60 points with 2 missing x and 2
missing y; R's unique sorted x after the deletion is `supsmu_x` (56 values, 0.013, 0.2166,
0.6713, ...; the complete x have no ties), and its smooth on the NA data equals the smooth
on the complete pairs.

Tests (registry entry `supsmu_missing` in `validation/methods/regression_na.json`); every
comparison is exact except the CLI text's grid, which is printed with 4 decimals:

- `test_validate_supsmu_na`: the x grid equals R's `supsmu_x` exactly; `n` is
  `n_complete`; the smooth equals R's `supsmu_y`; every smoothed value is finite and equals the smooth on the complete rows;
  a NaN weight and an infinite x are deleted jointly;
- `test_validate_mcp_regression_supsmu_na`: the first 20 grid points equal R's, the
  smoothed values are numbers, a missing weight drops its row, a text weights column is an
  error;
- `test_validate_cli_reg_supsmu_na`: the printed grid equals R's to its 4 decimals; the
  `-F json` `fitted_sample` equals the smooth on the complete pairs exactly (the numbers
  are read from the output text with `str::parse`, which is correctly rounded, as
  serde_json is too since the workspace enabled its `float_roundtrip` feature).

Property tests, not registered (no R value):
`test_mcp_regression_line_supsmu_too_few_complete_rows` and
`test_cli_reg_line_supsmu_too_few_complete_rows` check the too-few-rows error of supsmu and
line through MCP and the CLI.

### REG-31 (fixed)

The smooth on this case is now pinned: `supsmu_y` is compared at `Tol::coef(56)` (the old
smoother was up to 0.673 off). The `supsmu_missing` registry entry in
`validation/methods/regression_na.json` still lists REG-31 as a deviation; its removal is
left to the status integration.

## Deviations

None: the port agrees with R on every case.

## References

1. Friedman, J. H. (1984). A Variable Span Smoother. Technical Report No. 5, Laboratory for Computational Statistics, Stanford University.
2. Friedman, J. H., & Stuetzle, W. (1981). Projection Pursuit Regression. JASA, 76(376), 817-823.
3. R Core Team. supsmu() documentation.

## Timing (WP11-13)

Release build, same protocol as `loess.md`. Head/base at n = 50 / 200 / 1,000: automatic span 2.68 /
3.13 / 3.28 (three running-line passes and the CV smoothing of R's algorithm; pre-approved, WP11-5);
fixed span 0.1: 1.35 / 1.36 / 1.55.
Re-measured on the merged tree against `fe13b889` (one P core, three interleaved runs each,
criterion `supsmu`): automatic span 2.73 / 3.53 / 3.75 at n = 50 / 200 / 1,000, fixed span 0.1
1.13 / 1.33 / 1.39.

**Shared and reciprocal updates.** Two changes, in `smooth` and in the shared-window smoother:

- The three residual smooths of the cross-validation have the same span, x and weights, so the
  x part of Friedman's updating formulas (the window's weighted mean and variance of x and the
  update factors) is computed once for the three of them (`smooth_many`);
  `test_supsmu_shared_residual_smooths_are_bit_identical` holds it bit for bit to three
  separate `smooth` calls (noisy data with ties, zero and unequal weights, periodic and not,
  three bass values, two fixed spans).
- The Fortran updates a window mean as `m = (W_old m + w v) / W`, a division on the chain of
  the running mean at every point it enters or leaves. `smooth` uses the same update in
  increment form, `m += w (v - m) / W`, and the update factors multiply by 1 / W, which
  depends only on the weight sum and is computed off the chains. The rounding then differs from
  the Fortran's, but not its accuracy: the error of a step is relative to `v - m` instead of to
  `m`. `test_supsmu_reciprocal_updates_match_divisions` holds the smooths and leave-one-out
  residuals to the division form within 1e-10 of the response's scale for x in [0, 1]; for x
  shifted by 1000, where both forms lose digits to cancellation, it compares both with the
  unshifted smooth, and the new form's total error is 0.92 times the division form's (1.03e-9
  against 1.11e-9; at most 1.5 times is asserted). Against the R pins the largest relative
  difference is 6.6e-12 (weighted and periodic CV smooths), far inside `Tol::coef`.
  Multiplying by 1 / W in the Fortran's form `(W_old m + w v) * (1 / W)` was tried first and
  rejected: with constant W the error of 1 / W has one sign at every step, and with x shifted
  by 1000 the error grew to up to 15 times that of the division form.

Release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), criterion `supsmu`, three
interleaved rounds against `fe13b889`; new / base as the ratio of medians.

| Variant | n = 50 | n = 200 | n = 1,000 |
|---|---|---|---|
| automatic span | 1.92 | 2.31 | 2.45 |
| fixed span 0.1 | 0.96 | 1.13 | 1.18 |

The automatic span's 1.9-2.5x (three running-line passes and the CV smoothing of R's
algorithm) is accepted; the target was at most 2.5.

**Interior run and mean factor (T3-PERF5).** The fixed span's cost per point was instructions,
not divisions (about 280 per point, three per cycle): 1-based index arithmetic, the periodic
wrap tests and bounds checks on every read of the leaving and entering point.

- The window's state and Friedman's updates are one `Running` type (add, remove, fit), used by
  the indexed loop and by an interior run: without leave-one-out residuals and without
  wrapping, the windows of j = ibw + 2 ..= n − ibw take their points from slices. The
  arithmetic and its order are unchanged; `test_supsmu_interior_run_matches_indexed_loop`
  holds the run bit for bit to the indexed loop (noisy data with ties, zero and unequal
  weights, spans whose window covers all the data). The residual and final smooths of the
  automatic span take the same path.
- A mean is updated as `m += (v - m) * (w / W)` with the factor formed first (it was
  `w (v - m) / W` evaluated left to right), so a mean's chain is one subtraction,
  multiplication and addition. The rounding moves: the largest relative difference against
  the R pins is now 6.5e-12 (6.6e-12 before), and
  `test_supsmu_reciprocal_updates_match_divisions` passes unchanged.

The machine was shared and loaded (load average 15-70; criterion's wall-clock rounds
scattered by up to 2x), so the lane measured user-space CPU cycles per call from the hardware
counter (`perf_event_open`; median of 11 batches, one P core, three interleaved rounds) on the
criterion bench's data, against `fe13b889`:

| Variant | n = 50 | n = 200 | n = 1,000 |
|---|---|---|---|
| fixed span 0.1, before T3-PERF5 (criterion, above) | 0.96 | 1.13 | 1.18 |
| fixed span 0.1, T3-PERF5 (cycles) | 0.80 | 0.92-0.94 | 0.88 |

The fixed span is within 1.1x of base.
