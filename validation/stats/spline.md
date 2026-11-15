# Validation: Spline and Linear Interpolation (spline, approx)

## Method Overview

`spline` and `approx` (`crates/p2a-core/src/stats/spline.rs`) interpolate as `stats::spline` and
`stats::approx` do.

Without `xout`, the number of output points defaults as in R: `3 * length(x)` for `spline` and 50 for
`approx`, with `x` after ties are removed. `n = 1` is the lower end of the range and `n = 0` is no point
(`seq.int(.., length.out = n)`).

The MCP tool passes its own default of 200 points, a separate item (TOOL-25).

## Test Cases

Case `stat_ts_acf_spline_t3` in `validation/r/cases/stats_misc_t3.R`, dataset
`validation/datasets/generated/ts_acf_spline_t3.csv` (knots `kx`, `ky`).

| Quantity | R |
|---|---|
| `spline(kx, ky, method = "natural")`: n | 30 |
| first and last x | 0.3, 9.6 |
| first and last y | 0.35552, 0.7756729999999998 |
| same with `n = 1`: x, y | 0.3, 0.35552 |

**Rust tests:** `stats::spline::tests::test_validate_spline_t3_default_n`,
`stats::spline::tests::test_validate_spline_t3_n_one`. Tolerance: `Tol::coef`, n exact. Registry
`validation/methods/stats_misc_t3.json`, id `spline_t3`.

`stats::spline::tests::test_spline_passes_through_knots` checks the knots and the point count (not
R-pinned).

## Methods, ties and warnings (Tier 4)

The methods are R's: `fmm` (the default) fits the cubic through the first and last four knots at the
ends (Forsythe, Malcolm and Moler), `natural` has zero second derivatives at the ends and extrapolates
linearly, `periodic` wraps (when the first and last y differ, the last is set to the first, with R's
warning "spline: first and last y values differ - using y[1] for both"), `hyman` filters the FMM
slopes to a monotone spline (Hyman 1983) and refuses a non-monotone y, and `monoH.FC` (in
`splinefun`) is Fritsch-Carlson as R's `splinefunH0`. FMM and periodic splines extrapolate the end
cubics, as R. A cubic spline needs at least 4 distinct x (`InsufficientData` with the distinct count),
`approx` at least 2.

Tied x are averaged before fitting, as R's `regularize.values(ties = mean)`, with R's warning
"collapsing to unique 'x' values" in `SplineResult::warnings` / `ApproxResult::warnings`; the default
n counts the distinct x. `approx(method = "constant")` returns the knot's y at an exact knot for every
`f`, and `f = 0` / `f = 1` take only the left / right value.

Case `stat_spline_t4` in `validation/r/cases/stats_misc_t4.R`, dataset
`validation/datasets/generated/spline_t4.csv`.

| Test (`stats::spline::tests::`) | R |
|---|---|
| `test_validate_spline_fmm` | `spline(x, y, xout = xo)`, `spline(x, y, n = 11)` (`fmm_s`, `fmm_f_n11`, `fmm_f_out`) |
| `test_validate_spline_natural_outside` | natural spline inside and outside the knots (`natural_s`, `natural_f_out`) |
| `test_validate_spline_periodic` | periodic spline, equal and unequal end values (`periodic_p`, `periodic_u`) |
| `test_validate_spline_hyman` | hyman on increasing and decreasing y; refusal of a non-monotone y (`hyman_h`, `hyman_m`) |
| `test_validate_splinefun_monoh_fc` | `splinefun(method = "monoH.FC")` (`monofc_h`, `monofc_m`, `monofc_s`) |
| `test_validate_spline_ties` | natural and fmm with tied x; default n after ties (`ties_natural`, `ties_fmm`, `ties_default_n`) |
| `test_validate_approx_constant_at_knots` | constant approx at f = 0, 0.5, 1 |
| `test_validate_approx_ties` | linear and constant approx with ties; rule 2 |

Tolerance `Tol::coef(n)`; counts exact. Registry `validation/methods/stats_misc_t4.json`, ids
`spline_t4` and `approx_t4`.

The MCP tool `stats_spline` reaches the same core; it does not yet report the warnings.

## Timing against the previous implementation (Tier 4)

Release build, scratch harness at f80b0131 and with this change; `--no-default-features --features ci`;
one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; the two builds interleaved per row with
the order rotated per round, each process the median of its calls after a warm-up (calls calibrated to
about 0.8 s per process); cells are the median over five rounds (nine or ten for rows within 1.05-1.15x),
the ratio of the minima in brackets; 1-minute load average 1.0-3.0.

| Method | n | f80b0131 (µs) | this change (µs) | ratio [min] |
|---|---|---|---|---|
| spline, fmm, default n | 100 | 12.1 | 12.7 | 1.05 [1.05] |
| spline, fmm, default n | 1,000 | 143.6 | 146.6 | 1.02 [1.02] |
| spline, fmm, default n | 10,000 | 2,197.2 | 2,243.0 | 1.02 [1.02] |
| spline, natural, default n | 100 | 12.2 | 13.0 | 1.07 [1.07] |
| spline, natural, default n | 1,000 | 144.7 | 150.0 | 1.04 [1.04] |
| spline, natural, default n | 10,000 | 2,205.2 | 2,269.5 | 1.03 [1.02] |
| spline, monoH.FC, default n | 100 | 10.4 | 11.5 | 1.11 [1.10] |
| spline, monoH.FC, default n | 1,000 | 128.5 | 141.2 | 1.10 [1.10] |
| spline, monoH.FC, default n | 10,000 | 2,174.5 | 2,196.9 | 1.01 [1.01] |
| approx, linear | 100 | 7.8 | 8.7 | 1.11 [1.12] |
| approx, linear | 1,000 | 102.8 | 108.9 | 1.06 [1.05] |
| approx, linear | 10,000 | 1,511.4 | 1,575.5 | 1.04 [1.04] |
| approx, constant | 100 | 7.3 | 8.6 | 1.19 [1.20] |
| approx, constant | 1,000 | 94.6 | 103.8 | 1.10 [1.10] |
| approx, constant | 10,000 | 1,397.7 | 1,486.4 | 1.06 [1.06] |

Hyman splines are new (v0.5.0 had none): 14.3 / 162 / 2,647 µs at n = 100 / 1,000 / 10,000. The rows above 1.1x (approx at n = 100, about 1 µs, and
monoH.FC at n = 100) are R's tied-x averaging (`regularize.values`) and its monoFC slopes; they go to the user
(rule 5).
