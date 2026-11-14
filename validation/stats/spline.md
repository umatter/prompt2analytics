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

Release build, one core (`taskset`, `RAYON_NUM_THREADS=1`), median of five interleaved runs of a scratch harness at f80b0131 and with this change, on a shared machine (load average 20-60), so differences under about 10% are noise. Preliminary: to be repeated at integration.

| Method | n | f80b0131 (µs) | this change (µs) | ratio |
|---|---|---|---|---|
| spline, fmm, default n | 100 | 19.4 | 20.4 | 1.05 |
| spline, fmm, default n | 1,000 | 212.7 | 228.2 | 1.07 |
| spline, fmm, default n | 10,000 | 3,676.4 | 3,808.8 | 1.04 |
| spline, natural, default n | 100 | 20.0 | 21.0 | 1.05 |
| spline, natural, default n | 1,000 | 218.8 | 237.8 | 1.09 |
| spline, natural, default n | 10,000 | 3,837.4 | 3,894.4 | 1.01 |
