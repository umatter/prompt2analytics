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

## Known Differences

- **STAT-69, STAT-40:** the FMM, periodic and monotone splines are not pinned.
