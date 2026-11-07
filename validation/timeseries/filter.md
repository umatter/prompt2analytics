# Linear filter

`filter` in `crates/p2a-core/src/forecasting/tsutils.rs`.
R reference case: `ts_filter_ar_cpt` (`validation/r/cases/ts_utils.R`), series `x` of
`validation/datasets/generated/ts_utils.csv` (a 60-value random walk).

## filter

`filter(x, f, method, sides, init)` follows `stats::filter`.

| Rule | p2a | R |
|---|---|---|
| Convolution, `sides = 1` | y[i] = sum_j f[j] x[i - j] | the same |
| Convolution, `sides = 2` | y[i] = sum_j f[j] x[i + p/2 - j], so the window is x[i + p/2 - p + 1 ..= i + p/2] | C `cfilter`, `nshift = length(filter) %/% 2` |
| Outside the series | NaN | NA |
| Filter longer than the series (convolution) | `InvalidSpecification` | "'filter' is longer than time series" |
| NaN coefficient | `InvalidSpecification` | "missing values in 'filter'" |
| Recursive | y[i] = x[i] + sum_j f[j] y[i - 1 - j] | C `rfilter`, no length limit |

For an even length the window has one more value after position i than before it, as in R. The finding's
example on x = 1, 4, 2, 8, 5, 7, 3, 9, 6, 10: `filter(x, rep(.25, 4), sides = 2)` is
NA 3.75 4.75 5.5 5.75 6 6.25 7 NA NA.

**Convention kept as p2a's.** `init` holds the p values before `x[0]` oldest first; R's `init` is newest first. p2a
also pads or truncates `init` where R requires `length(init) == length(filter)`. The MCP tool never passes `init`.

**Tests.** `test_validate_filter_even_sides2`, `test_validate_filter_odd_sides2`,
`test_validate_filter_even_sides1`, `test_validate_filter_odd_sides1`, `test_validate_filter_recursive`
at `Tol::coef(60)`; `test_filter_even_two_sided_matches_r_cfilter`, `test_filter_longer_than_series_is_an_error` and
`test_filter_rejects_missing_coefficients` hold the example and the refusals.
