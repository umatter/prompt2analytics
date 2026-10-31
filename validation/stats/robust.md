# Validation: Robust summaries and interpolation tools (fivenum, IQR, mad, ecdf, density, spline, approx)

## Method Overview

Seven MCP tools in `crates/p2a-mcp/src/tools/handlers/stats.rs` read one or two dataset columns
and call the core functions of `p2a_core::stats::robust` and `p2a_core::stats::spline`:

| Tool | Core function | R function |
|---|---|---|
| `stats_fivenum` | `fivenum` | `fivenum` (Tukey's hinges) |
| `stats_iqr` | `iqr` | `IQR(type = 7)` |
| `stats_mad` | `run_mad` | `mad(center = median or mean, constant = 1.4826)` |
| `stats_ecdf` | `run_ecdf` | `ecdf(x)(at)` |
| `stats_density` | `run_density` | `density` (only the number of observations is pinned) |
| `stats_spline` | `spline` | `spline(method = "natural")` |
| `stats_approx` | `approx` | `approx(method = "linear")` |

The tools read their columns through the shared column extractor
(`crate::tools::common::numeric_column` / `numeric_columns`), so integer columns are accepted.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `fivenum`, `IQR`, `mad`, `ecdf`, `approx`, `spline` | R 4.3.3 |

## R case: `stat_descriptive_na`

`validation/r/cases/stats_mcp.R`, case `stat_descriptive_na`, dataset
`validation/datasets/generated/descriptive_na.csv` (15 rows: `id, v, vi, s, xs, ys`). `v` is
missing in rows 2, 9 and 13 (1-based), `vi` is an Int64 column missing in rows 4 and 11, `s` is
text, `xs` is missing in row 5 and `ys` in row 10 (13 complete pairs, no tied `xs`).

```r
fivenum(d$v); IQR(d$v, na.rm = TRUE); mad(d$v, na.rm = TRUE)
mad(d$v, center = mean(d$v, na.rm = TRUE), na.rm = TRUE)
ecdf(d$v[!is.na(d$v)])(c(15, 20, 25))
approx(d$xs, d$ys, xout = c(1.5, 4, 6.5, 8))                       # drops the incomplete pairs
spline(d$xs, d$ys, method = "natural", xout = c(1.5, 4, 6.5, 8))
```

| Quantity | R value |
|---|---|
| `fivenum(v)` | 10.954, 18.241, 21.0075, 25.595999999999997, 30.132 |
| `IQR(v)` / `mad(v)` / `mad(v, center = mean)` | 6.390499999999996 / 4.776937199999999 / 4.776937199999999 |
| `mean(v)` (the centre used) | 21.259916666666665 |
| `ecdf(v)` at 15, 20, 25 | 0.08333333333333333, 0.4166666666666667, 0.75 |
| values of `v` used | 12 |
| `fivenum(vi)` | 68, 85, 92, 101, 124 |
| `IQR(vi)` / `mad(vi)` / values used | 16 / 13.343399999999999 / 13 |
| `approx` at 1.5, 4, 6.5, 8 | 0.8082820053715308, -0.7521932489451476, 0.3538291876750702, 0.9312373477672532 |
| natural `spline` at 1.5, 4, 6.5, 8 | 0.7478985384585353, -0.7661760653678952, 0.3201682303488883, 1.128381748164739 |
| complete (x, y) pairs | 13 |

## Missing values

| Tool | Policy |
|---|---|
| `stats_fivenum`, `stats_iqr`, `stats_mad`, `stats_ecdf`, `stats_density` | the column's missing values (null or NaN) are dropped, as `na.rm = TRUE`, and reported under `missing`; `n` (and `data_n` for the density) counts the values used |
| `stats_spline`, `stats_approx` | rows with a missing x or y are dropped together, as R's `regularize.values` drops them, and reported; `n_knots` counts the pairs used |

`±Inf` is refused by the extractor (`NonFiniteValues`), a text column with "Column 's' contains
non-numeric values", an absent column with "Column 'nope' not found in dataset. ...". Nothing
is added to the JSON when no value was dropped.

## Tests

- `tools::handlers::stats::tests::test_validate_mcp_stats_robust_summaries_na` (p2a-mcp: the
  five-number summary, IQR, both MADs and the mean centre, the ECDF and `n` on `v` and on the
  Int64 `vi`; the density's `data_n`; `missing` of every call; the text and absent columns)
- `tools::handlers::stats::tests::test_validate_mcp_stats_spline_approx_na` (p2a-mcp: `approx`
  and the natural `spline` at `xout`, `n_knots`, `missing`)

**Tolerance:** `validation/README.md`, n = 12 to 15 (< 100): order statistics, centres and ECDF
values 1e-6 (`Tol::coef`), IQR, MAD and interpolated values 1e-5 relative (`Tol::stat`), counts
exact.

**Status:** R-pinned with known deviations (registry `validation/methods/mcp_stats.json`, id
`mcp_stats_robust_and_interpolation`).

## Known Differences

- **STAT-36:** `stats_iqr` computes every quantile type other than 1 and 7 as type 7; only the
  default type 7 is pinned.
- **STAT-37:** the density's FFT grid is mislabelled when the number of grid points is not a
  power of two; the density values are not pinned.
- **STAT-65:** the default density bandwidth is not `bw.nrd0`, and the non-Gaussian kernels
  scale `bw` differently from R; the bandwidth is not pinned.
- **STAT-69:** `stats_spline`'s default method `fmm` is a clamped spline, not R's
  Forsythe-Malcolm-Moler end conditions; only `method = "natural"` is pinned.
- **STAT-87:** `stats_fivenum`'s `iqr` field is the spread of the hinges, not R's `IQR`; it is
  not pinned.
