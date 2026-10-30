# Validation: Data layer (quality profile, binning, casts, missing values, SQL typing, charts)

## Method Overview

The data layer is where values enter p2a and are reshaped before any estimator sees them. This
document records the R conventions it follows, so that no value is silently turned into a null,
an empty string, a zero or a wrong category:

- **Data quality profile** (`generate_quality_profile`, MCP `data_quality_profile`,
  `suggest_cleaning`): a missing cell is a polars null or a NaN in a float column (R's `is.na`).
  `ColumnProfile.null_count` counts both, `nan_count` says how many are NaN, and completeness
  and the high-missing-rate issue use that count. Numeric statistics (min, max, mean, sd,
  quartiles, median, IQR outlier bounds, zero and negative counts) are taken over the present
  values; the quartiles are polars' linear quantile, R's `quantile(type = 7, na.rm = TRUE)`.
  Integer, unsigned and decimal columns get numeric statistics.
- **Binning** (`bin`, `bin_with`) follows `base::cut`: missing values and values outside the
  edges are NA (a null label), intervals are `(a, b]` (`right = TRUE`) or `[a, b)`, and
  `include_lowest` closes the outer interval at the lowest (highest, when left-closed) edge.
  Text columns are refused, as `cut()` refuses them.

## Reference Implementations

| Package | Language | Function | Version tested |
|---------|----------|----------|----------------|
| base, stats | R | `quantile(type = 7, na.rm = TRUE)`, `mean`, `sd`, `median`, `cut` | R 4.3.3 |

## Test Cases

### Data quality profile

Dataset `validation/datasets/generated/quality_na.csv` (40 rows: `id`, `x` normal with nulls in
rows 3, 17 and 29, `z` uniform with its last three cells missing), written by the case.

```r
set.seed(12)
x <- round(rnorm(40, 10, 3), 4); x[c(3, 17, 29)] <- NA
z <- c(round(runif(37, 0, 5), 4), NaN, NaN, NaN)
q <- quantile(x, c(0.25, 0.5, 0.75), type = 7, na.rm = TRUE)
sum(x < q[1] - 1.5 * (q[3] - q[1]) | x > q[3] + 1.5 * (q[3] - q[1]), na.rm = TRUE)
```

| Quantity | R |
|---|---|
| x q1, median, q3 | 7.8895999999999997, 9.1830999999999996, 11.0215 |
| x mean, sd | 9.5906459459459459, 2.667907832573702 |
| x min, max | 4.0071000000000003, 16.216100000000001 |
| x missing, IQR outliers | 3, 2 |
| z missing, median | 3, 2.2757000000000001 |
| completeness of (x, z) | 0.925 |

The old profile took the quartiles by index from a sort with the nulls first and its length
including them (q1 7.6613, median 9.1201, one outlier), and counted NaN cells as present.
The harness writes NaN as an empty field (read as null on both sides), so the NaN variant of
the case is built in Rust from the case dataset: z's three missing cells become NaN and the
same R values hold.

### Binning (cut)

Dataset `validation/datasets/generated/bin_cut.csv` (12 rows: `id`, `x` = -5, 0, 0.5, 1, 2.5,
3, 4.9, 5, 7, NA, 2, NA).

```r
cut(x, breaks = c(0, 1, 3, 5))                          # (0,1] (1,3] (3,5]
cut(x, breaks = c(0, 1, 3, 5), include.lowest = TRUE)   # [0,1] (1,3] (3,5]
cut(x, breaks = c(0, 1, 3, 5), right = FALSE)           # [0,1) [1,3) [3,5)
```

| Option | Bin index per row (0-based, -1 = NA) |
|---|---|
| right-closed (default) | -1 -1 0 0 1 1 2 2 -1 -1 1 -1 (5 NA: 2 missing, 3 outside) |
| `include.lowest = TRUE` | -1 0 0 0 1 1 2 2 -1 -1 1 -1 |
| `right = FALSE` | -1 0 0 1 1 2 2 -1 -1 -1 1 -1 |

The row rule is R's `.bincode`: with `k` bins and sorted edges, a present value `v` is in bin
`j - 1` for the `j` with `edges[j-1] < v <= edges[j]` (`[edges[j-1], edges[j])` when not
right-closed); `include.lowest` adds `v == edges[0]` to the first bin (`v == edges[k]` to the
last when left-closed); any other present value is NA (out of range) and a missing one NA
(missing). `bin_with` reports the edges, the rows per bin and both NA counts.

Equal-width edges are `cut(x, n)`'s: `seq.int(min, max, length.out = n + 1)` (interior edges
`min + i * (dx / n)`, `dx = max - min`) with the ends moved to `min - dx/1000` and
`max + dx/1000`; a constant column spans `min -/+ |min|/1000` (`-/+ 1/1000` around 0) with
interior edges `from + i * ((to - from) / n)`. Checked with R 4.3.3: `cut(x10, 7)` breaks
0.39130000000000004 ... 9.1086999999999989 and codes 1 0 4 2 3 0 6 4 5 2 for
`x10 <- c(2.3, 1.1, 5.7, 3.3, 4.8, 0.4, 9.1, 6.6, 7.2, 3.9)`; `cut(c(3, 3, 3), 2)` breaks
2.997 3 3.003; `cut(c(0, 0), 3)` codes 1 1; `cut(1:4, 2)` codes 0 0 1 1.

The quantile strategy is p2a's own: the edges are `-Inf`, `quantile(x, (1:(n-1))/n, type = 7,
na.rm = TRUE)` and `+Inf` (open outer bins, so it has no out-of-range rows). Tied interior
edges are an error (R's `cut` refuses duplicated breaks); an edge that interpolates from a
`-Inf` neighbour is `-Inf`, and a present `-Inf` (`+Inf`) that no interval holds goes to the
first (last) bin, so a column with `log(0)` values bins whatever their count.

## R reference

`validation/r/cases/data_quality.R`, cases `data_quality_profile_na` and `data_bin_cut`.

## Rust tests

- `p2a-core::data::quality::tests::test_validate_quality_profile_na`: every quantity above
  on the case dataset.
- `p2a-core::data::quality::tests::test_validate_quality_profile_nan_cells`: z's missing
  cells as NaN; `null_count` 3 of which `nan_count` 3, z's median and the completeness.
- `p2a-core::data::munging::features::tests::test_validate_bin_cut_right_closed`: the
  right-closed codes and the NA count through `bin`.
- `p2a-core::data::munging::features::tests::test_validate_bin_cut_options`: the three code
  vectors through `bin_with`, the missing and out-of-range counts and the summary text.

## Tolerance

Order statistics and moments: `Tol::new(1e-12, 0.0)` (deterministic; every quartile of the
case falls on an order statistic, mean and sd differ from R by a few ulp), tighter than the
README table. Counts: `Tol::exact()`. Binning: bin codes and counts `Tol::exact()`.

## Status

R-pinned: registry `validation/methods/data_layer.json`, ids `data_quality_profile` and
`data_bin`.

## Known Differences

- Infinite end edges are allowed in custom bins (as in R); NaN edges and edges that are not
  strictly increasing are refused. R sorts unsorted breaks (`cut(c(1, 4), c(5, 0, 2))` gives
  codes 0 1); p2a refuses them, because user labels follow the edges in the order given.
- The quantile strategy keeps infinite values in the outer bins where R's `cut` would refuse
  the repeated `-Inf` break (`cut(c(-Inf, 1, 2, 3), c(-Inf, -Inf, 1.5, 2.25, Inf))` is an error
  in R).
- `cut(x, 1)` is an error in R ("invalid number of intervals"); `EqualWidth(1)` gives one bin.

## References

- Hyndman, R. J. and Fan, Y. (1996). Sample quantiles in statistical packages. *The American
  Statistician* 50(4), 361-365 (R's type 7).
