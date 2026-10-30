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

## Reference Implementations

| Package | Language | Function | Version tested |
|---------|----------|----------|----------------|
| stats | R | `quantile(type = 7, na.rm = TRUE)`, `mean`, `sd`, `median` | R 4.3.3 |

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

## R reference

`validation/r/cases/data_quality.R`, case `data_quality_profile_na`.

## Rust tests

- `p2a-core::data::quality::tests::test_validate_quality_profile_na`: every quantity above
  on the case dataset.
- `p2a-core::data::quality::tests::test_validate_quality_profile_nan_cells`: z's missing
  cells as NaN; `null_count` 3 of which `nan_count` 3, z's median and the completeness.

## Tolerance

Order statistics and moments: `Tol::new(1e-12, 0.0)` (deterministic; every quartile of the
case falls on an order statistic, mean and sd differ from R by a few ulp), tighter than the
README table. Counts: `Tol::exact()`.

## Status

R-pinned: registry `validation/methods/data_layer.json`, id `data_quality_profile`.

## Known Differences

None.

## References

- Hyndman, R. J. and Fan, Y. (1996). Sample quantiles in statistical packages. *The American
  Statistician* 50(4), 361-365 (R's type 7).
