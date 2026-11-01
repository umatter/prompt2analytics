# Validation: Data layer (quality profile, binning, casts, missing values, SQL typing, charts)

## Method Overview

The data layer is where values enter p2a and are reshaped before any estimator sees them. This
document records the R conventions it follows, so that no value is silently turned into a null,
an empty string, a zero or a wrong category. A value is **missing** when it is a polars null or,
in a float column, NaN (R's `is.na`).

- **Data quality profile** (`generate_quality_profile`, MCP `data_quality_profile`,
  `suggest_cleaning`): `ColumnProfile.null_count` counts null and NaN cells, `nan_count` says
  how many of them are NaN, and completeness and the high-missing-rate issue use that count.
  Numeric statistics (min, max, mean, sd, quartiles, median, IQR outlier bounds, zero and
  negative counts) are taken over the present values; the quartiles are polars' linear
  quantile, R's `quantile(type = 7, na.rm = TRUE)`. Integer, unsigned and decimal columns get
  numeric statistics. The fill value `suggest_cleaning` proposes is that median.
- **Binning** (`bin`, `bin_with`, MCP `munge_bin`, CLI `munge bin`) follows `base::cut`:
  missing values and values outside the edges are NA (a null label), intervals are `(a, b]`
  (`right = TRUE`) or `[a, b)`, and `include_lowest` closes the outer interval at the lowest
  (highest, when left-closed) edge. Text columns are refused, as `cut()` refuses them.
- **Casting** (`cast`, `cast_with`, MCP `munge_cast`, CLI `munge cast`) converts text with R's
  coercion rules and reports how many present values became missing; `strict` refuses
  instead. `mutate` arithmetic and functions, `clip` and `bin` refuse text operands (R errors)
  instead of casting them to nulls; `clip` keeps NaN.
- **Missing values in `fill_na`**: NaN cells are filled like nulls (as `tidyr::replace_na`
  does); mean and median are those of the observed values, and a column without an observed
  value is an error instead of a fill with 0.
- **SQL result typing** (`query_sqlite`, `query_duckdb`, `query_file_with_duckdb` and the
  `db_*` MCP tools): a column's type comes from all its values, and every value keeps its type
  or is refused with an error; `QueryResult.warnings` names the columns loaded as text or
  approximated.
- **Interactive charts** (`scatter_interactive`, `histogram_interactive`): a missing group
  label is the group "NA" (ggplot2's convention); plotlars panicked on it.

## Reference Implementations

| Package | Language | Function | Version tested |
|---------|----------|----------|----------------|
| base, stats | R | `quantile(type = 7, na.rm = TRUE)`, `mean`, `sd`, `median`, `cut`, `as.integer`, `as.numeric`, `as.logical`, `pmin`/`pmax` | R 4.3.3 |
| tidyr | R | `replace_na` (fills NaN, as `is.na(NaN)` is TRUE) | behaviour quoted, not run |
| ggplot2 | R | missing discrete values shown as "NA" | convention quoted, not run |

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
including them (q1 7.6613, median 9.1201, one outlier), and counted NaN cells as present. The
harness writes NaN as an empty field (read as null on both sides), so the NaN variant of the
case is built in Rust from the case dataset: z's three missing cells become NaN and the same
R values hold.

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
first (last) bin, so a column with a few `log(0)` values bins. When the `-Inf` (`+Inf`) values
are frequent enough that two interior edges are `-Inf` (`+Inf`), those edges are tied and the
call is refused like any tied edges (use fewer bins or custom edges).

### Casting text to numbers

```r
v <- c("1", "1,234", "N/A", "3.7", "42", " 7", "$5", "", "-0.5e2")
as.integer(v)   # 1 NA NA 3 42 7 NA NA -50   Warning: NAs introduced by coercion ("" is NA silently)
as.numeric(v)   # 1 NA NA 3.7 42 7 NA NA -50
as.logical(c("TRUE", "true", "T", "True", "FALSE", "false", "F", "False", "yes", "1"))  # T T T T F F F F NA NA
as.integer(c(3.7, -3.7, NaN, Inf))               # 3 -3 NA NA (warning for Inf only)
pmin(pmax(c(NaN, -1, 5, NA, 2), 0), 3)           # NaN 0 3 NA 2
c("$5", "10") * 2                                # Error: non-numeric argument to binary operator
c(TRUE, FALSE, NA) + 1                           # 2 1 NA
```

`cast_with` trims text; `""` and `"NA"` are missing without being counted (R is silent about
them); integer targets take an integer literal or a number truncated toward zero within the
target's range; float targets parse a number (`inf` and `nan` included); the boolean target
takes `TRUE`/`true`/`T`/`True` and `FALSE`/`false`/`F`/`False`. Other sources are cast by
polars (numbers truncate toward zero, a non-finite or out-of-range number becomes missing), and
a present value that comes out missing counts as failed. The report names the count and up to
three distinct failed inputs, e.g. "column 'v': 2 values could not be converted to int and are
now missing (first: '1,234', 'x')"; `strict` makes it an error.

### Missing values in fill_na

```r
mean(c(NA_real_, NA_real_), na.rm = TRUE); median(c(NA_real_, NA_real_), na.rm = TRUE)   # NaN NA
x <- c(1, NA, NaN, 4, 10); c(mean(x, na.rm = TRUE), median(x, na.rm = TRUE))           # 5 4
is.na(NaN)                                                                             # TRUE
```

### SQL result typing (SQLite, DuckDB)

SQLite stores a type per value (a `DECIMAL(10,2)` column holds 10.00 as INTEGER and 12.50 as
REAL; sqlite3 3.45.1). The column type comes from all values:

| SQLite values | polars column |
|---|---|
| INTEGER only | Int64 |
| INTEGER and REAL | Float64 (warning when an integer beyond 2^53 is approximated) |
| any TEXT | String, numbers written as text (warning when numbers are mixed in) |
| BLOB only | Binary |
| BLOB with anything else | error naming the column (`CAST(col AS TEXT)`) |
| NULL only | String nulls |

DuckDB values are mapped by type (duckdb-rs 1.10506.1):

| DuckDB type | polars column |
|---|---|
| BOOLEAN | Boolean |
| TINYINT .. BIGINT | Int64 |
| UTINYINT, USMALLINT, UINTEGER, UBIGINT | UInt8, UInt16, UInt32, UInt64 |
| FLOAT, DOUBLE | Float64 |
| DECIMAL (every decimal literal such as 1.25) | Float64, from the exact decimal text |
| HUGEINT, UHUGEINT (every SUM over integers) | Float64 when every value is within 2^53, else String with a warning |
| DATE | Date |
| TIMESTAMP | Datetime(us) |
| TIME | Time |
| VARCHAR, ENUM | String |
| BLOB | Binary |
| LIST, STRUCT, ARRAY, MAP, UNION, INTERVAL, GEOMETRY | error naming the column and suggesting `CAST(col AS VARCHAR)` |

Expected encodings: `DATE '2024-03-01'` is day 19783, `TIMESTAMP '2024-03-01 12:34:56.789'` is
1709296496789000 us, `TIME '01:02:03.5'` is 3723500000000 ns.

### Interactive charts

A null in the group column of `scatter_interactive` or `histogram_interactive` (text or
numeric) is the group "NA"; the group column reaches plotlars as text.

## R reference

`validation/r/cases/data_quality.R`, cases `data_quality_profile_na` and `data_bin_cut`. The
cast, fill and SQL rules above quote the R (and sqlite3, DuckDB) outputs they follow; they are
pinned by property tests, not by a case.

## Rust tests

R-pinned (registry `validation/methods/data_layer.json`):

- `p2a-core::data::quality::tests::test_validate_quality_profile_na`: every quantity of the
  quality case.
- `p2a-core::data::quality::tests::test_validate_quality_profile_nan_cells`: z's missing cells
  as NaN; `null_count` 3, of which `nan_count` 3, z's median and the completeness.
- `p2a-core::data::munging::features::tests::test_validate_bin_cut_right_closed`: the
  right-closed codes and the NA count through `bin`.
- `p2a-core::data::munging::features::tests::test_validate_bin_cut_options`: the three code
  vectors through `bin_with`, the missing and out-of-range counts and the summary text.
- `p2a-mcp::tier2::wp04_data::test_validate_mcp_munge_bin_cut`: the three code vectors through
  `munge_bin` (`right`, `include_lowest`) and the NA line of the reply.
- `p2a-cli::tier2::wp04_data::test_validate_cli_munge_bin_cut`: the `Counts:` line of
  `p2a munge bin --method custom --edges 0,1,3,5` with no flag, `--include-lowest` and
  `--left-closed`.

Property and fixture tests (not registered: they load no case):

- quality profile: `test_quality_profile_counts_nan_as_missing`,
  `test_na_quality_profile_fixtures`; suggestions:
  `test_fill_suggestion_uses_the_median_of_present_values`,
  `test_nan_cells_get_a_missing_value_suggestion` (`data::suggestion::tests`);
- binning: `test_bin_missing_nan_and_out_of_range_values_are_na`,
  `test_bin_equal_width_follows_cut_n`, `test_na_bin_fixtures` and T0-WP20's tests adapted to
  the NA rule (`data::munging::features::tests`);
- casts and clip: `test_cast_text_to_number_follows_r_coercion`,
  `test_cast_reports_values_that_fail_to_convert`, `test_cast_text_to_bool_and_float_to_int`,
  `test_cast_columns_with_reports_each_column`, `test_clip_keeps_nan_and_refuses_text`
  (`data::munging::clean::tests`); mutate and filter:
  `test_mutate_refuses_text_operands`, `test_mutate_accepts_integer_and_boolean_operands`,
  `test_filter_float32_column` (`data::munging::transform::tests`);
- fill: `test_fill_na_mean_of_all_missing_column_is_an_error`,
  `test_fill_na_uses_present_values_and_fills_nan`, `test_na_fill_mean_fixtures`
  (`data::munging::clean::tests`);
- SQL (feature `database`, run locally; CI's `ci` feature set has no duckdb-sys):
  `test_sqlite_integer_and_real_column_is_float_with_every_value`,
  `test_sqlite_numbers_mixed_with_text_load_as_text_with_a_warning`,
  `test_sqlite_blobs_and_large_integers`, `test_duckdb_values_keep_their_types`,
  `test_duckdb_hugeint_sums_and_out_of_range_hugeints`,
  `test_duckdb_nested_and_interval_columns_are_refused_with_a_cast_hint`,
  `test_duckdb_file_query_keeps_dates_and_sums`,
  `duckdb_container_values_are_refused_unmaterialised` (`data::database::tests`), and the
  handler tests `duckdb_query_keeps_decimal_date_hugeint_and_unsigned_types`,
  `duckdb_query_refuses_list_columns_with_a_cast_hint` (`p2a-mcp`, `tools::handlers::database`);
- charts: `test_interactive_charts_label_a_missing_group_na`
  (`visualization::interactive::tests`);
- MCP and CLI paths (`tests/tier2/wp04_data.rs` of `p2a-mcp` and `p2a-cli`):
  `test_mcp_data_quality_profile_counts_nan`, `test_mcp_munge_fill_na_all_missing_is_an_error`,
  `test_mcp_munge_cast_reports_failed_values`, `test_mcp_munge_mutate_refuses_text`,
  `test_mcp_viz_interactive_with_missing_group`, `test_cli_munge_fill_na_all_missing_is_an_error`,
  `test_cli_munge_cast_reports_failed_values`, `test_cli_munge_mutate_refuses_text`,
  `test_cli_viz_interactive_with_missing_group`.

## Tolerance

Quality profile: order statistics and moments `Tol::new(1e-12, 0.0)` (deterministic; every
quartile of the case falls on an order statistic, mean and sd differ from R by a few ulp),
tighter than the README table; counts `Tol::exact()`. Binning: bin codes and counts
`Tol::exact()`.

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
- `cast` reports text that fails to parse even where R is silent (`as.logical("yes")` is NA
  without a warning); R's warning set is a subset of p2a's reports. R also parses hexadecimal
  text (`as.numeric("0x1A")` is 26), which p2a counts as failed.
- Carry-over (not owned by a Tier 2 package): `verification.rs` `preview_fill_na`
  still counts only nulls, so a preview can report no missing values for a column whose NaN
  cells `fill_na` fills.

## References

- Hyndman, R. J. and Fan, Y. (1996). Sample quantiles in statistical packages. *The American
  Statistician* 50(4), 361-365 (R's type 7).
- R Core Team. `?cut`, `?quantile`, `?as.integer`, `?NA` (R 4.3.3 documentation).
