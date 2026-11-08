# Munging and cleaning core

Value-level pins of the munging layer (`crates/p2a-core/src/data/munging/`) and the rules of the
cleaning verification. R code: `validation/r/cases/data_munging_t3.R`, one case per section. Registry:
`validation/methods/data_munging_t3.json`. R 4.3.3, dplyr 1.2.1, tidyr 1.3.2, stringr 1.6.0.

Tolerances: exact for codes, counts, integers and copied values; 1e-14 relative for a double
subtraction or ratio (the same IEEE operation on the same doubles as R); `Tol::coef(n)` and
`Tol::stat(n)` (`validation/README.md`) for the mean and sd of `describe`, whose measured agreement is
about 1e-15.

`cut` with custom edges and options and `fill_na` with mean and median are documented in
[data_layer.md](data_layer.md).

## Lag, lead, diff and pct_change

Positional, as `dplyr::lag` and `dplyr::lead` without `order_by`, within `group_by` groups in data
order; a time-aware lag (`plm::lag`, gaps) is DATA-39 (Tier 4). An integer column stays integer in `lag`
and `diff`; NA is a missing value, not NaN.

- R: `d %>% group_by(g) %>% mutate(lag2 = lag(x, 2), lead1 = lead(y, 1))`, `x - lag(x, k)`,
  `(y - lag(y)) / lag(y)`. Case `data_munging_lag`, on `munging_long.csv` (Int64 with NA, interleaved groups).
- Decisive values: `lag2_x_by_g = [NA, NA, NA, NA, 3, 5, NA, NA, 8, NA, 4, 2]`,
  `diff2_x = [NA, NA, NA, 5, NA, -1, 2, NA, 2, NA, 3, 9]`.
- Test: `test_validate_lag_dplyr`.

## Binning (cut) on integer data

`EqualWidth(n)` is `cut(x, n)` (R's range widened by 0.1% at both ends); `Quantile(n)` is `cut` on type-7
edges with open outer bins.

- R: `cut(x, 4)`; `cut(x, c(-Inf, quantile(x, c(1, 2) / 3, type = 7, na.rm = TRUE), Inf))`. Case
  `data_munging_cut`, on `munging_cut.csv`.
- Decisive values: `ew_breaks = [2.973, 9.75, 16.5, 23.25, 30.027]`, `q_edges = [9, 18]`, `n_na = 2`.
- Test: `test_validate_cut`.

## One-hot (model.matrix)

An integer factor and the full dummy set, as `model.matrix(~ 0 + factor(g))`. Not pinned: the baseline
of `drop_first`, an arbitrary level where R's treatment contrast drops the first sorted level (DATA-40,
Tier 4), and the names of text levels, which are quoted, and the dummy for NA (DATA-41, Tier 4). Both
are registry deviations.

- Case `data_munging_model_matrix`, on `munging_factor.csv`. Decisive value: `level_counts = [3, 4, 3]`.
- Test: `test_validate_model_matrix`.

## Melt (pivot_longer)

As `tidyr::pivot_longer`: row-major long format, missing values stay missing, and the value column has
the polars supertype of the value columns (integer + double gives double; integer + integer stays
integer; u32, f32 and bool keep their type on their own).

Text melted with non-text columns is refused, as tidyr refuses it ("Can't combine `a` <integer> and `s`
<character>."); a date, datetime, duration or time column melted with another kind of column is refused
too (vctrs refuses a date with anything but a datetime, and the supertype would give day counts, or read a
duration as a date). Dates combine with datetimes, and durations of different units with each other, at
the polars supertype.

- R: `pivot_longer(d, c(a, b, c), names_to = "var", values_to = "val")`. Case
  `data_munging_pivot_longer`, on `munging_wide.csv`.
- Decisive values: `long_val = [1, 0.5, 5, NA, 1.75, 6, 3, NA, 7, 4, 2.25, 8]`,
  `long_int_val = [1, 5, NA, 6, 3, 7, 4, 8]`.
- Tests: `test_validate_melt_pivot_longer`, `test_melt_keeps_dtypes_and_nulls`,
  `test_melt_refuses_text_with_numbers`.

## Fill strategies

`FillStrategy::parse` reads the keywords, case- and separator-insensitive: `mean`, `median`, `mode`
(`most_frequent`), `forward` (`ffill`, `locf`, `forward_fill`), `backward` (`bfill`, `nocb`,
`backward_fill`), `zero`, `constant` (`value`). An unknown keyword is an error; a constant keeps its case.

- **Constant**: parsed in the column's type, so the type is kept: every integer width with a range
  check, Float32/64, Boolean (as `as.logical`), Date (`YYYY-MM-DD`), String verbatim, Categorical/Enum
  labels. Any other type is an error.
- **Zero**: 0, FALSE or "". Date, categorical and other types have no zero (`tidyr::replace_na(<Date>,
  0)` errors too).
- **Mean, median**: see [data_layer.md](data_layer.md).
- **Mode**: `names(which.max(table(x)))`. Ties go to the first level in R's sorted order, not to the first
  occurrence (`k` ties 3, first in the data, and 1: R gives 1). All null is an error.
- **With no columns named**, a column the strategy or the constant does not fit is skipped and listed in
  `FillOutcome::skipped`. A named column gets the error.

Case `data_munging_fill_mode`, on `munging_fill.csv`. Decisive values: `k_mode = 1`, `f_mode = 2.5`,
`s_mode = "a"`. Tests: `test_validate_fill_mode`, `test_fill_constant_uint32_and_date`,
`test_fill_zero_and_constant_every_dtype`.

## Replace and missing values

`replace_with(.., new_value: Option<&str>, ..)`: `None` and, by default, the tokens `""` (after trim),
`"NA"` and `"null"` (any ASCII case) write a missing value, as `dplyr::na_if`. Old and new values are
parsed in the column's type, as constants are above; Float32 cells compare in f32. Text and category
labels compare exactly, so `"NA"` as an old value matches the text `NA`. Other column types are an
error. A missing value is never matched: that is `fill_na`. The float match keeps its absolute-epsilon
rule (DATA-69, Tier 4).

Case `data_munging_na_if`, on `munging_fill.csv`. Decisive value:
`k_na_if_3 = [NA, NA, 1, NA, 1, 2, NA, 5, 4, 6]`. Tests: `test_validate_replace_na_if`,
`test_replace_with_none_gives_null`, `test_replace_typed_every_dtype`.

## Value counts

`value_counts_with(.., normalize)`: counts, or proportions with every row in the denominator (the
missing values are a group of their own), as `dplyr::count(d, g, sort = TRUE)` and `n / sum(n)`. Tied
counts follow the order in which their values first appear in the column (deviation DATA-65:
`dplyr::count(sort = TRUE)` orders them by value; sorting every group by value cost 1.2-3x the count
itself on columns where most values are distinct). The order is the same on every call.

Case `data_munging_value_counts`, on `munging_counts.csv`. Decisive values: `vc_n = [4, 3, 2, 1]`,
`vc_proportion = [0.4, 0.3, 0.2, 0.1]`. Tests: `test_validate_value_counts`,
`test_value_counts_normalize`.

## Describe

One row of `<col>_count`, `_mean`, `_std`, `_min`, `_max` over the non-missing values.

Case `data_munging_describe`, on `munging_fill.csv`. Decisive values: `k`: count 8, mean 3.125, sd
1.807721533549109; `f`: count 8, mean 2.296875, sd 1.253455046490072. Test: `test_validate_describe`.

## Join column names

The right-hand columns are named by `dplyr:::add_suffixes`: a name that is taken gets the suffix, again
until it is unique, and earlier names win (`value_right_right`). The right names are checked against the
left columns polars keeps: all of them, except that a right join drops the left keys. The right keys
take part, as in `dplyr:::join_cols`, although inner and left joins then drop them. Where dplyr
silently drops a column under an empty suffix, an empty suffix that leaves a name on both sides is an error.

p2a's full join keeps both key columns (`k`, `k_right`; DATA-43, Tier 4), so the pin uses
`full_join(keep = TRUE)` and is a registry deviation.

Case `data_munging_join_suffix`, on `munging_join_frames.csv`. Decisive names: `left_join(AB, C)` gives `k,
value, value_right, value_right_right`; `cross_join(AB, C)` gives `k, value, value_right, k_right,
value_right_right`. Tests: `test_validate_join_suffix_dplyr`, `test_join_suffix_collision`.

## Filter

A row whose condition is NA is dropped, as `dplyr::filter`.

Case `data_munging_filter`, on `munging_long.csv`. Decisive values: `ids_x_ge_5 = [1, 4, 6, 9, 11, 12]`,
`ids_y_lt_9 = [2, 5, 6, 7, 10, 11]`. Test: `test_validate_filter_integer_with_nulls`.

## Verification of cleaning steps

A type change is a column-level change: `VerificationReport.dtype_changes`, a warning and a `Type
changes:` block. Cells of one type compare by value; when the type changed, cells compare as doubles if
both sides are numeric or boolean (`1L == 1.0` is TRUE), as text otherwise; a pair that cannot be cast is
skipped with a warning. Tests: `test_verify_cleaning_int_to_float_fill`,
`test_verify_cleaning_cast_text_to_int`, `test_apply_operation_dtype_change`.

The previews run the operation: `preview_fill_na` shows each sampled cell's real new value and reports an
unknown strategy as a warning; `preview_replace` counts typed matches in every column type. Tests:
`test_preview_fill_na_shows_real_fill_values`, `test_preview_replace_counts_typed_matches`.

## Wall-clock

`release`, `ci` features, against `fe13b889`, five interleaved rounds (order alternating) on a quiet
machine, 2026-11-08. "One core": `taskset` on one P core with `RAYON_NUM_THREADS=1` and
`POLARS_MAX_THREADS=1`; "pool": the default thread pool, no `taskset`. Each cell is the median of its
round medians; "ratio" is the median of the per-round ratios (this tree over the base), "min ratio" the
ratio of the minima. Load average 0.9-3.4 (the pool blocks of the run itself raise it). Data (seeded
ChaCha8): `lowcard` 10 integer levels; `highcard` uniform on 0..n, so most counts tie; `highcard_text`
the same values as `id%08d`; fill frames 10% missing.

| Row | n | base | this tree | ratio | min ratio |
|---|---:|---:|---:|---:|---:|
| `value_counts` low cardinality, one core | 100 | 13.38 µs | 4.56 µs | 0.34 | 0.32 |
| | 10,000 | 54.47 µs | 39.89 µs | 0.73 | 0.73 |
| | 1,000,000 | 4.02 ms | 3.55 ms | 0.88 | 0.90 |
| `value_counts` high-cardinality integers, one core | 100 | 14.21 µs | 3.21 µs | 0.23 | 0.22 |
| | 1,000 | 37.30 µs | 23.87 µs | 0.64 | 0.63 |
| | 10,000 | 345.8 µs | 321.4 µs | 0.93 | 0.94 |
| | 100,000 | 4.30 ms | 4.25 ms | 0.99 | 0.99 |
| | 1,000,000 | 89.60 ms | 90.30 ms | 1.01 | 0.92 |
| `value_counts` high-cardinality text, one core | 100 | 14.69 µs | 3.71 µs | 0.25 | 0.24 |
| | 1,000 | 41.08 µs | 26.78 µs | 0.65 | 0.65 |
| | 10,000 | 480.1 µs | 429.2 µs | 0.89 | 1.09 |
| | 100,000 | 6.77 ms | 6.76 ms | 0.98 | 0.98 |
| | 1,000,000 | 139.4 ms | 140.0 ms | 0.95 | 0.87 |
| `value_counts` high-cardinality integers, pool | 10,000 | 870.2 µs | 673.4 µs | 0.75 | 0.84 |
| | 1,000,000 | 53.80 ms | 54.08 ms | 1.02 | 0.95 |
| `value_counts` high-cardinality text, pool | 10,000 | 999.7 µs | 911.0 µs | 0.91 | 0.85 |
| | 100,000 | 8.84 ms | 7.28 ms | 0.82 | 0.92 |
| | 1,000,000 | 68.50 ms | 68.20 ms | 0.99 | 1.03 |
| `fill_na` mean, Float64 | 100 | 2.43 µs | 2.59 µs | 1.07 | 1.05 |
| | 10,000 | 117.9 µs | 114.6 µs | 0.97 | 0.96 |
| | 1,000,000 | 13.24 ms | 13.38 ms | 1.01 | 1.01 |
| `fill_na` mean, Int64 | 100 | 2.48 µs | 2.62 µs | 1.04 | 1.06 |
| | 10,000 | 95.79 µs | 92.34 µs | 0.96 | 0.95 |
| | 1,000,000 | 11.42 ms | 11.39 ms | 1.01 | 1.00 |
| `fill_na` mode, Int64 (new; base column: the base's mean fill) | 100 | 2.48 µs | 6.81 µs | 2.73 | 2.72 |
| | 10,000 | 95.79 µs | 161.2 µs | 1.68 | 1.63 |
| | 1,000,000 | 11.42 ms | 16.55 ms | 1.44 | 1.44 |
| `preview_replace`, a match | 100 | 0.66 µs | 6.41 µs | 9.49 | 9.30 |
| | 100,000 | 563.3 µs | 1.48 ms | 2.63 | 2.68 |
| `preview_replace`, no match | 100 | 0.54 µs | 1.98 µs | 3.74 | 4.29 |
| | 100,000 | 289.0 µs | 1.47 ms | 4.98 | 5.38 |
| `preview_fill_na`, constant on a clean frame | 100-100,000 | 0.24 µs | 2.25 µs | 9.5 | 9.4 |

Every other size of these rows is in the same range; fill mean at 1,000 and 100,000 rows is within
1.01x. The previews now run the operation (above) and are accepted by the maintainer (2026-11-08) as
measured earlier (2.3-5x and about +1.4 µs); the n = 100 match row and the fill preview's +2.0 µs here
are above those figures. The high-cardinality `value_counts` rows were re-measured on 2026-11-09 (same
conditions, five rounds, load average 1.7-3.5, base rebuilt in the same run) after ties came to follow first
appearance (see "Value counts"); with ties ordered by value they had run 1.19-3.01x of the base. In that
run the low-cardinality rows ran 0.18-1.06x, mean fill 0.88-1.06x, `preview_replace` 2.8-5.5x (9.4x for
the n = 100 match row) and the fill preview +2.05 µs; the other rows of the table are from 2026-11-08.
Mean fill is unchanged (within 1.1x).

## Not covered here (Tier 4)

DATA-11 (integer normalize and pct_change), DATA-36 (row and column order of deduplicate, group_by,
stack), DATA-71 (rank ties).
