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
order. With an integer `time` column the shift is time-aware instead, as `plm::lag` (see "Tier 4 pins").
An integer column stays integer in `lag` and `diff`; NA is a missing value, not NaN.

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

An integer factor and the full dummy set, as `model.matrix(~ 0 + factor(g))`. The level order, the
plain level names, the `drop_first` baseline and missing values are pinned on text and float factors in
"Tier 4 pins".

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
error. A missing value is never matched: that is `fill_na`. Floats match exactly, as R's `x == old`
(`-0` matches `0`). A replacement that matches nothing returns the column as it is.

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
take part, as in `dplyr:::join_cols`, although inner, left and full joins then drop them. Where dplyr
silently drops a column under an empty suffix, an empty suffix that leaves a name on both sides is an error.

A full join has one column per key, as `dplyr::full_join` (the right keys are coalesced into the left
ones and take no suffix).

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

`release`, `ci` features, against `ff8e7559`, five interleaved rounds (order alternating) on a quiet
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
1.01x. The previews now run the operation (above) and are accepted by the maintainer: `preview_replace`
at 2.3-5.5x (9.5x for the n = 100 match row) and the fill preview at about +2 µs (2026-11-08, extended
2026-11-09). Mode fill is a new strategy and runs 1.4-2.7x of the base's mean fill, which is a different
computation; it was accepted as such. The high-cardinality `value_counts` rows were re-measured on 2026-11-09 (same
conditions, five rounds, load average 1.7-3.5, base rebuilt in the same run) after ties came to follow first
appearance (see "Value counts"); with ties ordered by value they had run 1.19-3.01x of the base. In that
run the low-cardinality rows ran 0.18-1.06x, mean fill 0.88-1.06x, `preview_replace` 2.8-5.5x (9.4x for
the n = 100 match row) and the fill preview +2.05 µs; the other rows of the table are from 2026-11-08.
Mean fill is unchanged (within 1.1x).

### Tier 4 (2026-11-17)

`release`, `ci` features, against `f80b0131` (v0.5.0, the base of the paths waves 1-3 did not rewrite)
and `90a99318` (v0.5.2, the base of the `features.rs` and `clean.rs` rows); 9 interleaved processes per
tree, the tree order alternating each round, `taskset -c 2`, `RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`,
working directory `/dev/shm`. Each process times every row at one n, each row the median of 41-2,001
calls after 10 warm-ups; a cell is the median of the 9 process medians and "vs" the ratio of those
medians (this tree over the base). Data (seeded ChaCha8): panels of 10 rows per group, 10% missing; join
keys 1:1 (a permutation) and 1:10. The `f80b0131` and "this tree" columns and "vs f80b0131" are the
quiet re-time at the integration (2026-11-18, merged head 156e744f, nothing else running, load average
1.3-2.7). The `90a99318` column and "vs 90a99318" are the lane's run (load average 2.2-15.1, median
7.3), not repeated.

| Row | n | f80b0131 | 90a99318 | this tree | vs f80b0131 | vs 90a99318 |
|---|---:|---:|---:|---:|---:|---:|
| `pct_change`, Float64 (DATA-11) | 100 | 46.83 µs | 46.80 µs | 4.54 µs | 0.10 | 0.10 |
|  | 1,000 | 48.97 µs | 49.92 µs | 6.15 µs | 0.13 | 0.12 |
|  | 10,000 | 66.99 µs | 67.70 µs | 22.63 µs | 0.34 | 0.33 |
| `pct_change`, Int64 (DATA-11) | 100 | 47.56 µs | 48.58 µs | 5.80 µs | 0.12 | 0.12 |
|  | 1,000 | 53.43 µs | 54.76 µs | 8.74 µs | 0.16 | 0.16 |
|  | 10,000 | 110.65 µs | 115.60 µs | 33.54 µs | 0.30 | 0.29 |
| `normalize`, 2 Int64 + 2 Float64 (DATA-11) | 100 | 186.49 µs | 191.06 µs | 14.84 µs | 0.08 | 0.08 |
|  | 1,000 | 234.41 µs | 238.02 µs | 61.24 µs | 0.26 | 0.26 |
|  | 10,000 | 707.02 µs | 712.38 µs | 530.79 µs | 0.75 | 0.76 |
| `diff`, Float64, no groups (DATA-29) | 100 | 25.86 µs | 25.68 µs | 3.14 µs | 0.12 | 0.12 |
|  | 1,000 | 26.40 µs | 25.89 µs | 3.54 µs | 0.13 | 0.14 |
|  | 10,000 | 31.96 µs | 31.60 µs | 9.10 µs | 0.28 | 0.29 |
| `lag`, no groups | 100 | 20.77 µs | 20.50 µs | 1.88 µs | 0.09 | 0.09 |
|  | 1,000 | 20.56 µs | 20.68 µs | 1.87 µs | 0.09 | 0.09 |
|  | 10,000 | 20.54 µs | 20.86 µs | 1.89 µs | 0.09 | 0.09 |
| `lag`, 10 rows per group | 100 | 99.99 µs | 101.42 µs | 102.72 µs | 1.03 | 1.03 |
|  | 1,000 | 327.01 µs | 334.19 µs | 333.12 µs | 1.02 | 1.00 |
|  | 10,000 | 2.61 ms | 2.68 ms | 2.65 ms | 1.02 | 1.01 |
| `lead`, 10 rows per group | 100 | 100.72 µs | 101.91 µs | 102.63 µs | 1.02 | 1.04 |
|  | 1,000 | 328.08 µs | 335.57 µs | 333.00 µs | 1.01 | 1.00 |
|  | 10,000 | 2.62 ms | 2.69 ms | 2.65 ms | 1.01 | 1.01 |
| `lag`, grouped, 10% of periods missing | 100 | 99.44 µs | 100.21 µs | 100.49 µs | 1.01 | 1.01 |
|  | 1,000 | 329.25 µs | 341.68 µs | 332.32 µs | 1.01 | 0.99 |
|  | 10,000 | 2.64 ms | 2.70 ms | 2.68 ms | 1.02 | 1.00 |
| `fill_na` forward, Float64, no groups | 100 | 3.04 µs | 3.04 µs | 3.10 µs | 1.02 | 1.00 |
|  | 1,000 | 12.88 µs | 12.96 µs | 13.04 µs | 1.01 | 1.01 |
|  | 10,000 | 127.28 µs | 120.87 µs | 128.09 µs | 1.01 | 0.99 |
| `fill_na` forward, text, no groups | 100 | 2.21 µs | 2.28 µs | 2.26 µs | 1.02 | 1.00 |
|  | 1,000 | 5.71 µs | 6.24 µs | 5.71 µs | 1.00 | 0.92 |
|  | 10,000 | 40.17 µs | 45.80 µs | 40.24 µs | 1.00 | 1.00 |
| `one_hot_encode`, 10 levels (DATA-40/41) | 100 | 48.64 µs | 49.02 µs | 18.86 µs | 0.39 | 0.37 |
|  | 1,000 | 196.34 µs | 199.66 µs | 57.15 µs | 0.29 | 0.28 |
|  | 10,000 | 1.66 ms | 1.72 ms | 461.26 µs | 0.28 | 0.26 |
| `one_hot_encode`, n/10 levels | 100 | 42.79 µs | 42.63 µs | 17.55 µs | 0.41 | 0.40 |
|  | 1,000 | 1.01 ms | 1.03 ms | 367.19 µs | 0.36 | 0.36 |
|  | 10,000 | 72.29 ms | 72.70 ms | 26.75 ms | 0.37 | 0.37 |
| `rank`, Float64 with ties (DATA-71) | 100 | 22.60 µs | 22.00 µs | 21.85 µs | 0.97 | 1.02 |
|  | 1,000 | 41.76 µs | 43.30 µs | 41.60 µs | 1.00 | 1.03 |
|  | 10,000 | 234.40 µs | 235.54 µs | 240.37 µs | 1.03 | 1.05 |
| `join` full, 1:1 keys (DATA-43) | 100 | 88.28 µs | 89.81 µs | 90.89 µs | 1.03 | 1.04 |
|  | 1,000 | 149.78 µs | 151.97 µs | 151.48 µs | 1.01 | 1.00 |
|  | 10,000 | 656.77 µs | 660.40 µs | 691.73 µs | 1.05 | 1.05 |
| `join` full, 1:10 keys | 100 | 84.02 µs | 85.31 µs | 86.36 µs | 1.03 | 1.03 |
|  | 1,000 | 98.46 µs | 100.26 µs | 104.56 µs | 1.06 | 1.06 |
|  | 10,000 | 245.03 µs | 247.37 µs | 277.07 µs | 1.13 | 1.13 |
| `join` left, 1:1 keys | 100 | 84.82 µs | 85.99 µs | 85.19 µs | 1.00 | 1.01 |
|  | 1,000 | 149.54 µs | 149.71 µs | 150.38 µs | 1.01 | 1.00 |
|  | 10,000 | 575.94 µs | 590.98 µs | 533.55 µs | 0.93 | 0.99 |
| `join` left, 1:10 keys | 100 | 80.91 µs | 81.78 µs | 80.98 µs | 1.00 | 1.01 |
|  | 1,000 | 99.67 µs | 100.59 µs | 100.67 µs | 1.01 | 1.00 |
|  | 10,000 | 294.86 µs | 300.03 µs | 297.49 µs | 1.01 | 1.00 |
| `join` inner, 1:1 keys | 100 | 50.30 µs | 51.06 µs | 50.79 µs | 1.01 | 1.00 |
|  | 1,000 | 113.82 µs | 117.37 µs | 112.99 µs | 0.99 | 0.99 |
|  | 10,000 | 525.95 µs | 539.18 µs | 525.80 µs | 1.00 | 1.00 |
| `join` inner, 1:10 keys | 100 | 46.37 µs | 46.94 µs | 46.64 µs | 1.01 | 1.00 |
|  | 1,000 | 37.37 µs | 63.18 µs | 37.92 µs | 1.01 | 1.00 |
|  | 10,000 | 200.04 µs | 203.13 µs | 200.43 µs | 1.00 | 1.01 |
| `estimate_join_rows` full, 1:1 | 100 | 100.20 µs | 101.84 µs | 99.95 µs | 1.00 | 0.99 |
|  | 1,000 | 244.47 µs | 248.33 µs | 243.01 µs | 0.99 | 0.99 |
|  | 10,000 | 1.59 ms | 1.62 ms | 1.57 ms | 0.99 | 0.97 |
| `estimate_join_rows` full, 1:10 | 100 | 91.35 µs | 93.24 µs | 90.68 µs | 0.99 | 0.99 |
|  | 1,000 | 125.04 µs | 127.59 µs | 125.31 µs | 1.00 | 1.01 |
|  | 10,000 | 482.51 µs | 514.92 µs | 477.82 µs | 0.99 | 1.00 |
| `join` full, 5% missing keys (more rows: U1) | 100 | 94.09 µs | 95.44 µs | 97.71 µs | 1.04 | 1.04 |
|  | 1,000 | 186.07 µs | 196.32 µs | 284.86 µs | 1.53 | 1.51 |
|  | 10,000 | 1.08 ms | 1.09 ms | 16.53 ms | 15.34 | 15.37 |
| `join` left, 5% missing keys (more rows: U1) | 100 | 104.21 µs | 105.92 µs | 108.29 µs | 1.04 | 1.03 |
|  | 1,000 | 187.02 µs | 187.39 µs | 226.74 µs | 1.21 | 1.23 |
|  | 10,000 | 798.65 µs | 801.59 µs | 7.02 ms | 8.79 | 9.24 |
| `join` inner, 5% missing keys (more rows: U1) | 100 | 51.63 µs | 53.49 µs | 53.48 µs | 1.04 | 1.01 |
|  | 1,000 | 122.80 µs | 97.25 µs | 108.89 µs | 0.89 | 1.15 |
|  | 10,000 | 626.42 µs | 718.93 µs | 6.54 ms | 10.45 | 9.59 |
| `concat`, 3 frames, one schema (DATA-44) | 100 | 2.99 µs | 3.04 µs | 3.09 µs | 1.03 | 1.04 |
|  | 1,000 | 2.98 µs | 2.98 µs | 3.09 µs | 1.04 | 1.04 |
|  | 10,000 | 2.96 µs | 3.00 µs | 3.06 µs | 1.04 | 1.05 |
| `deduplicate` keep first, 10% duplicates (DATA-36) | 100 | 34.81 µs | 35.00 µs | 32.24 µs | 0.93 | 0.93 |
|  | 1,000 | 141.82 µs | 135.20 µs | 120.76 µs | 0.85 | 0.89 |
|  | 10,000 | 1.39 ms | 1.30 ms | 1.27 ms | 0.91 | 0.98 |
| `deduplicate` keep last | 100 | 36.22 µs | 36.61 µs | 16.19 µs | 0.45 | 0.44 |
|  | 1,000 | 151.80 µs | 154.35 µs | 73.06 µs | 0.48 | 0.50 |
|  | 10,000 | 1.48 ms | 1.49 ms | 1.06 ms | 0.72 | 0.72 |
| `group_by`, 10 groups, sum + mean (DATA-36) | 100 | 55.66 µs | 57.50 µs | 86.67 µs | 1.56 | 1.51 |
|  | 1,000 | 274.95 µs | 269.22 µs | 299.96 µs | 1.09 | 1.12 |
|  | 10,000 | 462.60 µs | 447.32 µs | 483.22 µs | 1.04 | 1.06 |
| `group_by`, n/10 groups | 100 | 55.75 µs | 57.58 µs | 86.62 µs | 1.55 | 1.52 |
|  | 1,000 | 289.75 µs | 289.52 µs | 319.23 µs | 1.10 | 1.12 |
|  | 10,000 | 542.14 µs | 547.96 µs | 635.57 µs | 1.17 | 1.17 |
| `drop_na`, 4 Float64 (5% NaN) + 2 text | 100 | 29.88 µs | 30.35 µs | 38.01 µs | 1.27 | 1.26 |
|  | 1,000 | 22.97 µs | 22.84 µs | 50.34 µs | 2.19 | 2.22 |
|  | 10,000 | 91.40 µs | 99.30 µs | 177.64 µs | 1.94 | 1.84 |
| `drop_na`, the same without NaN | 100 | 29.65 µs | 29.78 µs | 29.90 µs | 1.01 | 1.00 |
|  | 1,000 | 23.05 µs | 23.19 µs | 24.61 µs | 1.07 | 1.09 |
|  | 10,000 | 95.83 µs | 99.73 µs | 112.14 µs | 1.17 | 1.19 |
| `filter` Int64 gt (DATA-45) | 100 | 22.25 µs | 22.21 µs | 21.84 µs | 0.98 | 1.02 |
|  | 1,000 | 35.62 µs | 35.47 µs | 35.54 µs | 1.00 | 1.02 |
|  | 10,000 | 184.19 µs | 186.48 µs | 183.04 µs | 0.99 | 0.98 |
| `filter` Float64 gt | 100 | 21.97 µs | 21.93 µs | 22.06 µs | 1.00 | 1.00 |
|  | 1,000 | 36.12 µs | 35.93 µs | 35.81 µs | 0.99 | 1.02 |
|  | 10,000 | 182.67 µs | 185.10 µs | 184.37 µs | 1.01 | 0.99 |
| `filter` text eq | 100 | 21.28 µs | 21.65 µs | 21.26 µs | 1.00 | 0.98 |
|  | 1,000 | 27.80 µs | 27.84 µs | 27.54 µs | 0.99 | 1.02 |
|  | 10,000 | 102.73 µs | 108.26 µs | 106.57 µs | 1.04 | 0.97 |
| `replace`, no match, Float64 | 100 | 1.98 µs | 2.08 µs | 1.57 µs | 0.79 | 0.78 |
|  | 1,000 | 9.43 µs | 10.11 µs | 6.54 µs | 0.69 | 0.67 |
|  | 10,000 | 96.75 µs | 100.03 µs | 69.10 µs | 0.71 | 0.69 |
| `replace`, no match, text | 100 | 4.33 µs | 4.31 µs | 1.43 µs | 0.33 | 0.34 |
|  | 1,000 | 30.14 µs | 31.10 µs | 4.53 µs | 0.15 | 0.15 |
|  | 10,000 | 301.65 µs | 306.37 µs | 49.30 µs | 0.16 | 0.16 |
| `replace`, no match, categorical | 100 | 11.72 µs | 11.55 µs | 0.77 µs | 0.07 | 0.07 |
|  | 1,000 | 84.54 µs | 86.92 µs | 0.78 µs | 0.01 | 0.01 |
|  | 10,000 | 861.88 µs | 860.93 µs | 0.77 µs | 0.00 | 0.00 |
| `replace`, a match, Float64 | 100 | 1.88 µs | 2.03 µs | 2.09 µs | 1.12 | 1.05 |
|  | 1,000 | 8.47 µs | 9.14 µs | 9.19 µs | 1.08 | 1.00 |
|  | 10,000 | 73.17 µs | 79.96 µs | 78.63 µs | 1.07 | 0.99 |
| `replace`, a match, text | 100 | 4.52 µs | 4.45 µs | 4.68 µs | 1.04 | 1.05 |
|  | 1,000 | 31.80 µs | 32.66 µs | 32.10 µs | 1.01 | 1.02 |
|  | 10,000 | 319.11 µs | 324.67 µs | 319.77 µs | 1.00 | 1.00 |
| `replace`, a match, categorical | 100 | 11.89 µs | 11.82 µs | 12.40 µs | 1.04 | 1.04 |
|  | 1,000 | 86.56 µs | 89.02 µs | 88.61 µs | 1.02 | 1.00 |
|  | 10,000 | 883.53 µs | 880.74 µs | 866.25 µs | 0.98 | 1.00 |
| `cast_with` text to float, every value fails | 100 | 4.20 µs | 4.32 µs | 4.36 µs | 1.04 | 0.89 |
|  | 1,000 | 29.36 µs | 30.95 µs | 32.04 µs | 1.09 | 0.90 |
|  | 10,000 | 274.93 µs | 285.96 µs | 300.68 µs | 1.09 | 0.89 |
| `fill_na` mode, 10 columns without missing cells | 100 | 37.09 µs | 38.91 µs | 12.75 µs | 0.34 | 0.34 |
|  | 1,000 | 108.86 µs | 109.06 µs | 12.77 µs | 0.12 | 0.12 |
|  | 10,000 | 824.42 µs | 833.01 µs | 12.70 µs | 0.02 | 0.02 |
| `diff_with`, grouped (new) | 100 | - | - | 110.26 µs | - | - |
|  | 1,000 | - | - | 342.41 µs | - | - |
|  | 10,000 | - | - | 2.68 ms | - | - |
| `fill_na_by` forward, Float64 (new) | 100 | - | - | 95.29 µs | - | - |
|  | 1,000 | - | - | 149.37 µs | - | - |
|  | 10,000 | - | - | 699.11 µs | - | - |
| `fill_na_by` forward, text (new) | 100 | - | - | 94.24 µs | - | - |
|  | 1,000 | - | - | 143.87 µs | - | - |
|  | 10,000 | - | - | 632.61 µs | - | - |
| `lag_with`, time index, 10% of periods missing (new) | 100 | - | - | 180.76 µs | - | - |
|  | 1,000 | - | - | 358.19 µs | - | - |
|  | 10,000 | - | - | 2.51 ms | - | - |
| `concat`, Int64 with Float64 (new; the base refused it) | 100 | - | - | 3.88 µs | - | - |
|  | 1,000 | - | - | 4.40 µs | - | - |
|  | 10,000 | - | - | 8.65 µs | - | - |

Rows of existing paths above 1.10x:

- `drop_na` with NaN cells (1.27 / 2.19 / 1.94x; 29.9 to 38.0 µs, 23.0 to 50.3 µs, 91.4 to 177.6 µs): rows with NaN are dropped now (R's `is.na`), which costs one NaN
  mask per float column that holds NaN; the base dropped nothing for NaN, so it did less and returned
  more rows. A float column without NaN pays one scan (1.00 / 1.07 / 1.17x; 95.8 to 112.1 µs at 10,000).
- `group_by` (1.56x at n = 100, 55.7 to 86.7 µs; 1.05-1.17x above, 542 to 636 µs with n/10 groups at 10,000): the groups are sorted by their keys,
  missing last (U3, DATA-36); the base returned them in hash order. The sort costs the same as a plan step
  or on the collected frame.
- `join` full with 1:10 keys at n = 10,000 (1.13x, 245.0 to 277.1 µs): the key columns are coalesced (DATA-43).
- `replace` with a match on Float64 at n = 100 (1.12x, 1.88 to 2.09 µs; 1.08-1.09x above): not profiled.
- Joins with missing keys (1.2-1.5x at 1,000, 8.8-15.3x at 10,000; the full join 1.08 to 16.53 ms): missing
  keys now match each other (U1), so the result has more rows (with 5% missing keys on both sides, about
  (n/20)^2 more); the null-free rows above are the comparison. Peak RSS of the full join at 10,000 rows:
  25.3 MiB base, 40.8 MiB head (`/usr/bin/time`, median of 3).

Rule 5: none of these rows is in a sign-off (the `group_by` sort comes back to the user above 1.10x by the
2026-11-17 decisions): awaiting sign-off.

The positional shifts, `normalize`, one-hot and the categorical replace run their own column arithmetic
instead of a query plan, which is where the 0.1-0.4x come from. The new opt-in paths have no base:
grouped `diff` runs at 1.0-1.1x of the base's grouped `lag` (one `.over` partition); a grouped forward
fill at 0.3-1.0x of the base's grouped `lag` and 3-32x of its ungrouped fill; the time-indexed `lag` at
0.96-1.81x of the base's grouped positional `lag` (180.8 µs, 358.2 µs and 2.51 ms against 100.0 µs,
327.0 µs and 2.61 ms; a key check and a hash join), within the estimated 3-6x of the 2026-11-17 decisions.

## Tier 4 pins

Cases in `validation/r/cases/data_munging_t4.R`, registry `validation/methods/data_munging_t4.json`.
Conventions (decided 2026-11-17): missing join keys match each other (U1, dplyr's default
`na_matches = "na"` and `base::merge`); a missing one-hot value is missing in every dummy (U2,
`model.matrix` under `na.action = na.pass`); grouped results are sorted by their keys, missing last
(U3, `dplyr::summarise`); a time index is an integer column (U4, `munge_lag_lead` and `munge_diff`).

- Panel shifts (`data_munging_panel_shift`, on `munging_panel.csv`: integer `x`, unsorted rows, interleaved
  firms, firm 3 without 2003). Grouped `diff` and `pct_change` as `dplyr::mutate(group_by(d, firm), ...)`;
  `tidyr::fill` by firm, down and up; time-indexed lag, lead and diff as `plm::lag`/`lead`/`diff` on
  `pdata.frame(index = c("firm", "year"))`, mapped back to data order; `pct_change` and `normalize` of an
  integer column in floating point. Tests: `test_validate_panel_shift_integer`,
  `test_validate_panel_shift_grouped`, `test_validate_panel_shift_time_index`,
  `test_validate_fill_grouped_tidyr`, `test_validate_mcp_diff_group_by`, `test_validate_mcp_lag_time_index`.
- One-hot and rank (`data_munging_factor_t4`, on `munging_regions.csv`): levels as `levels(factor(x))`
  (`east, north, south, west`; floats as `1.5, 2.5, 3`), dummies named `<column>_<level>`, NA in every
  dummy, `drop_first` dropping `east`; `rank` with ties averaged as `rank(ties.method = "average")`.
  Tests: `test_validate_one_hot_model_matrix_na`, `test_validate_mcp_one_hot_names`, `test_validate_rank_average`.
- Joins, binds and order (`data_munging_combine_t4`): full, inner, semi and anti joins with missing keys
  (one key column; NA keys match); `dplyr::bind_rows` of an integer and a double `x` (double) and of
  `Income`/`income` (two half-missing columns); `distinct` keeping first, last and none in data order;
  `summarise` by group, NA last. Tests: `test_validate_join_na_keys_dplyr`, `test_validate_concat_bind_rows`,
  `test_validate_distinct_order`, `test_validate_group_by_order`.

Tolerances: exact for lags, leads, fills, dummies, ranks, names and integer values; 1e-14 relative for
differences, ratios and normalize (the same IEEE operations as R); group sums 1e-10.
