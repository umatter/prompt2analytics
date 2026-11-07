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

## Melt (pivot_longer)

As `tidyr::pivot_longer`: row-major long format, missing values stay missing, and the value column has
the polars supertype of the value columns (integer + double gives double; integer + integer stays
integer; u32, f32 and bool keep their type on their own).

Text melted with non-text columns is refused, as tidyr refuses it ("Can't combine `a` <integer> and `s`
<character>."); a date or time column melted with a column of another type is refused too (vctrs does,
and the supertype would give day counts). Date and time columns among themselves take the polars
supertype.

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
counts order by value, ascending, missing last.

Case `data_munging_value_counts`, on `munging_counts.csv`. Decisive values: `vc_n = [4, 3, 2, 1]`,
`vc_proportion = [0.4, 0.3, 0.2, 0.1]`. Tests: `test_validate_value_counts`,
`test_value_counts_normalize`.

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
