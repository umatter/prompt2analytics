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
