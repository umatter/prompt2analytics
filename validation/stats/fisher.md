# Validation: Fisher's Exact Test

## Method and options

Fisher's exact test of independence in a 2 × 2 table, as R's `stats::fisher.test` computes it
for a 2 × 2 table (R 4.3.3): the same p-values, the same conditional maximum-likelihood
estimate of the odds ratio and the same exact conditional confidence interval.

- Core: `p2a_core::stats::fisher::{fisher_exact_test, fisher_exact_test_int, run_fisher_test}`;
  MCP tool `hypothesis_fisher_exact`; CLI `p2a stats fisher`.
- `table` is `[[a, b], [c, d]]` of whole-number counts; `alternative` is `TwoSided` (default),
  `Greater` or `Less`; `conf_level` (for example 0.95) adds the confidence interval.
- `FisherExactResult.conditional_mle` is R's `estimate` (the conditional MLE);
  `odds_ratio` is the sample odds ratio `(a d) / (b c)`, as before. MCP returns both
  (`conditional_mle`, `sample_odds_ratio`; `odds_ratio` is the sample odds ratio) and labels
  the interval `odds_ratio_ci.method = "exact conditional (as R fisher.test)"`.

### Data, missing values and level order

`run_fisher_test(dataset, row, col, ..)` builds the table with
`p2a_core::stats::contingency_table` (`crates/p2a-core/src/stats/contingency.rs`):

- Rows with a missing value (null, or NaN in a float column) in either column are dropped,
  as R's `table()` drops `NA`, and reported in `FisherExactResult.missing` (Display prints a
  `Missing values:` line; MCP returns `missing`).
- Levels follow R `factor()` order: numbers by value (codes 2 and 10 give the rows "2", "10"),
  text in byte order (R's order under `LC_COLLATE=C`), booleans FALSE < TRUE, dates
  chronologically. The first level is the first row or column; `row_levels` and
  `col_levels` carry the labels.
- The levels are those of the kept rows. R's `table(x, y)` keeps a level of `x` that occurs
  only where `y` is missing as an all-zero row; p2a drops it, so such a column has one level
  fewer than in R.
- A table that is not 2 × 2 after dropping missing rows is refused, naming the levels of
  both columns: p2a implements the 2 × 2 case of `fisher.test` only.
- Datetime, struct, list and array columns are refused as categorical variables.

### Algorithm (R's 2 × 2 branch)

With `m = a + c`, `n = b + d`, `k = a + b` and `x = a`, the top-left cell ranges over
`lo = max(0, k - n) ..= hi = min(k, m)`. For an odds ratio `psi` its distribution is
`P(X = s) ∝ C(m, s) C(n, k - s) psi^s` (R's `dnhyper`), computed in log space
(`ln C` from `statrs::function::factorial::ln_binomial`, shifted by the maximum before
`exp`), so no table overflows (for the precision of very large tables, see "Known
differences").

- Two-sided p: the sum of `P(X = s)` over the support points with
  `P(X = s) <= P(X = x) (1 + 1e-7)` (R's `relErr`), at most 1.
- `less`: `P(X <= x)`; `greater`: `P(X >= x)`, both under independence (R's
  `pnhyper(x, 1)`, which R computes with `phyper`), as a ratio of sums of the same
  normalised weights. statrs' `Hypergeometric::cdf` sums the central probabilities one by
  one from 0: for `[[50, 1], [1, 50]]` it gives `P(X <= 50) = 0.99999999999995` where R and
  p2a give 1, and it adds one zero term for every value below the support.
- Conditional MLE (R's `mle`): 0 when `x = lo`, +Inf when `x = hi`; otherwise the `psi`
  where the mean of `X` equals `x`, solved on `(0, 1)` when the mean at `psi = 1` exceeds
  `x`, else as `1 / t` with `t` on `(eps, 1)`.
- Limits (R's `ncp.U`, `ncp.L`): `U(alpha)` solves `P(X <= x; psi) = alpha` (+Inf when
  `x = hi`), `L(alpha)` solves `P(X >= x; psi) = alpha` (0 when `x = lo`), with the same
  brackets and `1/t` substitution. The interval is `(L(alpha/2), U(alpha/2))` for
  `two.sided`, `(0, U(alpha))` for `less` and `(L(alpha), Inf)` for `greater`, with
  `alpha = 1 - conf_level`.
- Root finding: Brent's method exactly as R's `R_zeroin2` (`src/library/stats/src/zeroin.c`)
  with `tol = 0`, so the step tolerance is `2 eps |t|`, and at most 2000 iterations. R calls
  `uniroot` with its default `tol = .Machine$double.eps^0.25`, an absolute tolerance of about
  1e-4 on the `t` scale. When a root lies below `eps` (an odds ratio above 4.5e15, where R's
  `uniroot` stops with "f() values at end points not of opposite sign"), the bracket is widened
  to (`f64::MIN_POSITIVE`, 1), and to +Inf beyond that; no case reaches this.
- Counts must be finite whole numbers up to 2^53; NaN, ±Inf, fractions and larger values are
  refused, naming the cell (R rounds non-integers with a warning). Every pass over the
  support and every root-finding iteration calls `resource::checkpoint()`.

## R reference

Cases in `validation/r/cases/stats_contingency.R`:

- `stat_fisher_tables` (`validation/datasets/generated/fisher_tables.csv`): eleven 2 × 2
  tables, `fisher.test(matrix(c(a, b, c, d), 2, byrow = TRUE), alternative = ...)`, plus the
  same equations re-solved with `uniroot(tol = 1e-15, maxiter = 10000)` (keys `*_tight`).
- `stat_fisher_rows_2000` (`validation/datasets/generated/fisher_rows_2000.csv`): 2000 rows,
  `fisher.test(table(treat, outcome))`.
- `stat_contingency_na_codes` (`validation/datasets/generated/contingency_na_codes.csv`):
  `fisher.test(table(code, out))` with numeric codes 2 and 10 and one missing code.
- `stat_chisq_small_expected`: a 3 × 2 table, which `run_fisher_test` refuses.

### Pinned values (R 4.3.3; the case files hold 17 digits)

| table (a, b, c, d) | p two-sided | p less | p greater | cond. MLE (tight) | two-sided 95% CI (tight) |
|---|---|---|---|---|---|
| tea (3, 1, 1, 3) | 0.4857142857142858 | 0.9857142857142858 | 0.24285714285714288 | 6.408319658199671 | 0.21173559544657847, 626.2435305888135 |
| n1000 (260, 240, 240, 260) | 0.22947618847501033 | 0.9079506543045864 | 0.1147380942375051 | 1.1734230474766068 | 0.9086443280672413, 1.5158774870626694 |
| n1030 (265, 250, 250, 265) | 0.3829736392542066 | 0.840637665369774 | 0.1914868196271034 | 1.1234728071995923 | 0.8733737350788962, 1.4455394147356735 |
| n1040 (270, 250, 250, 270) | 0.23865072025381845 | 0.9036164246190448 | 0.1193253601269092 | 1.1662272127990503 | 0.9077037278172203, 1.4988545036237213 |
| n2000 (520, 480, 480, 520) | 0.08111006057913338 | 0.9666548918351516 | 0.04055503028956669 | 1.1735170755267106 | 0.9809033190255882, 1.4041857925265708 |
| n2000_sig (600, 400, 400, 600) | 4.2999524250476395e-19 | 1 | 2.1499762125237877e-19 | 2.249062695271797 | 1.8737613313883108, 2.701864802369515 |
| tiny_p (100, 10, 10, 100) | 4.906449339029371e-38 | 1 | 2.453224669514709e-38 | 95.5989668056595 | 36.90980394665555, 279.74227772698754 |
| rare (10, 1990, 20, 1980) | 0.09747735831867001 | 0.048738679159335066 | 0.9789969971592418 | 0.4975710532495369 | 0.2074346299118897, 1.1171023996494474 |
| unbal (100, 900, 120, 880) | 0.1744071041043372 | 0.08720355205216858 | 0.9333648487815597 | 0.8148981332067641 | 0.6083934477531157, 1.089728598973796 |
| zero_cell (0, 5, 3, 2) | 0.16666666666666657 | 0.08333333333333331 | 1 | 0 | 0, 2.026812771666662 |
| null_2400 (600, 600, 600, 600) | 1 | 0.5162816615831695 | 0.5162816615831692 | 1 | 0.8493349178048262, 1.1773918380567463 |

The one-sided limits (`less`: upper, `greater`: lower) are in the case file as well; the
`less` lower limit is 0 and the `greater` upper limit +Inf for every table (the harness
stores Inf as null; the test asserts `f64::INFINITY`).

`stat_fisher_rows_2000` gives the n2000 values from the two columns (rows `control`,
`treated`; columns `no`, `yes`). `stat_contingency_na_codes`: the table of `code` by `out`
after dropping the missing code is [4, 2, 1, 4] with rows "2", "10"; p 0.2424242424242425,
sample odds ratio 8, conditional MLE 6.439146409959153, 95% CI 0.32490046755594026,
490.15627523100136 (tight).

### Tight limits, not R's printed ones, are the reference

`fisher.test`'s printed estimate and limits carry the error of `uniroot(tol = eps^0.25)`: on
the eleven tables they are up to 0.69% off (tea: upper limit 621.9337505454173 printed,
626.2435305888135 exact). On extreme tables they are not usable: for `[[50, 1], [1, 50]]` R
prints the interval (127.00893441058923, 4503599627370496) and the estimate
1530.9010005032094, where the exact values are (126.58051644404978, 102061.81206230588) and
1512.5074333693019. p2a solves R's equations to full precision; the tests pin the `*_tight`
values at 1e-9 relative and R's printed values as a cross-check at 1e-2 relative.

## Tests

- `p2a-core`, `crates/p2a-core/src/stats/fisher.rs`:
  `stats::fisher::tests::test_validate_fisher_tables` (11 tables × 3 alternatives: p-values,
  conditional MLE, all limits, sample odds ratio, n; the four p-values below 1e-12 also at
  1e-9 relative), `stats::fisher::tests::test_validate_fisher_tables_row_swap` (the swapped
  tables give 1/psi, 1/U, 1/L and exchange `less` and `greater`; the swapped zero-cell table
  reaches `x = hi`), `stats::fisher::tests::test_validate_fisher_rows_2000`,
  `stats::fisher::tests::test_validate_fisher_na_codes_orientation`,
  `stats::fisher::tests::test_validate_fisher_needs_two_levels_per_column` and
  `stats::fisher::tests::test_validate_run_fisher_test_inside_tokio_runtime`.
- Not case-loading: `stats::fisher::tests::test_fisher_extreme_table_matches_tight_r`
  (`[[50, 1], [1, 50]]` against the tight re-solve, values in the test),
  `stats::fisher::tests::test_fisher_rejects_invalid_counts`,
  `stats::fisher::tests::test_fisher_one_sided_p_does_not_walk_below_the_support` and
  `stats::fisher::tests::test_fisher_honours_cancellation`.
- `p2a-mcp`: `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_fisher_exact_rows_2000`
  (inside a multi-thread tokio runtime; the upper limit of `greater` is the string `"Inf"`).
- The contingency builder: `stats::contingency::tests::test_validate_contingency_table_na_codes`
  and the other tests of method `contingency_table`.
- Registry: `validation/methods/stats_contingency.json`, method `fisher_exact`.

### Legacy tables (R-pinned)

Case `stat_fisher_legacy` (`validation/r/cases/stats_power_loglin.R`, dataset
`validation/datasets/generated/fisher_legacy.csv`, one row per table `a, b, c, d`): `fisher.test`
for the three alternatives with the limits and estimate re-solved tight as above. Each test runs
`check_legacy(<table>)`: p for `two.sided`, `less` and `greater` at 1e-9 relative; the two-sided
limits, the upper `less` and lower `greater` limit and the conditional MLE against the tight
re-solve at 1e-9 relative (1e-12 absolute); the sample odds ratio at 1e-15 relative and n exactly;
the `less` lower limit 0 and the `greater` upper limit `Inf`.

| Test | Table | Two-sided p | Tight CI (two-sided) |
|---|---|---|---|
| `stats::fisher::tests::test_validate_against_r_basic` | `basic` `[[1, 11], [9, 3]]` | 0.002759456185220086 | 0.0006360029488751081, 0.42586647569637387 |
| `stats::fisher::tests::test_validate_against_r_lady_tea` | `tea` `[[3, 1], [1, 3]]` | 0.4857142857142858 | 0.21173559544657847, 626.2435305888135 |
| `stats::fisher::tests::test_validate_against_r_one_sided` | `one_sided` `[[6, 2], [1, 7]]` | 0.04055944055944058 (greater 0.020279720279720286) | 1.0876028823630632, 1098.971432187753 |
| `stats::fisher::tests::test_validate_fisher_legacy_ci` | `ci` `[[10, 5], [5, 10]]` | 0.14311097805070672 | 0.7054372156512816, 23.921858982581192 |

Before T2-WP05 the CI of `ci` was (0, 40); the old test only checked that it contained the sample
odds ratio. Registry: `validation/methods/stats_power_loglin.json`, method `fisher_legacy`.

### Quoted values

`stats::fisher::tests::test_zero_cells` (`[[0, 5], [5, 10]]`, p 0.2663183695) and
`stats::fisher::tests::test_extreme_table` (`[[50, 1], [1, 50]]`, p 1.302273445e-26; R's printed
"odds ratio 1530.9" is the imprecise `uniroot` solution of the conditional MLE,
1512.5074333693019) compare with values quoted from R.

## Tolerance

- p-values: the p-value row of `validation/README.md` (`Tol::p_value(n)`); the four p-values
  below 1e-12 of `stat_fisher_tables` also at 1e-9 relative, since the absolute floor of
  `Tol::p_value` would hide a floored value.
- Conditional MLE and limits: 1e-9 relative (1e-12 absolute) against the tight re-solve;
  R's printed values at 1e-2 relative.
- Sample odds ratio 1e-15 relative; counts and n exact.

## Known differences

1. 2 × 2 tables only. R's `fisher.test` also handles r × c tables (network algorithm); use the
   chi-squared test of independence for larger tables.
2. Counts must be finite whole numbers (R rounds non-integers with a warning).
3. The levels are those of the kept rows (see "Data, missing values and level order").
4. `odds_ratio` is still the sample odds ratio; the conditional MLE, which R reports as the
   estimate, is `conditional_mle`.
5. Very large tables lose precision. The log weights are differences of `ln_binomial` values
   (`statrs`' `ln_gamma`), which cancel, so their relative error is of order
   `N ln N × 1e-16` for `N = a + b + c + d`: about 3e-9 at `N = 1e6`, 5e-6 at `N = 1e9` and a
   few percent at `N = 1e12`. R's `dhyper` computes the weights with `dbinom_raw` (`stirlerr`
   and `bd0`) and stays accurate there. No pinned case is affected (`N <= 4000`);
   `test_fisher_one_sided_p_does_not_walk_below_the_support` is the `N = 1e12` example, and
   needs an absolute tolerance of 0.05 on its one-sided p-value.

## Performance

The benchmarks `Fisher/twosided` and `Fisher/with_ci` of
`cargo bench -p p2a-core --bench comprehensive_benchmarks` time the test; the timings
measured for the earlier algorithm no longer apply.

## References

- Fisher, R. A. (1935). "The logic of inductive inference". *Journal of the Royal Statistical
  Society*, 98(1), 39-82.
- Fisher, R. A. (1922). "On the interpretation of χ² from contingency tables, and the
  calculation of P". *Journal of the Royal Statistical Society*, 85(1), 87-94.
- Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html
