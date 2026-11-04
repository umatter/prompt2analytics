# Validation: Granger causality F test (lag orders and small samples)

Registry: [`validation/methods/ts_var.json`](../../methods/ts_var.json) (method
`granger_test_lags`). R cases: `granger_short` and `var_canada` in
[`validation/r/cases/ts_var.R`](../../r/cases/ts_var.R). Integer series and missing values:
see [../granger.md](../granger.md) (case `ts_granger_int`).

## Method

`granger_test(dataset, dependent, cause, p)` regresses, on the n - p rows t = p + 1..n,

- restricted: `y_t ~ 1 + y_{t-1} + .. + y_{t-p}`;
- unrestricted: the same plus `x_{t-1} + .. + x_{t-p}` (2p + 1 coefficients),

and reports

```
F = ((RSS_R - RSS_U) / p) / (RSS_U / (n - 3p - 1)),   F ~ F(p, n - 3p - 1),
```

as `lmtest::grangertest(y ~ x, order = p)`. Both residual sums of squares come from one
pivoted QR of the unrestricted design (`linalg::solve::Qr`, `dqrdc2`, tolerance 1e-7): its
first p + 1 columns are the restricted design, so `RSS_U` is the sum of the squared effects
`(Q'y)_i` for i >= 2p + 1 and `RSS_R` for i >= p + 1, the values `lm.fit` gives on each
design. There is no absolute exact-fit threshold: F is
invariant to the scale of the data, and an exact fit (RSS_U = 0) gives F = inf and p = 0, as
in R.

## Requirements

- `n >= 3p + 2`, so that df2 = n - 3p - 1 >= 1; otherwise `InsufficientData { required: 3p + 2 }`
  with a context naming `n - 3p - 1`. R stops there with "there are aliased coefficients in
  the model".
- Lags of full rank: an aliased lag (identical or proportional series) is `SingularMatrix`.

## Default order

`run_granger_test` (and MCP `ts_granger` without `lags`) uses Schwert's
`floor(12 (n/100)^(1/4))`, capped at `(n - 2) %/% 3`, the largest order with df2 >= 1. When
the cap applies, the result carries the warning "default lag order S (Schwert's rule ...)
reduced to p ..." and Display prints it as a `Warning:` line. Fewer than 5 rows is
`InsufficientData { required: 5 }`. `lmtest::grangertest` defaults to order 1; the capped
Schwert order is p2a's own convention, written out in the case.

| n | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Schwert | 7 | 7 | 7 | 7 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| cap (n - 2) %/% 3 | 4 | 5 | 5 | 5 | 6 | 6 | 6 | 7 | 7 | 7 | 8 |

## Reference values

`granger_short`: `grangertest(y ~ x, order = p)` on a 20-row series
(`validation/datasets/generated/granger_short.csv`).

| p | F | p-value | df1 | df2 | rows used |
|---|---|---|---|---|---|
| 1 | 9.841430752662788 | 0.006365073485181647 | 1 | 16 | 19 |
| 2 | 3.5636895434735982 | 0.05834650125983499 | 2 | 13 | 18 |
| 3 | 2.504535998479804 | 0.11862063342433446 | 3 | 10 | 17 |
| 4 | 1.1307456329744108 | 0.4147751511477751 | 4 | 7 | 16 |
| 5 | 0.628494342903565 | 0.6918359077819944 | 5 | 4 | 15 |
| 6 | 2.703820251949798 | 0.43461793623757494 | 6 | 1 | 14 |

`schwert_lag` 8, `default_lag` 6. Orders 7 and 8 are refused.

`var_canada`: `grangertest(prod ~ e, order = 2)` on `vars::Canada`: F 1.7024999542289105,
p 0.18901456524925964, df2 77.

## Tests and tolerances

- `test_validate_granger_short_lags`: F, p, df1, df2 and rows used at orders 1-6, and the
  default order 6 with its warning.
- `test_validate_granger_canada_prod_on_e`: the Canada test.
- Not R-pinned: `test_granger_rejects_lags_without_residual_df` (orders 7 and 8 on 20 rows,
  the 4-row default), `test_granger_rejects_proportional_series` (x = 3 y is
  `SingularMatrix`), `test_granger_is_scale_invariant` (the series times 1e-10 give the same
  F to 1e-8).

Tolerances (validation/README.md): F `Tol::stat`, p `Tol::p_value` (1e-5 relative / 1e-12
absolute at n = 20 and 82); counts exact.

```bash
cargo test -p p2a-core --lib --no-default-features --features ci -- econometrics::timeseries::granger
```

## History

Before Tier 3 the test required only n > 2p + 1, so df2 = n - 3p - 1 underflowed for n <= 3p
(a panic in a debug build, df2 ~ 1.8e19 in a release build), and the Schwert default reached
that range for 16-25 rows. An RSS below an absolute 1e-15 was taken as an exact fit, which gave
F = inf for series measured in small units.
