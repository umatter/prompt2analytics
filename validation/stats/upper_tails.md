# Upper-tail p-values below 1e-16

**Status:** R-pinned (registry entry `stats_upper_tails` in `validation/methods/numeric_hygiene.json`).

## The problem (X-17)

A p-value computed as `1 - cdf(x)` is exactly 0 once `cdf(x)` rounds to 1, which in double precision happens
for every tail below about 1.1e-16. R computes these tails directly (`pchisq(x, df, lower.tail = FALSE)`,
`pf(f, d1, d2, lower.tail = FALSE)`), so it reports 2.97e-82 where `1 - cdf` reports 0. A p-value of exactly 0
also breaks downstream code that ranks or adjusts p-values: two variables with p = 0 tie, and the tie is
broken by position instead of by evidence.

p2a computes every upper tail through the shared helpers in `traits::estimator`
(`chi_squared_p_value`, `f_test_p_value`, `t_test_p_value`, `normal_p_value_two_sided`), which evaluate the
tail directly. statrs' `ChiSquared::sf` and `FisherSnedecor::sf` are not used for p-values: `ChiSquared::sf` is
1 for x below about 1.1e-15 and `FisherSnedecor::sf` loses about 8e-9 relative accuracy at F = 1e8.

## R calls and cases (`validation/r/cases/numeric_tails.R`)

| Case | R call | Value | R | `1 - cdf` |
|---|---|---|---|---|
| `tail_mcnemar` | `mcnemar.test(matrix(c(25, 10, 400, 30), 2), correct = TRUE)` | p | 2.974837071294514e-82 | 0 |
| `tail_mcnemar` | same, `correct = FALSE` | p | 1.1475572687301979e-82 | 0 |
| `tail_oneway` | `oneway.test(y ~ g, var.equal = TRUE)` (3 groups of 12) | p | 6.387978375889893e-23 | 0 |
| `tail_oneway` | same, `var.equal = FALSE` | p | 5.689495500640302e-18 | 0 |
| `tail_prop_trend` | `prop.trend.test(c(10, 60, 140, 240), rep(300, 4))` | p | 1.084724362027203e-93 | 0 |
| `tail_var_test` | `pf(f, 39, 44, lower.tail = FALSE)`, f = 1153.56 | p (greater) | 9.860655910834909e-56 | 0 |
| `tail_var_test` | twice that | p (two-sided) | 1.9721311821669818e-55 | 0 |

The statistics, degrees of freedom and confidence limits of each call are pinned too.

## Tests

| Test | Tolerance |
|---|---|
| `stats::mcnemar::tests::test_validate_mcnemar_tail_below_1e16` | statistics `Tol::stat(465)`; p relative `Tol::p_value(465).rel`, no absolute floor |
| `stats::oneway::tests::test_validate_oneway_tail_below_1e16` | F and Welch df `Tol::stat(36)`; equal-variance df exact; p relative only |
| `stats::proptrendtest::tests::test_validate_prop_trend_tail_below_1e16` | statistic `Tol::stat(1200)`; p relative only |
| `stats::vartest::tests::test_validate_var_test_extreme_ratio` | statistic and limits `Tol::stat(85)`; df exact; p relative only |
| `stats::vartest::tests::test_validate_var_test_ci_quantiles_at_large_df` | limits at df 2e7, 1e-10 relative |

The p-value tolerance has no absolute floor: `Tol::p_value`'s absolute 1e-12 would accept 0 for any tail far
below it, which is the bug these tests exist to catch.

## var.test

R's own `var.test` computes `1 - pf` and prints 0 (two-sided and "greater") for this ratio. p2a computes both
tails directly, the lower one as `P(F <= f) = P(F' >= 1/f)` with `F' ~ F(d2, d1)`, and reports the two-sided
p-value as `min(1, 2 min(lower, upper))`. The case records both: `var_test_p_*` (what R prints, which p2a matches
within `Tol::p_value`'s absolute 1e-12) and `p_upper`, `p_two_sided_accurate` (the exact tail, which p2a matches
relatively). This is the registry deviation of the entry.

### Confidence limits at large df

Above df 4e5 R's `qf` returns `qchisq(p, df1) / df1`, which ignores the variance of the denominator: at
df 2e7 its 0.975 point is 1.00061989 where the exact quantile is 1.00087691 (`pf` of R's value is 0.083, not
0.025), so R's `var.test` limits are 2.6e-4 too narrow there. p2a's limits come from statrs'
`FisherSnedecor::inverse_cdf`, which is within 3e-12 of the exact quantile; the case computes that one from
`qbeta` (`F = x / (1 - x)` for `x ~ Beta(df / 2, df / 2)`, key `ci_big_exact_estimate_1`) and records R's
`qf` values next to it (`qf_big`, `ci_big_estimate_1`) without comparing them. This is the entry's second
deviation.

## Conversion rules

| Pattern | Replacement |
|---|---|
| `2 (1 - Phi(|z|))` | `normal_p_value_two_sided(z)` |
| `1 - Phi(z)` | `Normal::standard().sf(z)` (erfc-based) |
| `2 (1 - T_df(|t|))` | `t_test_p_value(t, df)` |
| `1 - ChiSq_df(x)`, statrs `ChiSquared::sf(x)` | `chi_squared_p_value(x, df)` |
| `1 - F_{d1,d2}(f)`, statrs `FisherSnedecor::sf(f)` | `f_test_p_value(f, d1, d2)`; lower tail `f_test_p_value(1/f, d2, d1)` |
| `1 - erf(x)` | `erfc(x)` |
| discrete `1 - cdf(k)` | `DiscreteCDF::sf(k)` |
| `1 - F` where R's function computes the same difference | kept, marked `// numeric-ok: <reason>` |

Tukey HSD's adjusted p-values come from `r_mathlib::tukey_pdf(q, 1, k, df, lower_tail = false, false)`, R's
`ptukey(lower.tail = FALSE)`, the call `TukeyHSD` makes.

## The guard

`crates/p2a-validation/tests/numeric_guard.rs` (run by `cargo test -p p2a-validation`, in CI) fails on a
`1.0 - <cdf call>` in product code. A site where R computes the same difference carries
`// numeric-ok: <reason>` on its line or the line above.
