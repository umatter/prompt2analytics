# Validation: Test of Proportions (prop.test)

## Method Overview

`prop_test_one`, `prop_test_two` and `prop_test_k` (`crates/p2a-core/src/stats/proptest.rs`) are the
chi-square tests of `stats::prop.test`, with Yates' correction where the call asks for it. They are
library functions: no MCP tool or CLI command reaches them.

A `conf_level` outside (0, 1) is refused (see [hypothesis_tests.md](hypothesis_tests.md)):
`stats::proptest::tests::test_prop_test_rejects_invalid_level`.

## Test Cases

Case `stat_prop_t3` in `validation/r/cases/stats_misc_t3.R`; `prop.test(15, 100, p = 0.1)`, `prop.test(c(30, 40), c(100, 100))`,
and `prop.test(c(30, 40, 35), c(100, 100, 100))`.

| Quantity | R |
|---|---|
| one sample, with correction: X-squared, p (two-sided) | 2.25, 0.13361440253771517 |
| one sample, without: X-squared, p | 2.7777777777777777, 0.09558070454562881 |
| one sample, without: Wilson CI | 0.09305985283656631, 0.2328355959083236 |
| two samples, with correction: X-squared, p | 1.7802197802197803, 0.1821223410094948 |
| two samples, without: X-squared, p | 2.197802197802198, 0.13820766697402617 |
| k samples: X-squared, df, p | 2.197802197802198, 2, 0.33323707715622375 |

The p-values for `less` and `greater`, and the estimates, are in the case too.

**Rust test:** `stats::proptest::tests::test_validate_prop_test_t3`. Tolerance: `Tol::stat(100)`,
`Tol::p_value(100)` (k samples: n = 300), estimates `Tol::coef(100)`, df exact.

**Status:** R-pinned with deviations (registry `validation/methods/stats_misc_t3.json`, id
`prop_test_t3`).

## Known Differences

- **STAT-63:** with the correction, R's one-sample interval is the continuity-corrected Wilson
  interval; p2a's is the uncorrected Wilson interval. R's two-sample interval is Wald with Yates;
  p2a's is Newcombe's method 10. For the two-sample data above p2a gives -0.22717225103721761,
  0.031890373925969467 where R gives -0.24147838108648725, 0.04147838108648719. Not compared.
- **STAT-59:** for a one-sided alternative R's two-sample interval is one-sided; p2a's is two-sided.
  Not compared.
