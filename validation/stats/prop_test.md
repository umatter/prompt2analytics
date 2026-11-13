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
| one sample, with: corrected Wilson CI | 0.08914909572793359, 0.2385307679519066 |
| two samples, with correction: X-squared, p | 1.7802197802197803, 0.1821223410094948 |
| two samples, with: CI (Wald with Yates) | -0.24147838108648725, 0.04147838108648719 |
| two samples, with: CI, less | -1, 0.020340135687017077 |
| two samples, without: X-squared, p | 2.197802197802198, 0.13820766697402617 |
| k samples: X-squared, df, p | 2.197802197802198, 2, 0.33323707715622375 |

The p-values and intervals for `less` and `greater`, with and without the correction, and the
estimates, are in the case too.

Case `stat_prop_ci_t4` (`validation/r/cases/stats_exact_t4.R`; registry
`validation/methods/stats_exact_t4.json`, id `prop_ci_t4`) adds the intervals where the correction
is capped (|x - n p| < 0.5, |DELTA| below half of sum(1/n)), at the boundaries 0 and 1, and at the
0.90 level, for every alternative: `stats::proptest::tests::test_validate_prop_ci_t4`, tolerances
at n the total trials of each cell. For example `prop.test(c(15, 25), c(50, 50))`: two-sided
(-0.40799314123229585, 0.00799314123229583), less (-1, -0.02223118249261749).

The intervals are R's: one sample, the Wilson interval at p ± YATES / n with
YATES = min(0.5, |x - n p|); two samples, the Wald interval widened by YATES sum(1/n) with
YATES = min(0.5, |DELTA| / sum(1/n)); one-sided with `qnorm(conf_level)`.

**Rust test:** `stats::proptest::tests::test_validate_prop_test_t3`. Tolerance: `Tol::stat(100)`,
`Tol::p_value(100)` (k samples: n = 300), estimates `Tol::coef(100)`, df exact.

**Status:** R-pinned (registry `validation/methods/stats_misc_t3.json`, id `prop_test_t3`; and
`validation/methods/stats_exact_t4.json`, id `prop_ci_t4`).

## Wall-clock (signed off 2026-11-08)

Two-sample `prop_test_two`, medians of the round medians over five interleaved rounds against `ff8e7559`,
`release`, one P core, `RAYON_NUM_THREADS=1`, on a quiet machine (load average 1.1-2.0, 2026-11-08): 0.236 µs in
the base and 0.269 µs here at n = 100 (1.14x, about 33 ns per call: the `conf_level` check and the continuity rule),
1.04x at n = 1,000 and n = 10,000. The one-sample test is within 1.09x. Signed off by the user.

## Known Differences

None. STAT-63 (the corrected one-sample interval and the two-sample Wald interval with Yates,
where p2a had used the uncorrected Wilson interval and Newcombe's method 10) and STAT-59 (one-sided
two-sample intervals) were fixed in Tier 4.
