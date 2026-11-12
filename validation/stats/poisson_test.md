# Validation: Exact Poisson Test (poisson.test)

## Method Overview

The exact Poisson test evaluates hypotheses about the rate parameter of a Poisson distribution. It supports both one-sample tests (comparing a rate to a hypothesized value) and two-sample tests (comparing the ratio of two rates).

**Key Features:**
- Exact test (no asymptotic approximation)
- One-sample: Tests H₀: λ = r
- Two-sample: Tests H₀: λ₁/λ₂ = r using conditional binomial distribution
- Confidence intervals using chi-squared relationship (one-sample) or exact conditional method (two-sample)

**Mathematical Background:**

For one-sample tests, given X ~ Poisson(λT):
- Test statistic is X itself
- P-value computed exactly from Poisson distribution; the two-sided p-value is R's: the sum of
  P(X = k) over the counts with P(X = k) <= P(X = x) (1 + 1e-7)
- CI derived from chi-squared distribution relationship; one-sided for a one-sided alternative
  (`(0, upper)` for less, `(lower, Inf)` for greater, each at `conf_level`)

For two-sample tests, given X₁ ~ Poisson(λ₁T₁) and X₂ ~ Poisson(λ₂T₂):
- Under H₀: λ₁/λ₂ = r, conditioned on X₁ + X₂ = n:
- X₁ | n ~ Binomial(n, p) where p = rT₁/(rT₁ + T₂)
- P-value from binomial distribution (two-sided: `binom.test`'s rule); the interval is the
  binomial interval mapped to the rate ratio, one-sided for a one-sided alternative

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | poisson.test | R 4.3+ |

## Test Cases

All values are from case `stat_poisson_t3` in `validation/r/cases/stats_misc_t3.R`, pinned by
`stats::poissontest::tests::test_validate_poisson_test_t3` (registry
`validation/methods/stats_misc_t3.json`, id `poisson_test_t3`).

### One sample

| Call | Quantity | R |
|---|---|---|
| `poisson.test(137, 24.19893)` | estimate | 5.661407343217241 |
| | expected count | 24.19893 |
| | 95% CI | 4.753124804848748, 6.692709334176071 |
| `poisson.test(10, 5)` | 95% CI | 0.9590777392264869, 3.678071208403556 |
| | p, two-sided | 0.03856600430529028 |
| | p, less | 0.986304731401617 |
| | 95% CI, less | 0, 3.392443847144379 |
| | p, greater | 0.03182805730620481 |
| | 95% CI, greater | 1.085081139418259, Inf |
| `poisson.test(137, 24.19893)` | p, two-sided | 2.8452272641144545e-56 |
| `poisson.test(2, 1, r = 5)` | p, two-sided | 0.25802369355308846 |
| `poisson.test(150, 1)` | 95% CI | 126.95616130124485, 176.01716609707458 |
| `poisson.test(1e6, 1)` | 95% CI | 998040.9833402939, 1001961.9119454322 |

The intervals take their chi-square quantiles from `chi_squared_quantile`. Statrs' `Gamma::inverse_cdf`
gave (4.744084, 6.681476) for the first call and (126.7548828, 175.6572266) for `x = 150`; shapes from
125 to 160 were affected (STAT-60, closed).

### Two samples

| Call | Quantity | R |
|---|---|---|
| `poisson.test(c(11, 21), c(800, 3011))` | rate ratio | 1.9714880952380953 |
| | expected count | 6.717397008659145 |
| | 95% CI | 0.8584264033916392, 4.277265943654071 |
| | p, two-sided | 0.07966863303332952 |
| | p, less | 0.975931823227312 |
| | 95% CI, less | 0, 3.827386407850879 |
| | p, greater | 0.05600865978769969 |
| | 95% CI, greater | 0.9775843474967454, Inf |
| `poisson.test(c(11, 21), c(800, 3011), r = 2)` | p, two-sided | 1 |
| `poisson.test(c(3, 10), c(1, 1))` (symmetric null) | p | 0.09228515625000003 |
| | 95% CI | 0.05305399048476324, 1.1651186104840492 |

### Two-sided minlike p and one-sided intervals (Tier 4)

Case `stat_poisson_minlike_t4` (`validation/r/cases/stats_exact_t4.R`), pinned by
`stats::poissontest::tests::test_validate_poisson_minlike_t4` (registry
`validation/methods/stats_exact_t4.json`, id `poisson_minlike_t4`): the one-sample two-sided p at
eight (x, T) cells, among them x = 0, x = rT and counts of 1,000 and 10,000; the two-sample
two-sided p and interval with a zero count; and the one-sided p and interval at 0.95 and 0.90 for
one and two samples, e.g. `poisson.test(10, 2, r = 3, alternative = "greater")` (2.7127028485456472,
Inf). An infinite bound is stored as NA in the case and compared as such.

When P(X = x) underflows to 0 for x below rT, the opposite tail's probabilities underflow too
(their total is below 1e-300) and the p-value is P(X <= x) alone. A two-sided one-sample test whose
opposite tail would pass the largest supported count (2^52) is refused.

### Confidence levels and alternatives

A `conf_level` outside (0, 1) is refused (see [hypothesis_tests.md](hypothesis_tests.md)):
`stats::poissontest::tests::test_poisson_refuses_non_finite_parameters`. One-sided p-values have the
expected direction: `stats::poissontest::tests::test_poisson_alternatives`.

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| Estimates, expected counts | `Tol::coef(n)` | n = the total count |
| Intervals | `Tol::stat(n)` | |
| p-values | `Tol::p_value(n)` | |

## Known Differences

STAT-58 (two-sided p-value) and STAT-59 (one-sided intervals) were fixed in Tier 4; their values
are now pinned (above).

1. **Expected count at r != 1 (observation, no finding).** R's `parameter` for a two-sample test at `r = 2` is 7.505130460275579 (`rr_r2_r_parameter`, its own formula); p2a reports the null binomial's mean, 11.103882021253524 (`rr_r2_binomial_mean`). Not compared.

## Performance Comparison

| Test Type | Size | Rust (µs) | R (µs) | Speedup |
|-----------|------|-----------|--------|---------|
| One-sample | x=10, t=1 | 0.55 | 90 | ~164x |
| One-sample | x=100, t=10 | 0.60 | 100 | ~167x |
| One-sample | x=1000, t=100 | 0.64 | 110 | ~172x |
| One-sample | x=10000, t=1000 | 0.73 | 320 | ~438x |
| Two-sample | x=[5,10] | 51.4 | 150 | ~2.9x |
| Two-sample | x=[50,100] | 85.1 | 170 | ~2.0x |
| Two-sample | x=[500,1000] | 131.7 | 140 | ~1.1x |

**Notes:**
- One-sample tests are extremely fast in Rust (sub-microsecond), with 160-440x speedup
- Two-sample tests involve iterative CI calculation, making them slower but still faster than R
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)

## References

- Przyborowski, J. & Wilenski, H. (1940). "Homogeneity of Results in Testing Samples from Poisson Series". Biometrika, 31(3/4), 313-323.
- R Core Team. `stats::poisson.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/poisson.test.html
