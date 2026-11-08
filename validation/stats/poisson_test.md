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
- P-value computed exactly from Poisson distribution
- CI derived from chi-squared distribution relationship

For two-sample tests, given X₁ ~ Poisson(λ₁T₁) and X₂ ~ Poisson(λ₂T₂):
- Under H₀: λ₁/λ₂ = r, conditioned on X₁ + X₂ = n:
- X₁ | n ~ Binomial(n, p) where p = rT₁/(rT₁ + T₂)
- P-value from binomial distribution

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
| | p, less | 0.986304731401617 |
| | p, greater | 0.03182805730620481 |
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
| | p, less | 0.975931823227312 |
| | p, greater | 0.05600865978769969 |
| `poisson.test(c(3, 10), c(1, 1))` (symmetric null) | p | 0.09228515625000003 |
| | 95% CI | 0.05305399048476324, 1.1651186104840492 |

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

1. **STAT-58: two-sided p-value.** R sums the outcomes no more likely than the observed one; p2a doubles the smaller tail. They agree only for a symmetric null (`sym_p`). R's values: `big_p` 2.8452272641144545e-56, `x10_two_p` 0.03856600430529028, `x2_r5_p` 0.25802369355308846, `rr_two_p` 0.07966863303332952. Not compared.

2. **STAT-59: one-sided intervals.** p2a reports the two-sided interval for every alternative; R reports a one-sided one. R's values are `x10_less_ci`, `x10_greater_ci`, `rr_less_ci` and `rr_greater_ci` in the case. Not compared.

3. **Expected count at r != 1 (observation, no finding).** R's `parameter` for a two-sample test at `r = 2` is 7.505130460275579 (`rr_r2_r_parameter`, its own formula); p2a reports the null binomial's mean, 11.103882021253524 (`rr_r2_binomial_mean`). Not compared.

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
