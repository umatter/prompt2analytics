# Validation: Cochran-Mantel-Haenszel Test (mantelhaen.test)

## Method Overview

The Cochran-Mantel-Haenszel (CMH) test evaluates the null hypothesis that two nominal variables are conditionally independent in each stratum, assuming no three-way interaction. It is commonly used in epidemiological studies to control for confounding when analyzing stratified 2×2 tables.

**Key Features:**
- Tests conditional independence in stratified 2×2 tables
- Provides common odds ratio estimate (Mantel-Haenszel estimator)
- Supports Yates' continuity correction
- Returns 95% confidence interval for the odds ratio

**Mathematical Background:**
```
CMH = (|Σ(a_k - E[a_k])| - 0.5)² / Σ Var(a_k)

where:
  E[a_k] = n1_k × m1_k / n_k     (expected count for cell a in stratum k)
  Var(a_k) = n1_k × n2_k × m1_k × m2_k / (n_k² × (n_k - 1))

Common OR (Mantel-Haenszel) = Σ(a_k × d_k / n_k) / Σ(b_k × c_k / n_k)
```

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | mantelhaen.test | R 4.3+ |

## Test Cases

### Test 1: Rabbits Data (R Documentation Example)

**R Code:**
```r
Rabbits <- array(c(
  0, 0, 6, 5,
  3, 0, 3, 6,
  6, 2, 0, 4,
  5, 6, 1, 0,
  2, 5, 0, 0
), dim = c(2, 2, 5),
dimnames = list(
  Delay = c("None", "1.5h"),
  Response = c("Cured", "Died"),
  Penicillin.Level = c("1/8", "1/4", "1/2", "1", "4")
))
mantelhaen.test(Rabbits)

# Result:
# Mantel-Haenszel X-squared = 3.9286, df = 1, p-value = 0.04747
# common odds ratio estimate: 7
# 95 percent confidence interval: 1.026713 47.725133
```

**Results Comparison:**

| Metric | R Value | Rust Value | Tolerance |
|--------|---------|------------|-----------|
| CMH X² | 3.9286 | 3.9286 | 0.1 |
| df | 1 | 1 | exact |
| p-value | 0.04747 | 0.04747 | 0.01 |
| Common OR | 7.0 | 7.0 | 0.5 |
| 95% CI lower | 1.027 | 1.027 | 0.1 |
| 95% CI upper | 47.73 | 47.73 | 1.0 |

**Rust Test:** `crates/p2a-core/src/stats/mantelhaen.rs::tests::test_validate_mantelhaen_against_r`

### Test 2: Single Stratum

**Test:** Single 2×2 table should produce results similar to a standard chi-squared test.

**Results Comparison:**

| Metric | Rust Value | Notes |
|--------|------------|-------|
| CMH statistic | > 0 | Non-zero for associated data |
| df | 1 | Always 1 for 2×2×K |

**Rust Test:** `crates/p2a-core/src/stats/mantelhaen.rs::tests::test_mantelhaen_single_stratum`

## Numerical Precision Summary

| Component | Tolerance | Notes |
|-----------|-----------|-------|
| CMH statistic | 0.1 | Chi-squared with continuity correction |
| p-value | 0.01 | From chi-squared CDF |
| Common OR | 0.5 | Ratio of weighted products |
| CI bounds | 1.0 | Uses Robins et al. variance formula |

## Known Differences

1. **Continuity correction**: Both R and Rust apply Yates' correction when |DELTA| >= 0.5 (default: true).

2. **Odds ratio CI**: Both use Robins, Breslow, Greenland (1986) variance formula for the log odds ratio.

3. **Array layout**: R fills arrays column-major, so input tables need careful indexing:
   - R: c(a, c, b, d) fills to [[a,b], [c,d]]
   - Rust: Table2x2::new(a, b, c, d)

## Performance Comparison

| Number of Strata | Rust (µs) | R (µs) | Speedup |
|------------------|-----------|--------|---------|
| 5 | 0.89 | 360 | ~404x |
| 10 | 1.24 | 340 | ~274x |
| 50 | 6.5 | 860 | ~132x |
| 100 | 12.9 | 1,620 | ~126x |

**Notes:**
- Rust consistently outperforms R by 126-404x depending on problem size
- Speedup is highest for small problems where R overhead dominates
- Rust benchmarks from Criterion (median); R benchmarks from system.time (mean of 100 iterations)

## References

- Cochran, W. G. (1954). "Some Methods for Strengthening the Common χ² Tests". Biometrics, 10(4), 417-451.
- Mantel, N. & Haenszel, W. (1959). "Statistical Aspects of the Analysis of Data from Retrospective Studies of Disease". JNCI, 22(4), 719-748.
- Robins, J., Breslow, N., & Greenland, S. (1986). "Estimators of the Mantel-Haenszel variance consistent in both sparse data and large-strata limiting models". Biometrics, 42(2), 311-323.
- R Core Team. `stats::mantelhaen.test()` function. https://stat.ethz.ch/R-manual/R-devel/library/stats/html/mantelhaen.test.html
