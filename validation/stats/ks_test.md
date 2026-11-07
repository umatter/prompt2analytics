# Validation: Kolmogorov-Smirnov Test

## Method Overview

The Kolmogorov-Smirnov (KS) test is a nonparametric test of the equality of continuous probability distributions. It can be used to:
1. Compare a sample with a reference probability distribution (one-sample test)
2. Compare two samples to test whether they come from the same distribution (two-sample test)

**Key Parameters:**
- `x`: First sample (numeric vector)
- `y`: Second sample or theoretical distribution
- `alternative`: "two.sided" (default), "greater", or "less"

**Test Statistic:**
```
D = sup_x |F_n(x) - G_m(x)|  (two-sample)
D = sup_x |F_n(x) - F_0(x)|  (one-sample)
```

Where F_n and G_m are empirical CDFs, and F_0 is the theoretical CDF.

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `ks.test()` | R 4.3.2 |
| scipy.stats | Python | `ks_2samp()`, `kstest()` | SciPy 1.11 |

## Test Cases

### One-sample tests against distributions with parameters (R-pinned)

Case `stat_ks_t3` in `validation/r/cases/stats_misc_t3.R`; data in
`validation/datasets/generated/ks_t3.csv`. No ties in any sample.

```r
ks.test(xn, "pnorm", 2, 2, exact = FALSE)     # also alternative = "greater" and "less"
ks.test(xu, "punif", -1.5, 3, exact = FALSE)
ks.test(xe, "pexp", 0.8, exact = FALSE)
```

| Reference | n | D | p (asymptotic) | D+ | p (greater) | D- | p (less) |
|---|---|---|---|---|---|---|---|
| `pnorm(2, 2)` | 30 | 0.20855369992548178 | 0.14705778828807003 | 0.012858791667085596 | 0.9901281385125487 | 0.20855369992548178 | 0.07355817088636814 |
| `punif(-1.5, 3)` | 25 | 0.24555555555555558 | 0.09809220328878654 | 0.15522222222222223 | 0.29978284585485815 | 0.24555555555555558 | 0.049051890902218225 |
| `pexp(0.8)` | 40 | 0.2258535956589462 | 0.03378814171051292 | 0.02180511454303294 | 0.9626772811058978 | 0.2258535956589462 | 0.016894152315484544 |

Pinned with `Tol::stat(n)` and `Tol::p_value(n)`, and n exactly, by
`stats::ks::tests::test_validate_ks_t3_one_sample`.

### Two-sample tests (R-pinned)

```r
ks.test(x5, y5, exact = FALSE)   # 1.2, 1.5, 1.8, 2.1, 2.4 against 2, 2.5, 3, 3.5, 4
ks.test(xa, ya, exact = FALSE)   # n = 60 and 45
```

| Samples | D | D+ | p (greater) | D- | p (less) |
|---|---|---|---|---|---|
| `x5`, `y5` | 0.8 | 0.8 | 0.04076220397836616 | 0 | 1 |
| `xa`, `ya` | 0.23888888888888887 | 0.23888888888888887 | 0.05313493619748799 | 0.07222222222222223 | 0.7647135434753901 |

Pinned by `stats::ks::tests::test_validate_ks_t3_two_sample`. The two-sided p-values are a known
deviation (below).

### Reference parameters

`TheoreticalDistribution::validate` checks the parameters once, before any point is evaluated:
the mean finite; sd and rate finite and above 0; min and max finite with min below max. The error
is `InvalidSpecification` naming the parameter ("KS reference normal: sd must be finite and > 0,
got 0"), tested by `stats::ks::tests::test_ks_rejects_invalid_parameters`. `ks.test` is more
permissive: it evaluates a point mass at `sd = 0` and warns and returns NaN for a negative rate.
p2a refuses these degenerate references (registry deviation STAT-26 on `ks_test_t3`).

`KsDistribution` is the option table for the CLI and MCP surface (`normal`/`pnorm`/`norm`/`gaussian`,
`uniform`/`punif`/`unif`, `exponential`/`pexp`/`exp`), and `TheoreticalDistribution::from_kind`
applies R's defaults (mean 0, sd 1, min 0, max 1, rate 1).

## Two-sample test with missing values and integer columns

**R Code** (case `stat_two_sample_na` in `validation/r/cases/stats_onesample.R`; `xi`, `yi` are
Int64 columns of `validation/datasets/generated/two_sample_na.csv` with 20 and 18 values and ties):
```r
ks2 <- suppressWarnings(ks.test(d$xi, d$yi))   # each sample drops its own NA
```

| Quantity | R | Status |
|---|---|---|
| D | 0.3333333333333333 | pinned (`Tol::stat`) |
| n, m | 20, 18 | pinned (exact) |
| p-value | 0.15653867760174539 (exact, conditional on ties) | **known deviation STAT-27** |

**Known deviation (STAT-27):** p2a computes the asymptotic Kolmogorov p-value with Stephens'
correction (0.19563779265722 on this case); R 4.3.3 reports the exact p-value conditional on
the ties. For comparison, R's `ks.test(..., exact = FALSE)` gives 0.24318687396544358. The
p-value is left out of the test's `Check` until STAT-27 is fixed.

**Missing values:** `ks_test` and `run_ks_test` read each column through the shared extractor:
the one-sample test and each sample of the two-sample test drop their own missing values (null
or NaN; `ks.test` does the same), report them in `KsTestResult.missing` and in a last Display
line `Missing values: ...`, and accept integer columns. A text column is `NonNumericColumn`.

**Rust tests:**
- `stats::ks::tests::test_validate_ks_test_two_sample_na` (D, n, m)
- `stats::ks::tests::test_validate_na_ks_test_missing_rows_and_errors` (one- and two-sample `missing`, Display line, text and absent columns)

**Status:** R-pinned with a known deviation (registry `validation/methods/stats_onesample.json`,
id `ks_test_two_sample`).

## Known Differences

**STAT-27 (registry `ks_test_t3`):** p2a has no exact distribution. It reports an asymptotic p-value
where R's default is exact, applies Stephens' (1970) factor to the two-sided two-sample p-value and
cuts the series off at `sqrt(n) D >= 3` (p = 0, R about 3e-8). R's values are recorded in the case
for the method that replaces this:

| Case key | R | p2a |
|---|---|---|
| `norm_p_exact` | 0.12705346811680995 | 0.14705778828807003 (asymptotic, equal to `norm_p_asym`) |
| `unif_p_exact` | 0.08193161051708098 | 0.09809220328878654 |
| `exp_p_exact` | 0.028313228455144945 | 0.03378814171051292 |
| `two5_p_exact` / `two5_p_asym` | 0.07936507936507936 / 0.08151888641220961 | 0.036146190769284631 |
| `twol_p_exact` / `twol_p_asym` | 0.0922124055286588 / 0.10625393011376794 | 0.08997965487690876 |

**Direction of the one-sided tests (R's convention):** "greater" means the CDF of x lies above that
of y, so x tends to be smaller; "less" means the CDF of x lies below that of y, so x tends to be
larger.

**Ties:** two-sample ties are grouped by exact equality, as R's `unique` does, and D is computed
after all tied observations are processed.

## Performance Comparison

### Two-Sample KS Test

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100 | 7.8 | 220 | ~28x |
| n=1,000 | 88 | 400 | ~5x |
| n=10,000 | 1,700 | 3,840 | ~2x |
| n=100,000 | 26,300 | 30,900 | ~1.2x |

### One-Sample KS Test (vs Normal)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100 | 4.5 | 180 | ~40x |
| n=1,000 | 52 | 260 | ~5x |
| n=10,000 | 582 | 2,020 | ~3.5x |
| n=100,000 | 7,000 | 17,300 | ~2.5x |

*Benchmarks run on Linux with Rust Criterion (100 samples) and R microbenchmark/system.time (50 iterations). Speedup decreases at large n due to O(n log n) sorting dominating both implementations.*

## MCP tool: missing values and integer columns

`hypothesis_ks_test` calls the dataset wrapper `ks_test`, so the tool drops each sample's own
missing values, accepts integer columns and reports the drop under `missing` (for a two-sample
test the summaries of the two samples are summed: `n_total` counts the rows of both columns, and
`by_column` is `{"xi": 2, "yi": 4}` on the case above). Before, the handler read its columns
with `into_no_null_iter` (a missing cell entered the sample as 0) and refused Int64 columns
("Column 'xi' is not numeric.").

**MCP test:** `tools::handlers::hypothesis::tests::test_validate_mcp_hypothesis_ks_test_two_sample_na`
(`statistic_D`, `n`, `n_2` against D, n and m above, `Tol::stat(38)` and exact; `missing`). The
p-value is not pinned (STAT-27, as above). Registry `validation/methods/mcp_stats.json`, id
`mcp_hypothesis_ks_test`.

## MCP Tool Usage

```json
{
  "tool": "hypothesis_ks_test",
  "dataset": "my_data",
  "x": "sample1",
  "y": "sample2",
  "alternative": "two.sided"
}
```

For one-sample test against normal:
```json
{
  "tool": "hypothesis_ks_test",
  "dataset": "my_data",
  "x": "sample1",
  "distribution": "normal",
  "mean": 0,
  "sd": 1
}
```

## References

- Kolmogorov, A. N. (1933). "Sulla determinazione empirica di una legge di distribuzione". *Giornale dell'Istituto Italiano degli Attuari*, 4, 83-91.
- Smirnov, N. V. (1939). "On the estimation of the discrepancy between empirical curves of distribution for two independent samples". *Bulletin of Moscow University*, 2(2), 3-16.
- Marsaglia, G., Tsang, W. W., & Wang, J. (2003). "Evaluating Kolmogorov's distribution". *Journal of Statistical Software*, 8(18), 1-4.
- Stephens, M. A. (1970). "Use of the Kolmogorov-Smirnov, Cramer-Von Mises and Related Statistics Without Extensive Tables". *JRSS B*, 32(1), 115-122.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ks.test.html
