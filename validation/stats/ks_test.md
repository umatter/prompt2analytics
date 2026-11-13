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
| stats | R | `ks.test()` | R 4.3.3 |
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
`stats::ks::tests::test_validate_ks_t3_one_sample` (asymptotic p through `exact: Some(false)`),
together with R's default exact two-sided p (`norm_p_exact` 0.12705346811680995, `unif_p_exact`
0.08193161051708098, `exp_p_exact` 0.028313228455144945).

### Two-sample tests (R-pinned)

```r
ks.test(x5, y5, exact = FALSE)   # 1.2, 1.5, 1.8, 2.1, 2.4 against 2, 2.5, 3, 3.5, 4
ks.test(xa, ya, exact = FALSE)   # n = 60 and 45
```

| Samples | D | D+ | p (greater) | D- | p (less) |
|---|---|---|---|---|---|
| `x5`, `y5` | 0.8 | 0.8 | 0.04076220397836616 | 0 | 1 |
| `xa`, `ya` | 0.23888888888888887 | 0.23888888888888887 | 0.05313493619748799 | 0.07222222222222223 | 0.7647135434753901 |

Pinned by `stats::ks::tests::test_validate_ks_t3_two_sample`, with the two-sided p, exact
(`two5_p_exact` 0.07936507936507936, `twol_p_exact` 0.0922124055286588) and asymptotic
(`two5_p_asym` 0.08151888641220961, `twol_p_asym` 0.10625393011376794). The one-sided exact p of
the same samples (case `stat_ks_t3_exact_t4`) are pinned by
`stats::ks::tests::test_validate_ks_t3_one_sided_exact_t4`.

### Reference parameters

`TheoreticalDistribution::validate` checks the parameters once, before any point is evaluated:
the mean finite; sd and rate finite and above 0; min and max finite with min below max. The error
is `InvalidSpecification` naming the parameter ("KS reference normal: sd must be finite and > 0,
got 0"), tested by `stats::ks::tests::test_ks_rejects_invalid_parameters`. `ks.test` is more
permissive: it evaluates a point mass at `sd = 0` and warns and returns NaN for a negative rate.
p2a refuses these degenerate references (registry deviation STAT-26 on `ks_test_t3`).

`KsDistribution` is the option table for the CLI and MCP surface (`normal`/`pnorm`/`norm`/`gaussian`,
`uniform`/`punif`/`unif`, `exponential`/`pexp`/`exp`), and `TheoreticalDistribution::from_kind`
applies R's defaults (mean 0, sd 1, min 0, max 1, rate 1). A parameter the kind does not take is
an `InvalidSpecification` naming both, as `ks.test` stops with "unused argument":
`stats::ks::tests::test_ks_from_kind_refuses_parameters_of_another_kind`.

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
| p-value | 0.15653867760174539 (exact, conditional on ties) | pinned (`Tol::p_value`) |

R 4.3.3 and p2a report the exact p-value conditional on the ties (R's default for n m < 10000);
`exact: Some(false)` gives the asymptotic 0.24318687396544358.

**Missing values:** `ks_test` and `run_ks_test` read each column through the shared extractor:
the one-sample test and each sample of the two-sample test drop their own missing values (null
or NaN; `ks.test` does the same), report them in `KsTestResult.missing` and in a last Display
line `Missing values: ...`, and accept integer columns. A text column is `NonNumericColumn`.

**Rust tests:**
- `stats::ks::tests::test_validate_ks_test_two_sample_na` (D, p, n, m)
- `stats::ks::tests::test_validate_na_ks_test_missing_rows_and_errors` (one- and two-sample `missing`, Display line, text and absent columns)

**Status:** R-pinned (registry `validation/methods/stats_onesample.json`,
id `ks_test_two_sample`).

## P-values (as R 4.3.3)

`ks_test_one_sample_with`, `ks_test_two_sample_with` and `ks_test_with` take `exact: Option<bool>`;
`None` is R's default: exact for a one-sample test with n < 100 and no ties, and for a two-sample
test with n m < 10000 (with or without ties). A one-sample test with ties uses the asymptotic
distribution with R's warning ("ties should not be present for the Kolmogorov-Smirnov test").

- **One-sample, two-sided, exact:** Marsaglia, Tsang and Wang (2003), as R's `pkolmogorov2x`
  (matrix power with decimal rescaling).
- **One-sample, one-sided, exact:** Birnbaum and Tingey's upper tail, summed directly.
- **Two-sample, exact:** the lattice-path count of R's `psmirnov`, conditional on the tie blocks
  of the pooled sample and accumulated as an upper tail, so small p-values keep their digits.
- **Asymptotic:** the limiting Kolmogorov distribution with R's series and tolerance (1e-6)
  for two-sided tests (no Stephens factor and no cut-off at `sqrt(n) D >= 3`), and
  `exp(-2 n D²)` (one-sample) or `exp(-2 m n/(m + n) D²)` (two-sample) one-sided.

Every buffer goes through `resource::try_*`, and the matrix power and the lattice recursion check
the call's budget once per step.

Case `stat_rank_ks_t4` (registry `validation/methods/stats_rank_t4.json`, id `ks_t4`) pins exact
one-sample p at n = 30 and 99 (`o30`, `o99`, `o99b`, `e99`, every alternative), exact two-sample p
with and without ties (`t57`, `t75`), and asymptotic p at n = 100, 150, 400 and with ties
(`a100`, `a150`, `a400`, `o20t`): `test_validate_ks_one_sample_exact_t4`,
`test_validate_ks_two_sample_exact_t4`, `test_validate_ks_asymptotic_t4`. The exact two-sample
tail is also checked against full enumeration (`test_psmirnov_exact_matches_enumeration`).

## Known Differences

**STAT-27, alternative `less` with ties (two-sample exact; registry `ks_t4`).** R evaluates the
distribution of D+ for both one-sided alternatives. With ties D- has a different distribution, so
p2a computes P(D- >= d) for `less` (`t75_p_less_exact`, which the case checks by full
enumeration); R's value `t75_p_less_r` is kept out of the `Check`. Without ties both agree.

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
(`statistic_D`, `p_value`, `n`, `n_2` against D, p, n and m above, `Tol::stat(38)`,
`Tol::p_value(38)` and exact; `missing`). Registry `validation/methods/mcp_stats.json`, id
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
