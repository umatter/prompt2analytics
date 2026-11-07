# Validation: Spectral Density Estimation

## Method Overview

Spectral density estimation (spectrum analysis) estimates the power spectrum of a time series, showing how variance is distributed across frequency components. Two methods are implemented:

1. **Periodogram (pgram)**: FFT-based estimation with optional smoothing via modified Daniell kernels
2. **AR-based (ar)**: Fits an AR model and computes its theoretical spectrum

Key parameters:
- `spans`: Vector of odd integers for Daniell kernel smoothing
- `taper`: Proportion of data to taper (reduces spectral leakage)
- `detrend`: Whether to remove linear trend

## Reference Implementations

| Package | Language | Function | Version Tested |
|---------|----------|----------|----------------|
| stats | R | `spectrum`, `spec.pgram`, `spec.ar` | R 4.3.x |

## Test Cases

### Frequency grid, peak and interval multipliers (R-pinned)

Case `stat_spectrum_t3` in `validation/r/cases/stats_misc_t3.R`; data in
`validation/datasets/generated/spectrum_t3.csv` (n = 96, a sine of period 8 plus AR(1) noise).

```r
sp <- spec.pgram(d$x, spans = c(3, 3), taper = 0.1, detrend = TRUE, plot = FALSE)
spec_ci <- function(df, coverage = 0.95) {   # stats:::plot.spec's private spec.ci, verbatim
  tail <- 1 - coverage
  upper.quantile <- 1 - tail * pchisq(df, df, lower.tail = FALSE)
  lower.quantile <- tail * pchisq(df, df)
  1 / (qchisq(c(upper.quantile, lower.quantile), df) / df)
}
```

| Quantity | R | Tolerance |
|---|---|---|
| `freq` | k / 96, k = 1..48 | `Tol::coef(96)` |
| `peak_freq` | 0.125 | `Tol::coef(96)` |
| `spec_ci(df, 0.95)` at df 6.552102376599634 | 0.4161299390667409, 4.203003061009554 | 1e-10 relative |
| `spec_ci(df, 0.95)` at df 2 | 0.2502670177925996, 31.136857826846274 | 1e-10 relative |
| `spec_ci(df, 0.95)` at df 18 | 0.5648042334184131, 2.1531260261391543 | 1e-10 relative |
| `spec_ci(df, 0.95)` at df 7.5 | 0.4362718681455841, 3.721713558236682 | 1e-10 relative |
| `spec_ci(df, 0.90)` at the same df | `ci90_at_df` in the case | 1e-10 relative |

**Rust test:** `stats::spectrum::tests::test_validate_spectrum_t3_grid_and_ci`.

### Confidence interval

`SpectrumResult::confidence_interval_multipliers(level)` is R's `plot.spec` rule: the two tails are
split in proportion to `P(X <= df)` and `P(X > df)` for `X ~ chi^2(df)`, so the multipliers are
`df / qchisq(1 - (1 - level) P(X > df), df)` and `df / qchisq((1 - level) P(X <= df), df)`. The level
is validated. The df that goes in is p2a's own (STAT-39, below).

**AR spectra report no interval.** R's `spec.ar` has no df. `spectrum_ar` sets df to infinity,
`confidence_interval_multipliers` returns `Ok(None)` for it, the older `confidence_multipliers` returns
`(NaN, NaN)`, and Display says "Degrees of Freedom: not defined". Tested by
`stats::spectrum::tests::test_spectrum_ar_no_ci`.

### Properties

These compare no R number: `test_spectrum_sine_peak_property`, `test_raw_periodogram_length_property`,
`test_detrend_linear_has_no_power`, `test_two_frequency_peaks_property`,
`test_white_noise_spectrum_cv_property` (all in `stats::spectrum::tests`).

## Missing values

`run_spectrum` and `run_spectrum_ar` refuse a column with missing (empty) cells, as R's
`spectrum` and `spec.ar` do under `na.action = na.fail`, naming the column and the number of
missing cells (R's `sum(is.na(x))`); the count is pinned with the ACF readers'
([acf.md](../stats/acf.md)).

## Known Differences from R

**STAT-39 (df and bandwidth):** on the case above R gives df 6.552102376599634 and bandwidth
0.01084201041388611; p2a gives 18 and 0.03125. The interval multipliers are pinned at given df, so
they carry p2a's df until STAT-39 is fixed.

**STAT-68 (ordinates):** p2a's spectral ordinates are about twice R's and differ in the taper and in the
smoothing at the edges. The peak (`spec` 7.658276496605042 at frequency 0.125 in R) is at the same
frequency. R's `spec`, `raw_spec`, `raw_df` and `raw_bandwidth` are recorded in the case.

## Performance Comparison

### Raw Periodogram (no smoothing)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 5.0       | 330    | **66x faster** |
| n=1,000      | 40        | 540    | **13x faster** |
| n=10,000     | 396       | 1,690  | **4.3x faster** |
| n=100,000    | 4,840     | 16,830 | **3.5x faster** |

### Smoothed Periodogram (spans=c(3,3), taper=0.1)

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 9.5       | 620    | **65x faster** |
| n=1,000      | 84        | 1,140  | **14x faster** |
| n=10,000     | 850       | 4,840  | **5.7x faster** |
| n=100,000    | 16,430    | 44,620 | **2.7x faster** |

### AR-based Spectrum

| Dataset Size | Rust (µs) | R (µs) | Speedup |
|--------------|-----------|--------|---------|
| n=100        | 194       | 1,560  | **8x faster** |
| n=1,000      | 2,680     | 4,020  | **1.5x faster** |
| n=10,000     | 40,500    | 14,000 | R 2.9x faster |

**Implementation**: Uses `rustfft` for O(n log n) FFT computation, matching R's algorithmic
complexity while eliminating interpreter overhead. Rust is faster than R across all dataset
sizes for the periodogram method.

*Benchmarks run on 2026-01-19. Rust benchmarks via Criterion, R benchmarks via system.time().*

## References

- Priestley, M. B. (1981). *Spectral Analysis and Time Series*. Academic Press.
- Percival, D. B. & Walden, A. T. (1993). *Spectral Analysis for Physical Applications*. Cambridge University Press.
- R Documentation: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/spectrum.html
