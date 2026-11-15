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

### Periodogram ordinates (R-pinned, `stats_misc_t4.json` `spectrum_pgram_t4`)

Case `stat_spectrum_t4` in `validation/r/cases/stats_misc_t4.R`: `spec.pgram` on the n = 96 series
above and on `validation/datasets/generated/spectrum_t4.csv` (97 points, AR(0.6) plus a trend), over
taper 0, 0.1 and 0.25, spans none, 7, (3, 3) and (3, 5), detrending, demeaning only and neither,
`pad = 0.5`, odd n (the first 75 points) and n = 97 with and without `fast` padding. Frequencies,
ordinates and `n.used` are compared (`Tol::coef(96)`, `n.used` exact) by
`test_validate_spectrum_pgram_raw_tapered`, `test_validate_spectrum_pgram_smoothed_taper_demean`,
`test_validate_spectrum_pgram_padding` and `test_validate_spectrum_pgram_odd_and_fast`, and the
`stat_spectrum_t3` ordinates (`spec`, `raw_spec`) by `test_validate_spectrum_t3_grid_and_ci`.

`spectrum` computes the periodogram as `spec.pgram`: R's detrending (centred times); the split
cosine bell of `spec.taper` (`m = floor(n p)` points at each end, weights
`0.5 (1 - cos(pi (2j - 1) / (2m)))`, no renormalisation) with the ordinates divided by
`u2 = 1 - (5/8) 2p`; `trunc(n * pad)` zeros of padding and, with `fast = true` (R's default), more
zeros up to the next length with no prime factor above 5 (`nextn`; 97 points are analysed at
length 100); `|fft|^2 / N0` with `N0` the unpadded length; the zero-frequency ordinate replaced by
the mean of its neighbours; and modified Daniell smoothing that wraps around circularly as
`kernapply(circular = TRUE)` does. Ordinates are on R's scale (no one-sided doubling) at
frequencies `j / N`, `j = 1..floor(N/2)`. The circular smoothing runs over the ordinates
`0..floor(N/2)` with mirrored indices, which equals R's pass over all `N` because the periodogram of
a real series is symmetric; `daniell_mirrored_pass_matches_full_circular_convolution` checks it
against a naive circular convolution, and `test_validate_daniell_interior_against_r` pins all 32
ordinates of two smoothed periodograms, edges included. `nextn_matches_r` checks `nextn(1:200)`
and `padding_truncates_as_r` the `trunc` of the padding.

### Confidence interval

`SpectrumResult::confidence_interval_multipliers(level)` is R's `plot.spec` rule: the two tails are
split in proportion to `P(X <= df)` and `P(X > df)` for `X ~ chi^2(df)`, so the multipliers are
`df / qchisq(1 - (1 - level) P(X > df), df)` and `df / qchisq((1 - level) P(X <= df), df)`. The level
is validated.

### Degrees of freedom and bandwidth

As `spec.pgram`: the combined modified Daniell kernel `k` of the spans gives `df.kernel = 2 / sum(k^2)`
and `bandwidth.kernel = sqrt(sum((1/12 + i^2) k_i))` (2 and `sqrt(1/12)` unsmoothed); the df is divided
by the taper factor `u4 / u2^2`, `u4 = 1 - (93/128) 2 taper`, and scaled by `n / n.used`, the bandwidth
divided by `n.used`. `_df` and `_bw` of every `stat_spectrum_t4` variant and `df`, `bandwidth`,
`raw_df`, `raw_bandwidth` of `stat_spectrum_t3` are pinned (`Tol::se(96)`).

### AR spectrum

`spectrum_ar` evaluates at `spec.ar`'s grid `seq(0, 0.5, length.out = n.freq)` (frequency 0 included)
with `ar.yw`'s `var.pred`, the Levinson innovation variance times `n / (n - (order + 1))` (the AIC
uses the unscaled variance, as `ar.yw`). `test_validate_spectrum_ar_fixed_order` pins frequencies,
ordinates, coefficients and `var.pred` of `spec.ar(x, n.freq = 50, order = 2)`;
`test_validate_spectrum_ar_aic` the AIC order, ordinates and `var.pred` of `spec.ar(x, n.freq = 50)`
(registry `spectrum_ar_t4`).

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


## Timing against the previous implementation (Tier 4)

Release build, scratch harness at f80b0131 and with this change; `--no-default-features --features ci`;
one core (`taskset -c 2`, `RAYON_NUM_THREADS=1`), cwd `/dev/shm`; the two builds interleaved per row with
the order rotated per round, each process the median of its calls after a warm-up (calls calibrated to
about 0.8 s per process); cells are the median over five rounds (nine or ten for rows within 1.05-1.15x),
the ratio of the minima in brackets; 1-minute load average 1.0-3.0.

| Method | n | f80b0131 (µs) | this change (µs) | ratio [min] |
|---|---|---|---|---|
| periodogram, raw | 100 | 4.5 | 4.3 | 0.95 [0.96] |
| periodogram, raw | 1,000 | 35.8 | 33.6 | 0.94 [0.94] |
| periodogram, raw | 10,000 | 471.7 | 515.3 | 1.09 [1.09] |
| periodogram, spans (3, 3) | 100 | 5.6 | 5.4 | 0.95 [0.96] |
| periodogram, spans (3, 3) | 1,000 | 45.8 | 40.9 | 0.89 [0.90] |
| periodogram, spans (3, 3) | 10,000 | 643.4 | 585.9 | 0.91 [0.91] |
| periodogram, spans (7, 9) | 100 | 5.6 | 5.5 | 0.99 [1.00] |
| periodogram, spans (7, 9) | 1,000 | 45.8 | 41.7 | 0.91 [0.91] |
| periodogram, spans (7, 9) | 10,000 | 642.5 | 585.1 | 0.91 [0.91] |
| periodogram, default (fast padding) | 10,007 | 2,683.5 | 494.0 | 0.18 [0.18] |
| AR spectrum, order 2 | 100 | 27.1 | 28.8 | 1.06 [1.06] |
| AR spectrum, order 2 | 1,000 | 41.4 | 42.9 | 1.04 [1.04] |
| AR spectrum, order 2 | 10,000 | 181.1 | 182.8 | 1.01 [1.01] |
| AR spectrum, AIC order, 500 frequencies | 100 | 93.6 | 95.2 | 1.02 [1.02] |
| AR spectrum, AIC order, 500 frequencies | 1,000 | 2,218.9 | 2,220.0 | 1.00 [1.00] |
| AR spectrum, AIC order, 500 frequencies | 10,000 | 39,341.2 | 39,331.1 | 1.00 [1.00] |

At n = 10,007 (a prime) the default now pads the series to a highly composite
length as R's `fast = TRUE` (this tree with `fast = false`: 2.65 ms). Peak RSS of a
spans (3, 3) periodogram at n = 10^6 (one call, whole process): 66.5 MB v0.5.0, 58.6 MB this tree.

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
