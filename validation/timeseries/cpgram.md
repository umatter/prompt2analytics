# Cumulative Periodogram Validation

## Method Overview

The cumulative periodogram (`cpgram`) is used for testing whether a time series is white noise. It plots the cumulative sum of the periodogram ordinates against frequency, with confidence bands based on the Kolmogorov-Smirnov distribution.

**Key Parameters:**
- `x`: Time series data
- `taper`: Proportion of data to taper (default: 0.1)

## Reference Implementations

| Package | Function | Notes |
|---------|----------|-------|
| R stats | `cpgram()` | Reference (primarily plotting) |
| R stats | `spectrum()` | Core periodogram computation |

## Algorithm

1. Compute periodogram using FFT: I(f) = |FFT(x)|^2 / n
2. Normalize: cumsum(I) / sum(I)
3. Plot against normalized frequency (0, 0.5)
4. Add 95% confidence bands: y +/- 1.358 / sqrt(n)

## Test Cases

### Test Case 1: White Noise (Should Pass Test)

**R Code:**
```r
set.seed(42)
x <- rnorm(256)
cpgram(x)  # Should stay within bands
# Extract periodogram values
spec <- spectrum(x, plot = FALSE)
cumspec <- cumsum(spec$spec) / sum(spec$spec)
# For white noise, cumspec should be approximately linear
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with TS-22 for the tests and TS-9 for the estimator):
- `forecasting::cpgram::tests::test_cpgram_white_noise` (`crates/p2a-core/src/forecasting/cpgram.rs`): max deviation < 1 on zero-mean data.

### Test Case 2: Non-White Noise (AR Process)

**R Code:**
```r
set.seed(42)
# AR(1) process with strong autocorrelation
x <- arima.sim(n = 256, list(ar = 0.9))
cpgram(x)  # Should deviate from bands at low frequencies
```

**Rust tests**: no AR(1) test exists (not R-pinned; the R-pinned case comes with TS-22).

### Test Case 3: Periodogram Values

**R Code:**
```r
x <- c(1, 2, 3, 4, 5, 6, 7, 8)  # Simple sequence
spec <- spectrum(x, plot = FALSE, detrend = FALSE)
print(spec$spec)
print(cumsum(spec$spec) / sum(spec$spec))
```

**Rust tests** (not R-pinned: no committed R case is compared; the R-pinned case comes with TS-22 for the tests and TS-9 for the estimator):
- `forecasting::cpgram::tests::test_cpgram_basic` (`crates/p2a-core/src/forecasting/cpgram.rs`): n, taper 0.1, equal lengths, a last value within 0.01 of 1 and frequencies in (0, 0.5].
- `forecasting::cpgram::tests::test_cpgram_confidence_bands` (`crates/p2a-core/src/forecasting/cpgram.rs`): 0 ≤ lower ≤ upper ≤ 1.
- `forecasting::cpgram::tests::test_validate_periodogram_against_r` (`crates/p2a-core/src/forecasting/cpgram.rs`), added in Tier 0 (T0-WP18): the periodogram of the deterministic series `mk(16)` (formula in the test; R `Mod(fft(x - mean(x)))^2/16`, ordinates 2-9) within 1e-9 relative of R 4.3.3 values typed into the test.
- `forecasting::cpgram::tests::test_validate_cpgram_untapered_against_r` (`crates/p2a-core/src/forecasting/cpgram.rs`), added in Tier 0 (T0-WP18): the untapered cumulative periodogram of `mk(65)` (the body of `stats::cpgram` with `taper = 0`, 32 values) within 1e-10 relative of R 4.3.3 values typed into the test.

The two Tier 0 tests are quoted-value checks, not R-pinned. The default tapered case is not compared with R (TS-9).

## Numerical Precision Summary

| Series Length | Periodogram Tolerance |
|--------------|----------------------|
| n < 100 | 1e-6 |
| n = 100-1000 | 1e-8 |
| n > 1000 | 1e-10 |

## Known Differences

1. **Tapering**: R uses cosine taper by default; our implementation matches this
2. **Detrending**: R detrends by default; our implementation includes this option
3. **FFT normalization**: Minor differences in scaling conventions

## Performance Notes

- O(n log n) complexity due to FFT
- Rust implementation 5-20x faster than R depending on series length
- Memory efficient for long time series

## References

1. Brockwell, P. J., & Davis, R. A. (1991). Time Series: Theory and Methods. Springer.
2. Priestley, M. B. (1981). Spectral Analysis and Time Series. Academic Press.
3. R Core Team. cpgram() documentation.
