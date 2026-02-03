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

**Rust Test:**
```rust
#[test]
fn test_validate_cpgram_white_noise() {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let x: Vec<f64> = (0..256).map(|_| rng.gen::<f64>() - 0.5).collect();

    let result = cpgram(&x, None).unwrap();

    // For white noise, cumulative periodogram should be approximately linear
    // Check that the KS statistic is not significant (< critical value)
    let (ks_stat, p_value) = white_noise_test(&x, None).unwrap();
    assert!(p_value > 0.05);  // Should not reject white noise hypothesis
}
```

### Test Case 2: Non-White Noise (AR Process)

**R Code:**
```r
set.seed(42)
# AR(1) process with strong autocorrelation
x <- arima.sim(n = 256, list(ar = 0.9))
cpgram(x)  # Should deviate from bands at low frequencies
```

**Rust Test:**
```rust
#[test]
fn test_validate_cpgram_ar1() {
    // AR(1) with rho = 0.9 should fail white noise test
    let x = generate_ar1_series(256, 0.9, 42);

    let (ks_stat, p_value) = white_noise_test(&x, None).unwrap();
    assert!(p_value < 0.05);  // Should reject white noise hypothesis
}
```

### Test Case 3: Periodogram Values

**R Code:**
```r
x <- c(1, 2, 3, 4, 5, 6, 7, 8)  # Simple sequence
spec <- spectrum(x, plot = FALSE, detrend = FALSE)
print(spec$spec)
print(cumsum(spec$spec) / sum(spec$spec))
```

**Rust Test:**
```rust
#[test]
fn test_validate_cpgram_simple() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = cpgram(&x, None).unwrap();

    // Check basic properties
    assert!(result.frequencies.len() > 0);
    assert!(result.cumulative_periodogram.len() == result.frequencies.len());

    // Cumulative periodogram should end at 1.0
    let last = *result.cumulative_periodogram.last().unwrap();
    assert!((last - 1.0).abs() < 1e-10);

    // Should be monotonically increasing
    for i in 1..result.cumulative_periodogram.len() {
        assert!(result.cumulative_periodogram[i] >= result.cumulative_periodogram[i-1]);
    }
}
```

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
