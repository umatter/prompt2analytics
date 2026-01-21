#!/usr/bin/env Rscript
# Spectral Density Estimation Validation Script
# Compares R stats::spectrum results with p2a-core output

library(stats)

set.seed(42)

# =============================================================================
# Test Case 1: Sine wave with known frequency
# =============================================================================

cat("=== Test Case 1: Sine Wave (frequency = 0.1, period = 10) ===\n")

# Generate sine wave with period 10 (frequency 0.1)
n <- 100
x1 <- sin(2 * pi * (1:n) / 10)

# Raw periodogram (no smoothing)
sp_raw <- spec.pgram(x1, spans = NULL, taper = 0, detrend = TRUE, plot = FALSE)
cat("\nRaw Periodogram:\n")
cat(sprintf("  Number of frequencies: %d\n", length(sp_raw$freq)))
cat(sprintf("  Peak frequency: %.4f\n", sp_raw$freq[which.max(sp_raw$spec)]))
cat(sprintf("  Peak spectral density: %.6f\n", max(sp_raw$spec)))

# Smoothed periodogram with spans = c(3, 3)
sp_smooth <- spec.pgram(x1, spans = c(3, 3), taper = 0.1, detrend = TRUE, plot = FALSE)
cat("\nSmoothed Periodogram (spans = c(3, 3), taper = 0.1):\n")
cat(sprintf("  Number of frequencies: %d\n", length(sp_smooth$freq)))
cat(sprintf("  Peak frequency: %.4f\n", sp_smooth$freq[which.max(sp_smooth$spec)]))
cat(sprintf("  Peak spectral density: %.6f\n", max(sp_smooth$spec)))
cat(sprintf("  Bandwidth: %.6f\n", sp_smooth$bandwidth))
cat(sprintf("  Degrees of freedom: %.2f\n", sp_smooth$df))

# Save first 10 frequency-spectrum pairs for validation
cat("\nFirst 10 frequency-spectrum pairs (smoothed):\n")
for (i in 1:min(10, length(sp_smooth$freq))) {
  cat(sprintf("  freq=%.4f, spec=%.6e\n", sp_smooth$freq[i], sp_smooth$spec[i]))
}

# =============================================================================
# Test Case 2: AR(1) process
# =============================================================================

cat("\n\n=== Test Case 2: AR(1) Process (phi = 0.7) ===\n")

# Generate AR(1) data
n2 <- 200
e <- rnorm(n2)
x2 <- numeric(n2)
phi <- 0.7
x2[1] <- e[1]
for (t in 2:n2) {
  x2[t] <- phi * x2[t-1] + e[t]
}

# AR-based spectrum
sp_ar <- spec.ar(x2, plot = FALSE)
cat(sprintf("\nAR-based spectrum (order selected by AIC = %d):\n", sp_ar$order))
cat(sprintf("  Number of frequencies: %d\n", length(sp_ar$freq)))
cat(sprintf("  Method: %s\n", sp_ar$method))

# Peak for AR(1) should be at low frequencies
cat(sprintf("  Peak frequency: %.4f\n", sp_ar$freq[which.max(sp_ar$spec)]))
cat(sprintf("  Peak spectral density: %.6f\n", max(sp_ar$spec)))

# =============================================================================
# Test Case 3: White noise
# =============================================================================

cat("\n\n=== Test Case 3: White Noise ===\n")

n3 <- 100
x3 <- rnorm(n3)

sp_wn <- spec.pgram(x3, spans = c(5, 5), taper = 0.1, detrend = FALSE, plot = FALSE)
cat(sprintf("\nSmoothed Periodogram (spans = c(5, 5)):\n"))
cat(sprintf("  Mean spectrum: %.6f\n", mean(sp_wn$spec)))
cat(sprintf("  SD spectrum: %.6f\n", sd(sp_wn$spec)))
cat(sprintf("  Coefficient of variation: %.2f\n", sd(sp_wn$spec) / mean(sp_wn$spec)))
cat("  (White noise should have relatively flat spectrum)\n")

# =============================================================================
# Test Case 4: Specific values for validation
# =============================================================================

cat("\n\n=== Test Case 4: Validation Data ===\n")

# Simple deterministic series for exact comparison
x4 <- 1:20

sp4 <- spec.pgram(x4, spans = NULL, taper = 0.1, detrend = TRUE, plot = FALSE)
cat("\nDetrended linear series (1:20):\n")
cat(sprintf("  Total spectral power: %.10f\n", sum(sp4$spec)))
cat("  (Should be near zero after detrending a linear series)\n")

# Series with known spectral content
x5 <- cos(2 * pi * (1:50) / 5) + 0.5 * cos(2 * pi * (1:50) / 10)
sp5 <- spec.pgram(x5, spans = c(3), taper = 0.1, detrend = TRUE, plot = FALSE)

cat("\nTwo-frequency series (f=0.2 and f=0.1):\n")
cat(sprintf("  Frequencies with highest power:\n"))
ord <- order(sp5$spec, decreasing = TRUE)
for (i in 1:5) {
  cat(sprintf("    freq=%.4f, spec=%.6e\n", sp5$freq[ord[i]], sp5$spec[ord[i]]))
}

# =============================================================================
# Save expected values for Rust validation
# =============================================================================

cat("\n\n=== Expected Values for Rust Validation ===\n")

# Test 1: Sine wave
cat("\n# Test 1: Sine wave (n=100, period=10)\n")
cat(sprintf("SINE_PEAK_FREQ = %.6f\n", sp_smooth$freq[which.max(sp_smooth$spec)]))
cat(sprintf("SINE_BANDWIDTH = %.10f\n", sp_smooth$bandwidth))
cat(sprintf("SINE_DF = %.4f\n", sp_smooth$df))

# Test 4: Detrended linear
cat(sprintf("LINEAR_TOTAL_POWER = %.10f\n", sum(sp4$spec)))

cat("\n# All tests completed successfully.\n")
