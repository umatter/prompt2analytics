#!/usr/bin/env Rscript
# Box-Pierce and Ljung-Box Tests Validation Script
# Compares R Box.test results with p2a-core Rust output
#
# References:
# - Box, G. E. P. & Pierce, D. A. (1970). JASA, 65, 1509-1526.
# - Ljung, G. M. & Box, G. E. P. (1978). Biometrika, 65, 297-303.

cat("=== Box-Pierce and Ljung-Box Tests Validation ===\n\n")

# Test Case 1: Simple linear trend
cat("Test 1: Linear Trend (x = 1:10)\n")
cat("-" , rep("-", 50), "\n", sep="")
x1 <- 1:10

# Ljung-Box
lb1 <- Box.test(x1, lag = 5, type = "Ljung-Box")
cat("Ljung-Box:\n")
cat(sprintf("  X-squared = %.4f\n", lb1$statistic))
cat(sprintf("  df = %d\n", lb1$parameter))
cat(sprintf("  p-value = %.6f\n", lb1$p.value))

# Box-Pierce
bp1 <- Box.test(x1, lag = 5, type = "Box-Pierce")
cat("Box-Pierce:\n")
cat(sprintf("  X-squared = %.4f\n", bp1$statistic))
cat(sprintf("  df = %d\n", bp1$parameter))
cat(sprintf("  p-value = %.6f\n", bp1$p.value))

# Test Case 2: Longer linear trend
cat("\nTest 2: Linear Trend (x = 1:30, lag=10)\n")
cat("-" , rep("-", 50), "\n", sep="")
x2 <- 1:30

lb2 <- Box.test(x2, lag = 10, type = "Ljung-Box")
cat("Ljung-Box:\n")
cat(sprintf("  X-squared = %.4f\n", lb2$statistic))
cat(sprintf("  df = %d\n", lb2$parameter))
cat(sprintf("  p-value = %.2e\n", lb2$p.value))

# Test Case 3: With fitdf adjustment
cat("\nTest 3: Ljung-Box with fitdf=2 (x = 1:10, lag=5)\n")
cat("-" , rep("-", 50), "\n", sep="")
lb3 <- Box.test(x1, lag = 5, type = "Ljung-Box", fitdf = 2)
cat(sprintf("  X-squared = %.4f\n", lb3$statistic))
cat(sprintf("  df = %d\n", lb3$parameter))
cat(sprintf("  p-value = %.6f\n", lb3$p.value))

# Test Case 4: White noise (pseudo-random)
cat("\nTest 4: White Noise Approximation (set.seed(42), rnorm(50), lag=10)\n")
cat("-" , rep("-", 50), "\n", sep="")
set.seed(42)
x4 <- rnorm(50)

lb4 <- Box.test(x4, lag = 10, type = "Ljung-Box")
cat("Ljung-Box:\n")
cat(sprintf("  X-squared = %.4f\n", lb4$statistic))
cat(sprintf("  df = %d\n", lb4$parameter))
cat(sprintf("  p-value = %.6f\n", lb4$p.value))

bp4 <- Box.test(x4, lag = 10, type = "Box-Pierce")
cat("Box-Pierce:\n")
cat(sprintf("  X-squared = %.4f\n", bp4$statistic))
cat(sprintf("  df = %d\n", bp4$parameter))
cat(sprintf("  p-value = %.6f\n", bp4$p.value))

# Test Case 5: AR(1) process (highly autocorrelated)
cat("\nTest 5: AR(1) Process (phi=0.9, n=100, lag=10)\n")
cat("-" , rep("-", 50), "\n", sep="")
set.seed(123)
n <- 100
phi <- 0.9
ar1 <- numeric(n)
ar1[1] <- rnorm(1)
for (i in 2:n) {
  ar1[i] <- phi * ar1[i-1] + rnorm(1)
}

lb5 <- Box.test(ar1, lag = 10, type = "Ljung-Box")
cat("Ljung-Box:\n")
cat(sprintf("  X-squared = %.4f\n", lb5$statistic))
cat(sprintf("  df = %d\n", lb5$parameter))
cat(sprintf("  p-value = %.2e\n", lb5$p.value))

# Save expected results for Rust comparison
cat("\n=== Expected Values for Rust Tests ===\n")
cat("Test 1 (x=1:10, lag=5):\n")
cat(sprintf("  Ljung-Box: X² = %.6f, df = %d, p = %.6f\n",
            lb1$statistic, lb1$parameter, lb1$p.value))
cat(sprintf("  Box-Pierce: X² = %.6f, df = %d, p = %.6f\n",
            bp1$statistic, bp1$parameter, bp1$p.value))
cat(sprintf("  Ljung-Box (fitdf=2): X² = %.6f, df = %d, p = %.6f\n",
            lb3$statistic, lb3$parameter, lb3$p.value))

cat("\nTest 2 (x=1:30, lag=10):\n")
cat(sprintf("  Ljung-Box: X² = %.4f, df = %d, p < 2.2e-16\n",
            lb2$statistic, lb2$parameter))
