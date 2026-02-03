#!/usr/bin/env Rscript
# ACF/PACF/CCF Validation Script
# Compares R stats::acf results with p2a-core output

cat("=== ACF/PACF/CCF Validation Script ===\n\n")

# Test Case 1: Linear trend (simple case)
cat("Test 1: Linear Trend (n=10)\n")
x1 <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

cat("ACF values:\n")
acf_result <- acf(x1, lag.max = 5, plot = FALSE, demean = TRUE)
print(round(acf_result$acf, 6))

cat("\nPACF values:\n")
pacf_result <- pacf(x1, lag.max = 5, plot = FALSE)
print(round(pacf_result$acf, 6))

# Save expected values for Rust comparison
write.csv(
  data.frame(
    lag = 0:5,
    acf = as.vector(acf_result$acf)
  ),
  "validation/expected/acf_test1.csv",
  row.names = FALSE
)

write.csv(
  data.frame(
    lag = 1:5,
    pacf = as.vector(pacf_result$acf)
  ),
  "validation/expected/pacf_test1.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 2: AR(1) simulated data (n=100)
cat("\nTest 2: Simulated AR(1) process (phi=0.7, n=100)\n")
set.seed(42)
n <- 100
e <- rnorm(n, 0, 0.5)
x2 <- numeric(n)
x2[1] <- e[1]
for (t in 2:n) {
  x2[t] <- 0.7 * x2[t-1] + e[t]
}

cat("ACF values (first 10 lags):\n")
acf_ar1 <- acf(x2, lag.max = 10, plot = FALSE)
print(round(acf_ar1$acf, 6))

cat("\nPACF values (first 10 lags):\n")
pacf_ar1 <- pacf(x2, lag.max = 10, plot = FALSE)
print(round(pacf_ar1$acf, 6))

write.csv(
  data.frame(
    lag = 0:10,
    acf = as.vector(acf_ar1$acf)
  ),
  "validation/expected/acf_ar1.csv",
  row.names = FALSE
)

write.csv(
  data.frame(
    lag = 1:10,
    pacf = as.vector(pacf_ar1$acf)
  ),
  "validation/expected/pacf_ar1.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 3: Cross-correlation between two series
cat("\nTest 3: Cross-correlation\n")
x3 <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
y3 <- c(2.0, 4.0, 5.0, 4.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0)

ccf_result <- ccf(x3, y3, lag.max = 3, plot = FALSE)
cat("CCF values:\n")
print(data.frame(
  lag = ccf_result$lag,
  ccf = round(as.vector(ccf_result$acf), 6)
))

write.csv(
  data.frame(
    lag = as.vector(ccf_result$lag),
    ccf = as.vector(ccf_result$acf)
  ),
  "validation/expected/ccf_test1.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 4: Autocovariance (not normalized)
cat("\nTest 4: Autocovariance\n")
acvf_result <- acf(x1, lag.max = 5, type = "covariance", plot = FALSE)
cat("Autocovariance values:\n")
print(round(acvf_result$acf, 6))

write.csv(
  data.frame(
    lag = 0:5,
    acvf = as.vector(acvf_result$acf)
  ),
  "validation/expected/acvf_test1.csv",
  row.names = FALSE
)

cat("\n=== Validation Complete ===\n")
cat("Expected values saved to validation/expected/\n")
