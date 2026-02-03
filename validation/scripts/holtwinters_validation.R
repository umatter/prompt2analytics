#!/usr/bin/env Rscript
# Holt-Winters Validation Script
# Compares R stats::HoltWinters() results with p2a-core implementation

# ============================================================================
# Test Case 1: AirPassengers - Classic seasonal data (monthly, period=12)
# ============================================================================

cat("=== Holt-Winters Validation ===\n\n")

# Using the classic AirPassengers dataset
data(AirPassengers)
y <- as.numeric(AirPassengers)

cat("Test 1: AirPassengers (n=144, period=12, multiplicative seasonality)\n")
cat("Data: First 10 values:", head(y, 10), "\n\n")

# Fit multiplicative Holt-Winters
hw_mult <- HoltWinters(ts(y, frequency = 12), seasonal = "multiplicative")

cat("R HoltWinters (multiplicative):\n")
cat("  alpha:", round(hw_mult$alpha, 6), "\n")
cat("  beta:", round(hw_mult$beta, 6), "\n")
cat("  gamma:", round(hw_mult$gamma, 6), "\n")
cat("  SSE:", round(hw_mult$SSE, 4), "\n")
cat("  Final level:", round(hw_mult$coefficients["a"], 6), "\n")
cat("  Final trend:", round(hw_mult$coefficients["b"], 6), "\n")

# Seasonal coefficients
seasonal <- hw_mult$coefficients[grep("^s", names(hw_mult$coefficients))]
cat("  Seasonal coefficients:", round(seasonal, 6), "\n\n")

# Fitted values (first 10)
fitted <- hw_mult$fitted[, "xhat"]
cat("  Fitted (first 10):", round(fitted[1:10], 4), "\n\n")

# Save expected values for Rust comparison
write.csv(data.frame(
  alpha = hw_mult$alpha,
  beta = hw_mult$beta,
  gamma = hw_mult$gamma,
  sse = hw_mult$SSE,
  level = hw_mult$coefficients["a"],
  trend = hw_mult$coefficients["b"]
), "validation/expected/holtwinters_airpassengers_mult.csv", row.names = FALSE)

# ============================================================================
# Test Case 2: AirPassengers with additive seasonality
# ============================================================================

cat("Test 2: AirPassengers (additive seasonality)\n")

hw_add <- HoltWinters(ts(y, frequency = 12), seasonal = "additive")

cat("R HoltWinters (additive):\n")
cat("  alpha:", round(hw_add$alpha, 6), "\n")
cat("  beta:", round(hw_add$beta, 6), "\n")
cat("  gamma:", round(hw_add$gamma, 6), "\n")
cat("  SSE:", round(hw_add$SSE, 4), "\n\n")

write.csv(data.frame(
  alpha = hw_add$alpha,
  beta = hw_add$beta,
  gamma = hw_add$gamma,
  sse = hw_add$SSE,
  level = hw_add$coefficients["a"],
  trend = hw_add$coefficients["b"]
), "validation/expected/holtwinters_airpassengers_add.csv", row.names = FALSE)

# ============================================================================
# Test Case 3: Synthetic quarterly data (period=4)
# ============================================================================

cat("Test 3: Synthetic quarterly data (n=40, period=4)\n")

set.seed(42)
n <- 40
period <- 4
t <- 1:n
seasonal_pattern <- c(0.9, 1.0, 1.2, 0.9)  # Quarterly pattern
trend <- 100 + 2 * t  # Linear trend
seasonality <- rep(seasonal_pattern, n / period)
noise <- rnorm(n, sd = 3)
y_synth <- trend * seasonality + noise

cat("Synthetic data: First 12 values:", round(y_synth[1:12], 4), "\n")

hw_synth <- HoltWinters(ts(y_synth, frequency = period), seasonal = "multiplicative")

cat("R HoltWinters result:\n")
cat("  alpha:", round(hw_synth$alpha, 6), "\n")
cat("  beta:", round(hw_synth$beta, 6), "\n")
cat("  gamma:", round(hw_synth$gamma, 6), "\n")
cat("  SSE:", round(hw_synth$SSE, 4), "\n\n")

write.csv(data.frame(
  alpha = hw_synth$alpha,
  beta = hw_synth$beta,
  gamma = hw_synth$gamma,
  sse = hw_synth$SSE,
  level = hw_synth$coefficients["a"],
  trend = hw_synth$coefficients["b"]
), "validation/expected/holtwinters_synthetic_quarterly.csv", row.names = FALSE)

# Also save the synthetic data for Rust tests
write.csv(data.frame(value = y_synth), "validation/expected/holtwinters_synthetic_data.csv", row.names = FALSE)

# ============================================================================
# Test Case 4: Forecasting (h=12 ahead)
# ============================================================================

cat("Test 4: Forecasting 12 periods ahead\n")

fc <- predict(hw_mult, n.ahead = 12)
cat("R forecast (12 periods):", round(fc, 4), "\n\n")

write.csv(data.frame(forecast = as.numeric(fc)),
          "validation/expected/holtwinters_forecast.csv", row.names = FALSE)

# ============================================================================
# Test Case 5: Non-seasonal (simple exponential smoothing with trend)
# ============================================================================

cat("Test 5: Non-seasonal trend model\n")

set.seed(123)
y_ns <- 50 + 0.5 * (1:30) + rnorm(30, sd = 2)

# Non-seasonal Holt-Winters (gamma=FALSE or just HoltWinters on non-seasonal ts)
hw_ns <- HoltWinters(ts(y_ns), gamma = FALSE)

cat("R HoltWinters (non-seasonal):\n")
cat("  alpha:", round(hw_ns$alpha, 6), "\n")
cat("  beta:", round(hw_ns$beta, 6), "\n")
cat("  gamma: NULL (non-seasonal)\n")
cat("  SSE:", round(hw_ns$SSE, 4), "\n\n")

write.csv(data.frame(
  alpha = hw_ns$alpha,
  beta = hw_ns$beta,
  sse = hw_ns$SSE,
  level = hw_ns$coefficients["a"],
  trend = hw_ns$coefficients["b"]
), "validation/expected/holtwinters_nonseasonal.csv", row.names = FALSE)

write.csv(data.frame(value = y_ns), "validation/expected/holtwinters_nonseasonal_data.csv", row.names = FALSE)

# ============================================================================
# Summary
# ============================================================================

cat("=== Validation Summary ===\n")
cat("Generated expected values in validation/expected/:\n")
cat("  - holtwinters_airpassengers_mult.csv\n")
cat("  - holtwinters_airpassengers_add.csv\n")
cat("  - holtwinters_synthetic_quarterly.csv\n")
cat("  - holtwinters_synthetic_data.csv\n")
cat("  - holtwinters_forecast.csv\n")
cat("  - holtwinters_nonseasonal.csv\n")
cat("  - holtwinters_nonseasonal_data.csv\n")
