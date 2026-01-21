#!/usr/bin/env Rscript
# LOESS Validation Script
# Compares R loess() results with p2a-core output

# ============================================================================
# Test Case 1: Simple linear data with noise
# ============================================================================

cat("=== LOESS Validation ===\n\n")

set.seed(42)

# Test Case 1: Nearly linear data
x1 <- 1:20
y1 <- 2 * x1 + 1 + rnorm(20, sd = 1)

fit1 <- loess(y1 ~ x1, span = 0.75, degree = 2, family = "gaussian")

cat("Test Case 1: Linear data with noise (n=20, span=0.75, degree=2)\n")
cat("R loess fitted values:\n")
print(round(fitted(fit1), 6))
cat("\nR loess summary:\n")
print(summary(fit1))
cat("\n")

# Save expected values
write.csv(
  data.frame(
    x = x1,
    y = y1,
    fitted = fitted(fit1),
    residuals = residuals(fit1)
  ),
  "validation/expected/loess_test1.csv",
  row.names = FALSE
)

# ============================================================================
# Test Case 2: Sinusoidal data
# ============================================================================

x2 <- seq(0, 4 * pi, length.out = 50)
y2 <- sin(x2) + 0.2 * rnorm(50)

fit2 <- loess(y2 ~ x2, span = 0.3, degree = 2, family = "gaussian")

cat("Test Case 2: Sinusoidal data (n=50, span=0.3, degree=2)\n")
cat("R loess ENP (equivalent number of parameters):", fit2$enp, "\n")
cat("R loess RSE:", sqrt(sum(residuals(fit2)^2) / (length(y2) - fit2$enp)), "\n")
cat("R loess RSS:", sum(residuals(fit2)^2), "\n")
cat("\n")

write.csv(
  data.frame(
    x = x2,
    y = y2,
    fitted = fitted(fit2),
    residuals = residuals(fit2)
  ),
  "validation/expected/loess_test2.csv",
  row.names = FALSE
)

# ============================================================================
# Test Case 3: Robust fitting with outliers
# ============================================================================

x3 <- 1:30
y3 <- 2 * x3 + 1
y3[10] <- 100  # Outlier
y3[20] <- -50  # Outlier

fit3_normal <- loess(y3 ~ x3, span = 0.75, degree = 2, family = "gaussian")
fit3_robust <- loess(y3 ~ x3, span = 0.75, degree = 2, family = "symmetric")

cat("Test Case 3: Data with outliers\n")
cat("Gaussian family RSS:", sum(residuals(fit3_normal)^2), "\n")
cat("Symmetric (robust) family RSS:", sum(residuals(fit3_robust)^2), "\n")
cat("\n")

write.csv(
  data.frame(
    x = x3,
    y = y3,
    fitted_normal = fitted(fit3_normal),
    fitted_robust = fitted(fit3_robust)
  ),
  "validation/expected/loess_test3.csv",
  row.names = FALSE
)

# ============================================================================
# Test Case 4: Different degrees
# ============================================================================

x4 <- seq(0, 10, length.out = 100)
y4 <- x4^2 / 20 + 0.5 * rnorm(100)

fit4_deg1 <- loess(y4 ~ x4, span = 0.5, degree = 1, family = "gaussian")
fit4_deg2 <- loess(y4 ~ x4, span = 0.5, degree = 2, family = "gaussian")

cat("Test Case 4: Degree comparison (quadratic data)\n")
cat("Degree 1 ENP:", fit4_deg1$enp, "\n")
cat("Degree 2 ENP:", fit4_deg2$enp, "\n")
cat("Degree 1 RSS:", sum(residuals(fit4_deg1)^2), "\n")
cat("Degree 2 RSS:", sum(residuals(fit4_deg2)^2), "\n")
cat("\n")

write.csv(
  data.frame(
    x = x4,
    y = y4,
    fitted_deg1 = fitted(fit4_deg1),
    fitted_deg2 = fitted(fit4_deg2)
  ),
  "validation/expected/loess_test4.csv",
  row.names = FALSE
)

# ============================================================================
# Test Case 5: Different spans
# ============================================================================

x5 <- seq(0, 2 * pi, length.out = 100)
y5 <- sin(x5) + cos(2 * x5) + 0.1 * rnorm(100)

fit5_span03 <- loess(y5 ~ x5, span = 0.3, degree = 2, family = "gaussian")
fit5_span05 <- loess(y5 ~ x5, span = 0.5, degree = 2, family = "gaussian")
fit5_span08 <- loess(y5 ~ x5, span = 0.8, degree = 2, family = "gaussian")

cat("Test Case 5: Span comparison (complex waveform)\n")
cat("Span 0.3 ENP:", fit5_span03$enp, "\n")
cat("Span 0.5 ENP:", fit5_span05$enp, "\n")
cat("Span 0.8 ENP:", fit5_span08$enp, "\n")
cat("\n")

write.csv(
  data.frame(
    x = x5,
    y = y5,
    fitted_span03 = fitted(fit5_span03),
    fitted_span05 = fitted(fit5_span05),
    fitted_span08 = fitted(fit5_span08)
  ),
  "validation/expected/loess_test5.csv",
  row.names = FALSE
)

cat("=== Validation complete. Expected results saved to validation/expected/ ===\n")
