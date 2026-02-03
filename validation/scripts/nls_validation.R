#!/usr/bin/env Rscript
# NLS (Nonlinear Least Squares) Validation Script
# Compares R's stats::nls() results with p2a-core Rust implementation
#
# Run with: Rscript validation/scripts/nls_validation.R

set.seed(42)

cat("=======================================================\n")
cat("NLS Validation: R vs p2a-core Rust\n")
cat("=======================================================\n\n")

# =============================================================================
# Test Case 1: Exponential Decay
# Model: y = a * exp(-b * x) + c
# =============================================================================

cat("=== Test Case 1: Exponential Decay ===\n")
cat("Model: y = a * exp(-b * x) + c\n\n")

# Same data as Rust tests
x_exp <- c(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
y_exp <- c(11.8, 9.7, 8.0, 6.5, 5.7, 4.8, 4.2, 3.7, 3.4, 3.1, 2.9)

fit_exp <- nls(y_exp ~ a * exp(-b * x_exp) + c,
               start = list(a = 8, b = 0.3, c = 1))

cat("R nls() Results:\n")
print(summary(fit_exp))

cat("\nCoefficients for comparison:\n")
coef_exp <- coef(fit_exp)
se_exp <- summary(fit_exp)$coefficients[, "Std. Error"]
cat(sprintf("  a = %.6f (SE = %.6f)\n", coef_exp["a"], se_exp["a"]))
cat(sprintf("  b = %.6f (SE = %.6f)\n", coef_exp["b"], se_exp["b"]))
cat(sprintf("  c = %.6f (SE = %.6f)\n", coef_exp["c"], se_exp["c"]))
cat(sprintf("  RSS = %.6f\n", sum(residuals(fit_exp)^2)))
cat(sprintf("  Sigma = %.6f\n", summary(fit_exp)$sigma))

# =============================================================================
# Test Case 2: Michaelis-Menten Kinetics
# Model: V = Vmax * S / (Km + S)
# =============================================================================

cat("\n=== Test Case 2: Michaelis-Menten Kinetics ===\n")
cat("Model: V = Vmax * S / (Km + S)\n\n")

S <- c(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
V <- c(28.6, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1)

fit_mm <- nls(V ~ Vmax * S / (Km + S),
              start = list(Vmax = 150, Km = 0.05))

cat("R nls() Results:\n")
print(summary(fit_mm))

cat("\nCoefficients for comparison:\n")
coef_mm <- coef(fit_mm)
se_mm <- summary(fit_mm)$coefficients[, "Std. Error"]
cat(sprintf("  Vmax = %.6f (SE = %.6f)\n", coef_mm["Vmax"], se_mm["Vmax"]))
cat(sprintf("  Km = %.6f (SE = %.6f)\n", coef_mm["Km"], se_mm["Km"]))
cat(sprintf("  RSS = %.6f\n", sum(residuals(fit_mm)^2)))

# =============================================================================
# Test Case 3: Logistic Growth
# Model: y = K / (1 + exp(-r * (x - x0)))
# =============================================================================

cat("\n=== Test Case 3: Logistic Growth ===\n")
cat("Model: y = K / (1 + exp(-r * (x - x0)))\n\n")

x_log <- seq(0, 10, by = 0.5)
# Generate data with true K=100, r=1.5, x0=5
y_log <- 100 / (1 + exp(-1.5 * (x_log - 5))) + rnorm(length(x_log), 0, 2)

fit_log <- nls(y_log ~ K / (1 + exp(-r * (x_log - x0))),
               start = list(K = 80, r = 1.0, x0 = 4))

cat("R nls() Results:\n")
print(summary(fit_log))

coef_log <- coef(fit_log)
se_log <- summary(fit_log)$coefficients[, "Std. Error"]
cat("\nCoefficients for comparison:\n")
cat(sprintf("  K = %.6f (SE = %.6f)\n", coef_log["K"], se_log["K"]))
cat(sprintf("  r = %.6f (SE = %.6f)\n", coef_log["r"], se_log["r"]))
cat(sprintf("  x0 = %.6f (SE = %.6f)\n", coef_log["x0"], se_log["x0"]))

# =============================================================================
# Test Case 4: Power Law
# Model: y = a * x^b
# =============================================================================

cat("\n=== Test Case 4: Power Law ===\n")
cat("Model: y = a * x^b\n\n")

x_pow <- seq(1, 10, by = 0.5)
# Generate data with true a=2, b=1.5
y_pow <- 2 * x_pow^1.5 + rnorm(length(x_pow), 0, 0.5)

fit_pow <- nls(y_pow ~ a * x_pow^b,
               start = list(a = 1, b = 1))

cat("R nls() Results:\n")
print(summary(fit_pow))

coef_pow <- coef(fit_pow)
se_pow <- summary(fit_pow)$coefficients[, "Std. Error"]
cat("\nCoefficients for comparison:\n")
cat(sprintf("  a = %.6f (SE = %.6f)\n", coef_pow["a"], se_pow["a"]))
cat(sprintf("  b = %.6f (SE = %.6f)\n", coef_pow["b"], se_pow["b"]))

# =============================================================================
# Summary Table for Rust Comparison
# =============================================================================

cat("\n=======================================================\n")
cat("SUMMARY TABLE FOR RUST TEST VALIDATION\n")
cat("=======================================================\n\n")

cat("Exponential Decay (y = a*exp(-b*x) + c):\n")
cat(sprintf("  Expected: a=%.4f, b=%.4f, c=%.4f, RSS=%.4f\n",
            coef_exp["a"], coef_exp["b"], coef_exp["c"], sum(residuals(fit_exp)^2)))

cat("\nMichaelis-Menten (V = Vmax*S/(Km+S)):\n")
cat(sprintf("  Expected: Vmax=%.4f, Km=%.4f, RSS=%.4f\n",
            coef_mm["Vmax"], coef_mm["Km"], sum(residuals(fit_mm)^2)))

cat("\nLogistic Growth (y = K/(1+exp(-r*(x-x0)))):\n")
cat(sprintf("  Expected: K=%.4f, r=%.4f, x0=%.4f\n",
            coef_log["K"], coef_log["r"], coef_log["x0"]))

cat("\nPower Law (y = a*x^b):\n")
cat(sprintf("  Expected: a=%.4f, b=%.4f\n",
            coef_pow["a"], coef_pow["b"]))

# =============================================================================
# Save expected results
# =============================================================================

cat("\n=======================================================\n")
cat("Saving expected results to CSV...\n")

# Create validation/expected directory if it doesn't exist
dir.create("validation/expected", recursive = TRUE, showWarnings = FALSE)

# Save exponential decay results
exp_results <- data.frame(
  parameter = c("a", "b", "c", "RSS", "sigma"),
  value = c(coef_exp["a"], coef_exp["b"], coef_exp["c"],
            sum(residuals(fit_exp)^2), summary(fit_exp)$sigma),
  std_error = c(se_exp["a"], se_exp["b"], se_exp["c"], NA, NA)
)
write.csv(exp_results, "validation/expected/nls_exponential_decay.csv", row.names = FALSE)

# Save Michaelis-Menten results
mm_results <- data.frame(
  parameter = c("Vmax", "Km", "RSS"),
  value = c(coef_mm["Vmax"], coef_mm["Km"], sum(residuals(fit_mm)^2)),
  std_error = c(se_mm["Vmax"], se_mm["Km"], NA)
)
write.csv(mm_results, "validation/expected/nls_michaelis_menten.csv", row.names = FALSE)

cat("Done. Results saved to validation/expected/\n")
cat("=======================================================\n")
