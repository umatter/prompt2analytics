#!/usr/bin/env Rscript
#
# R Validation Script for IV/2SLS
# Generates reference values for p2a-core validation tests
#
# Dependencies: AER (for ivreg)
#

cat("=== IV/2SLS Validation Script ===\n\n")

# Install packages if needed
if (!requireNamespace("AER", quietly = TRUE)) {
  install.packages("AER", repos = "https://cloud.r-project.org")
}

library(AER)

# =============================================================================
# Test 1: Basic 2SLS - Just-Identified Case
# =============================================================================
cat("--- Test 1: Basic 2SLS (Just-Identified) ---\n")

set.seed(42)
n <- 200

# Generate valid instrument z
z <- rnorm(n)

# Endogenous x: correlated with error term through common factor
u_common <- rnorm(n)
x_endog <- 2.0 * z + u_common + rnorm(n, 0, 0.5)

# Outcome: y = 0.5 + 1.0*x_endog + error (error correlated with x_endog)
epsilon <- 0.5 * u_common + rnorm(n, 0, 0.5)  # Error correlated with u_common
y <- 0.5 + 1.0 * x_endog + epsilon

data_iv <- data.frame(y = y, x = x_endog, z = z)

# 2SLS using AER::ivreg
iv_fit <- ivreg(y ~ x | z, data = data_iv)

cat("\n2SLS Coefficients:\n")
print(coef(iv_fit))

cat("\n2SLS Standard Errors:\n")
print(sqrt(diag(vcov(iv_fit))))

cat("\n2SLS Summary:\n")
iv_sum <- summary(iv_fit)
print(iv_sum)

# Compare with OLS (biased)
ols_fit <- lm(y ~ x, data = data_iv)
cat("\nOLS Coefficients (biased due to endogeneity):\n")
print(coef(ols_fit))

# =============================================================================
# Test 2: Over-Identified Case (Multiple Instruments)
# =============================================================================
cat("\n--- Test 2: Over-Identified 2SLS (2 Instruments for 1 Endogenous) ---\n")

set.seed(42)
n <- 300

# Two instruments
z1 <- rnorm(n)
z2 <- rnorm(n)

# Common error component (creates endogeneity)
u_common <- rnorm(n)

# Endogenous x
x_endog <- 1.5 * z1 + 1.0 * z2 + u_common + rnorm(n, 0, 0.5)

# Outcome with endogeneity
epsilon <- 0.5 * u_common + rnorm(n, 0, 0.5)
y <- 1.0 + 0.8 * x_endog + epsilon

data_over <- data.frame(y = y, x = x_endog, z1 = z1, z2 = z2)

# 2SLS with multiple instruments
iv_over <- ivreg(y ~ x | z1 + z2, data = data_over)

cat("\n2SLS Coefficients:\n")
print(coef(iv_over))

cat("\n2SLS Standard Errors:\n")
print(sqrt(diag(vcov(iv_over))))

# Sargan test for overidentifying restrictions
cat("\nSargan Test (via summary diagnostics):\n")
iv_over_sum <- summary(iv_over, diagnostics = TRUE)
# The Sargan test is reported as Wu-Hausman in some versions
print(iv_over_sum$diagnostics)

# =============================================================================
# Test 3: First-Stage Diagnostics
# =============================================================================
cat("\n--- Test 3: First-Stage Diagnostics ---\n")

# Using same data_over
first_stage <- lm(x ~ z1 + z2, data = data_over)

cat("\nFirst-Stage Coefficients:\n")
print(coef(first_stage))

cat("\nFirst-Stage R-squared:\n")
cat(summary(first_stage)$r.squared, "\n")

cat("\nFirst-Stage F-statistic:\n")
fs_sum <- summary(first_stage)
cat("  F-stat:", fs_sum$fstatistic[1], "\n")
cat("  df1:", fs_sum$fstatistic[2], "\n")
cat("  df2:", fs_sum$fstatistic[3], "\n")

# Stock-Yogo weak instrument test guidance:
# F > 10 suggests instruments are not weak (single endogenous regressor)
cat("\nWeak Instrument Test:\n")
if (fs_sum$fstatistic[1] > 10) {
  cat("  F > 10: Instruments appear strong\n")
} else {
  cat("  F < 10: Potential weak instrument problem\n")
}

# =============================================================================
# Test 4: IV with Exogenous Controls
# =============================================================================
cat("\n--- Test 4: IV with Exogenous Controls ---\n")

set.seed(42)
n <- 250

# Exogenous control
w <- rnorm(n)

# Instrument
z <- rnorm(n)

# Common error
u <- rnorm(n)

# Endogenous x
x_endog <- 1.5 * z + 0.5 * w + u + rnorm(n, 0, 0.3)

# Outcome
epsilon <- 0.4 * u + rnorm(n, 0, 0.4)
y <- 0.5 + 0.8 * x_endog + 0.6 * w + epsilon

data_ctrl <- data.frame(y = y, x = x_endog, w = w, z = z)

# 2SLS with exogenous control
# Formula: y ~ x + w | z + w (w is both control and instrument for itself)
iv_ctrl <- ivreg(y ~ x + w | z + w, data = data_ctrl)

cat("\n2SLS Coefficients:\n")
print(coef(iv_ctrl))

cat("\n2SLS Standard Errors:\n")
print(sqrt(diag(vcov(iv_ctrl))))

# =============================================================================
# Test 5: Card (1995) Style Education-Wage Example
# =============================================================================
cat("\n--- Test 5: Returns to Education (Simulated Card-style) ---\n")

set.seed(42)
n <- 500

# Instrument: proximity to college (affects education)
near_college <- rbinom(n, 1, 0.5)

# Unobserved ability (creates endogeneity)
ability <- rnorm(n)

# Education (endogenous: affected by ability)
educ <- 10 + 2 * near_college + 0.5 * ability + rnorm(n, 0, 1)

# Wage (log)
lwage <- 1.0 + 0.1 * educ + 0.3 * ability + rnorm(n, 0, 0.3)

card_data <- data.frame(lwage = lwage, educ = educ, near_college = near_college)

# OLS (biased due to ability)
ols_card <- lm(lwage ~ educ, data = card_data)

# 2SLS (using proximity to college as instrument)
iv_card <- ivreg(lwage ~ educ | near_college, data = card_data)

cat("\nOLS Return to Education (biased upward):\n")
print(coef(ols_card))

cat("\n2SLS Return to Education (consistent):\n")
print(coef(iv_card))

cat("\n2SLS Standard Errors:\n")
print(sqrt(diag(vcov(iv_card))))

# First stage
first_stage_card <- lm(educ ~ near_college, data = card_data)
cat("\nFirst-Stage F-statistic:", summary(first_stage_card)$fstatistic[1], "\n")

# =============================================================================
# Write expected values to CSV
# =============================================================================
cat("\n--- Writing expected values to CSV ---\n")

# Basic IV results
iv_basic <- data.frame(
  variable = names(coef(iv_fit)),
  coefficient = coef(iv_fit),
  std_error = sqrt(diag(vcov(iv_fit)))
)
write.csv(iv_basic, "validation/expected/iv_basic.csv", row.names = FALSE)

# Over-identified IV results
iv_overid <- data.frame(
  variable = names(coef(iv_over)),
  coefficient = coef(iv_over),
  std_error = sqrt(diag(vcov(iv_over)))
)
write.csv(iv_overid, "validation/expected/iv_overidentified.csv", row.names = FALSE)

cat("\nExpected values written to validation/expected/\n")
cat("=== IV/2SLS Validation Script Complete ===\n")
