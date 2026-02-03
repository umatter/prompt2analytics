#!/usr/bin/env Rscript
# Factor Analysis Validation Script
# Compares R factanal() results with p2a-core output

# ============================================================================
# Test 1: Synthetic Two-Factor Structure
# ============================================================================

cat("=== Factor Analysis Validation ===\n\n")

set.seed(42)
n <- 200

# Generate latent factors
f1 <- rnorm(n)
f2 <- rnorm(n)

# Generate observed variables with known factor structure
# Variables 1-3 load on Factor 1
# Variables 4-6 load on Factor 2
data <- data.frame(
  x1 = 0.8 * f1 + sqrt(1 - 0.8^2) * rnorm(n),
  x2 = 0.7 * f1 + sqrt(1 - 0.7^2) * rnorm(n),
  x3 = 0.75 * f1 + sqrt(1 - 0.75^2) * rnorm(n),
  x4 = 0.8 * f2 + sqrt(1 - 0.8^2) * rnorm(n),
  x5 = 0.7 * f2 + sqrt(1 - 0.7^2) * rnorm(n),
  x6 = 0.75 * f2 + sqrt(1 - 0.75^2) * rnorm(n)
)

cat("Test 1: Two-Factor Structure\n")
cat("----------------------------\n")
cat("n =", n, ", p = 6, k = 2\n\n")

# Run factor analysis with no rotation
result_none <- factanal(data, factors = 2, rotation = "none")

cat("Unrotated Solution:\n")
cat("Uniquenesses:\n")
print(round(result_none$uniquenesses, 4))
cat("\nLoadings:\n")
print(round(result_none$loadings[], 4))
cat("\nChi-squared:", round(result_none$STATISTIC, 4), "\n")
cat("df:", result_none$dof, "\n")
cat("p-value:", round(result_none$PVAL, 4), "\n")

# Run factor analysis with varimax rotation
result_varimax <- factanal(data, factors = 2, rotation = "varimax")

cat("\n\nVarimax Rotated Solution:\n")
cat("Uniquenesses:\n")
print(round(result_varimax$uniquenesses, 4))
cat("\nLoadings:\n")
print(round(result_varimax$loadings[], 4))
cat("\nChi-squared:", round(result_varimax$STATISTIC, 4), "\n")
cat("df:", result_varimax$dof, "\n")
cat("p-value:", round(result_varimax$PVAL, 4), "\n")

# Run factor analysis with promax rotation
result_promax <- factanal(data, factors = 2, rotation = "promax")

cat("\n\nPromax Rotated Solution:\n")
cat("Uniquenesses:\n")
print(round(result_promax$uniquenesses, 4))
cat("\nLoadings:\n")
print(round(result_promax$loadings[], 4))

# ============================================================================
# Test 2: Factor Scores
# ============================================================================

cat("\n\n=== Test 2: Factor Scores ===\n")

# Regression method
result_reg <- factanal(data, factors = 2, rotation = "varimax", scores = "regression")
cat("\nFactor Scores (Regression method, first 5 rows):\n")
print(round(head(result_reg$scores, 5), 4))

# Bartlett method
result_bart <- factanal(data, factors = 2, rotation = "varimax", scores = "Bartlett")
cat("\nFactor Scores (Bartlett method, first 5 rows):\n")
print(round(head(result_bart$scores, 5), 4))

# ============================================================================
# Test 3: Communalities
# ============================================================================

cat("\n\n=== Test 3: Communalities ===\n")
communalities <- 1 - result_varimax$uniquenesses
cat("Communalities:\n")
print(round(communalities, 4))
cat("\nSum of squared loadings per factor:\n")
ss_loadings <- colSums(result_varimax$loadings[]^2)
print(round(ss_loadings, 4))

# ============================================================================
# Save expected values for Rust validation
# ============================================================================

cat("\n\n=== Expected Values for Rust Validation ===\n")
cat("(Copy these to Rust tests)\n\n")

cat("Expected uniquenesses (varimax):\n")
cat("vec![", paste(round(result_varimax$uniquenesses, 6), collapse = ", "), "]\n\n")

cat("Expected communalities (varimax):\n")
cat("vec![", paste(round(communalities, 6), collapse = ", "), "]\n\n")

cat("Expected chi-squared:", round(result_varimax$STATISTIC, 6), "\n")
cat("Expected df:", result_varimax$dof, "\n")
cat("Expected p-value:", round(result_varimax$PVAL, 6), "\n")

cat("\n=== Validation Complete ===\n")
