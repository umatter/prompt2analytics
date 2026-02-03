#!/usr/bin/env Rscript
# Canonical Correlation Analysis Validation Script
# Compares R cancor() results with p2a-core output

# Test Case 1: LifeCycleSavings dataset (classic example from R docs)
cat("=== Test Case 1: LifeCycleSavings ===\n")

data(LifeCycleSavings)
pop <- LifeCycleSavings[, 2:3]  # pop15, pop75
oec <- LifeCycleSavings[, -(2:3)]  # sr, dpi, ddpi

result1 <- cancor(pop, oec)

cat("\nCanonical Correlations:\n")
print(result1$cor)

cat("\nX Coefficients (pop):\n")
print(result1$xcoef)

cat("\nY Coefficients (oec):\n")
print(result1$ycoef)

# Save expected results
write.csv(data.frame(
  canonical_correlation = result1$cor
), "validation/expected/cancor_test1_correlations.csv", row.names = FALSE)

write.csv(result1$xcoef, "validation/expected/cancor_test1_xcoef.csv")
write.csv(result1$ycoef, "validation/expected/cancor_test1_ycoef.csv")

# Test Case 2: Synthetic data with known structure
cat("\n\n=== Test Case 2: Synthetic Data ===\n")

set.seed(42)
n <- 100

# Generate correlated data
z <- rnorm(n)
x1 <- z + rnorm(n, 0, 0.5)
x2 <- 0.8 * z + rnorm(n, 0, 0.5)
x3 <- 0.5 * z + rnorm(n, 0, 0.7)

y1 <- 0.9 * z + rnorm(n, 0, 0.4)
y2 <- 0.7 * z + rnorm(n, 0, 0.6)

X <- cbind(x1, x2, x3)
Y <- cbind(y1, y2)

result2 <- cancor(X, Y)

cat("\nCanonical Correlations:\n")
print(result2$cor)

cat("\nX Coefficients:\n")
print(result2$xcoef)

cat("\nY Coefficients:\n")
print(result2$ycoef)

# Verify canonical variates are uncorrelated
xscores <- scale(X, center = TRUE, scale = FALSE) %*% result2$xcoef
yscores <- scale(Y, center = TRUE, scale = FALSE) %*% result2$ycoef

cat("\nCorrelation between X canonical variates (should be ~identity):\n")
print(round(cor(xscores), 4))

cat("\nCorrelation between Y canonical variates (should be ~identity):\n")
print(round(cor(yscores), 4))

cat("\nCorrelation between X and Y canonical variates (diagonal should match canonical correlations):\n")
print(round(cor(xscores, yscores), 4))

# Save expected results for Test 2
write.csv(data.frame(
  canonical_correlation = result2$cor
), "validation/expected/cancor_test2_correlations.csv", row.names = FALSE)

write.csv(result2$xcoef, "validation/expected/cancor_test2_xcoef.csv")
write.csv(result2$ycoef, "validation/expected/cancor_test2_ycoef.csv")

# Save the synthetic data
write.csv(cbind(X, Y), "validation/expected/cancor_test2_data.csv", row.names = FALSE,
          col.names = c("x1", "x2", "x3", "y1", "y2"))

cat("\n\n=== Test Case 3: Simple 2x2 ===\n")

# Simple case with 2 variables each
set.seed(123)
n <- 50
z <- rnorm(n)
X3 <- cbind(z + rnorm(n, 0, 0.3), 0.5*z + rnorm(n, 0, 0.5))
Y3 <- cbind(0.8*z + rnorm(n, 0, 0.3), 0.3*z + rnorm(n, 0, 0.6))

result3 <- cancor(X3, Y3)

cat("\nCanonical Correlations:\n")
print(result3$cor)

write.csv(data.frame(
  canonical_correlation = result3$cor
), "validation/expected/cancor_test3_correlations.csv", row.names = FALSE)

cat("\n\nValidation complete. Expected results saved to validation/expected/\n")
