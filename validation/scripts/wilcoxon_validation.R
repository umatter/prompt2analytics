#!/usr/bin/env Rscript
# Wilcoxon Test Validation Script
# Compares R wilcox.test() results with p2a-core output
#
# Run with: Rscript validation/scripts/wilcoxon_validation.R

cat("=== Wilcoxon Test R Validation ===\n\n")

# ==============================================================================
# Test 1: Two-sample rank sum test
# ==============================================================================

cat("--- Test 1: Two-sample rank sum test ---\n")

x1 <- c(1.2, 2.3, 3.1, 4.5, 5.2)
y1 <- c(2.1, 3.4, 4.2, 5.8, 6.1, 7.2)

# Normal approximation with continuity correction
result1 <- wilcox.test(x1, y1, exact = FALSE, correct = TRUE)
cat("x:", x1, "\n")
cat("y:", y1, "\n")
cat("Result:\n")
print(result1)
cat("\nExpected for Rust validation:\n")
cat(sprintf("  statistic (W): %.1f\n", result1$statistic))
cat(sprintf("  p-value: %.6f\n", result1$p.value))
cat("\n")

# ==============================================================================
# Test 2: One-sample signed rank test
# ==============================================================================

cat("--- Test 2: One-sample signed rank test ---\n")

x2 <- c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)
mu2 <- 1.5

result2 <- wilcox.test(x2, mu = mu2, exact = FALSE, correct = TRUE)
cat("x:", x2, "\n")
cat("mu:", mu2, "\n")
cat("Result:\n")
print(result2)
cat("\nExpected for Rust validation:\n")
cat(sprintf("  statistic (V): %.1f\n", result2$statistic))
cat(sprintf("  p-value: %.6f\n", result2$p.value))
cat("\n")

# ==============================================================================
# Test 3: Exact test (small sample, no ties)
# ==============================================================================

cat("--- Test 3: Exact test (small sample) ---\n")

x3 <- c(1, 2, 3)
y3 <- c(4, 5, 6)

result3 <- wilcox.test(x3, y3, exact = TRUE)
cat("x:", x3, "\n")
cat("y:", y3, "\n")
cat("Result:\n")
print(result3)
cat("\nExpected for Rust validation:\n")
cat(sprintf("  statistic (W): %.1f\n", result3$statistic))
cat(sprintf("  p-value: %.6f\n", result3$p.value))
cat("\n")

# ==============================================================================
# Test 4: Paired signed rank test
# ==============================================================================

cat("--- Test 4: Paired signed rank test ---\n")

x4 <- c(125, 115, 130, 140, 140, 115, 140, 125, 140, 135)
y4 <- c(110, 122, 125, 120, 140, 124, 123, 137, 135, 145)

result4 <- wilcox.test(x4, y4, paired = TRUE, exact = FALSE, correct = TRUE)
cat("x:", x4, "\n")
cat("y:", y4, "\n")
cat("Differences:", x4 - y4, "\n")
cat("Result:\n")
print(result4)
cat("\nExpected for Rust validation:\n")
cat(sprintf("  statistic (V): %.1f\n", result4$statistic))
cat(sprintf("  p-value: %.6f\n", result4$p.value))
cat("\n")

# ==============================================================================
# Test 5: With ties
# ==============================================================================

cat("--- Test 5: Data with ties ---\n")

x5 <- c(1.0, 2.0, 2.0, 3.0, 4.0)
y5 <- c(2.0, 3.0, 3.0, 4.0, 5.0)

result5 <- wilcox.test(x5, y5, exact = FALSE, correct = TRUE)
cat("x:", x5, "\n")
cat("y:", y5, "\n")
cat("Result:\n")
print(result5)
cat("\nExpected for Rust validation:\n")
cat(sprintf("  statistic (W): %.1f\n", result5$statistic))
cat(sprintf("  p-value: %.6f\n", result5$p.value))
cat("\n")

# ==============================================================================
# Test 6: One-sided alternatives
# ==============================================================================

cat("--- Test 6: One-sided alternatives ---\n")

x6 <- c(1.0, 2.0, 3.0, 4.0, 5.0)
y6 <- c(6.0, 7.0, 8.0, 9.0, 10.0)

result6a <- wilcox.test(x6, y6, alternative = "less", exact = FALSE, correct = TRUE)
result6b <- wilcox.test(x6, y6, alternative = "greater", exact = FALSE, correct = TRUE)
result6c <- wilcox.test(x6, y6, alternative = "two.sided", exact = FALSE, correct = TRUE)

cat("x:", x6, "\n")
cat("y:", y6, "\n")
cat("\nAlternative 'less' (x < y):\n")
cat(sprintf("  p-value: %.6f\n", result6a$p.value))
cat("\nAlternative 'greater' (x > y):\n")
cat(sprintf("  p-value: %.6f\n", result6b$p.value))
cat("\nAlternative 'two.sided':\n")
cat(sprintf("  p-value: %.6f\n", result6c$p.value))
cat("\n")

# ==============================================================================
# Test 7: Confidence interval
# ==============================================================================

cat("--- Test 7: Confidence interval ---\n")

x7 <- c(1.0, 2.0, 3.0, 4.0, 5.0)
y7 <- c(3.0, 4.0, 5.0, 6.0, 7.0)

result7 <- wilcox.test(x7, y7, conf.int = TRUE, conf.level = 0.95)
cat("x:", x7, "\n")
cat("y:", y7, "\n")
cat("Result with CI:\n")
print(result7)
cat("\nExpected for Rust validation:\n")
cat(sprintf("  estimate (Hodges-Lehmann): %.4f\n", result7$estimate))
cat(sprintf("  CI: (%.4f, %.4f)\n", result7$conf.int[1], result7$conf.int[2]))
cat("\n")

# ==============================================================================
# Summary table
# ==============================================================================

cat("=== Summary Table for Validation ===\n\n")
cat("Test | Description                     | W/V     | p-value   | Notes\n")
cat("-----|--------------------------------|---------|-----------|------\n")
cat(sprintf("1    | Two-sample rank sum            | %.1f    | %.6f | approx w/ CC\n",
            result1$statistic, result1$p.value))
cat(sprintf("2    | One-sample signed rank         | %.1f    | %.6f | mu=1.5\n",
            result2$statistic, result2$p.value))
cat(sprintf("3    | Exact (small sample)           | %.1f     | %.6f | n=3 each\n",
            result3$statistic, result3$p.value))
cat(sprintf("4    | Paired signed rank             | %.1f    | %.6f | n=10 pairs\n",
            result4$statistic, result4$p.value))
cat(sprintf("5    | With ties                      | %.1f    | %.6f | has ties\n",
            result5$statistic, result5$p.value))

cat("\n=== Validation Complete ===\n")
