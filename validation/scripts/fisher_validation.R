#!/usr/bin/env Rscript
# Fisher's Exact Test Validation Script
# Compares R stats::fisher.test results with p2a-core Rust output
#
# Run with: Rscript validation/scripts/fisher_validation.R

cat("Fisher's Exact Test Validation\n")
cat("==============================\n\n")

# ============================================================================
# Test Case 1: Lady Tasting Tea (Classic Example)
# ============================================================================

cat("Test Case 1: Lady Tasting Tea\n")
cat("-------------------------------\n")

M1 <- matrix(c(3, 1, 1, 3), nrow = 2, byrow = TRUE)
cat("Table:\n")
print(M1)

result1 <- fisher.test(M1)
cat("\nR Results:\n")
cat(sprintf("  p-value: %.10f\n", result1$p.value))
cat(sprintf("  odds ratio (CML estimate): %.10f\n", result1$estimate))
cat(sprintf("  95%% CI: (%.6f, %.6f)\n", result1$conf.int[1], result1$conf.int[2]))

# Sample odds ratio (for comparison with Rust)
sample_or <- (M1[1,1] * M1[2,2]) / (M1[1,2] * M1[2,1])
cat(sprintf("  sample odds ratio: %.6f\n", sample_or))

cat("\nExpected Rust values:\n")
cat(sprintf("  p_value: ~%.4f\n", result1$p.value))
cat(sprintf("  sample_odds_ratio: %.6f\n", sample_or))
cat("\n")

# ============================================================================
# Test Case 2: Significant Association
# ============================================================================

cat("Test Case 2: Significant Association\n")
cat("-------------------------------------\n")

M2 <- matrix(c(1, 11, 9, 3), nrow = 2, byrow = TRUE)
cat("Table:\n")
print(M2)

result2 <- fisher.test(M2)
cat("\nR Results:\n")
cat(sprintf("  p-value: %.10f\n", result2$p.value))
cat(sprintf("  odds ratio (CML): %.10f\n", result2$estimate))
cat(sprintf("  95%% CI: (%.6f, %.6f)\n", result2$conf.int[1], result2$conf.int[2]))

sample_or2 <- (M2[1,1] * M2[2,2]) / (M2[1,2] * M2[2,1])
cat(sprintf("  sample odds ratio: %.6f\n", sample_or2))

cat("\nExpected Rust values:\n")
cat(sprintf("  p_value: ~%.6f (should be < 0.01)\n", result2$p.value))
cat(sprintf("  sample_odds_ratio: %.6f\n", sample_or2))
cat("\n")

# ============================================================================
# Test Case 3: One-Sided Tests
# ============================================================================

cat("Test Case 3: One-Sided Tests\n")
cat("-----------------------------\n")

M3 <- matrix(c(6, 2, 1, 7), nrow = 2, byrow = TRUE)
cat("Table:\n")
print(M3)

result3_two <- fisher.test(M3, alternative = "two.sided")
result3_greater <- fisher.test(M3, alternative = "greater")
result3_less <- fisher.test(M3, alternative = "less")

cat("\nR Results:\n")
cat(sprintf("  two.sided p-value: %.10f\n", result3_two$p.value))
cat(sprintf("  greater p-value: %.10f\n", result3_greater$p.value))
cat(sprintf("  less p-value: %.10f\n", result3_less$p.value))

sample_or3 <- (M3[1,1] * M3[2,2]) / (M3[1,2] * M3[2,1])
cat(sprintf("  sample odds ratio: %.6f\n", sample_or3))

cat("\nExpected Rust values:\n")
cat(sprintf("  two_sided: ~%.6f\n", result3_two$p.value))
cat(sprintf("  greater: ~%.6f\n", result3_greater$p.value))
cat(sprintf("  less: ~%.6f\n", result3_less$p.value))
cat("\n")

# ============================================================================
# Test Case 4: Zero Cell
# ============================================================================

cat("Test Case 4: Zero Cell\n")
cat("-----------------------\n")

M4 <- matrix(c(0, 5, 5, 10), nrow = 2, byrow = TRUE)
cat("Table:\n")
print(M4)

result4 <- fisher.test(M4)
cat("\nR Results:\n")
cat(sprintf("  p-value: %.10f\n", result4$p.value))
cat(sprintf("  odds ratio (CML): %.6f\n", result4$estimate))

cat("\nExpected Rust values:\n")
cat(sprintf("  p_value: ~%.4f\n", result4$p.value))
cat("  sample_odds_ratio: 0.0 (zero cell)\n")
cat("\n")

# ============================================================================
# Test Case 5: Extreme Table
# ============================================================================

cat("Test Case 5: Extreme Table\n")
cat("---------------------------\n")

M5 <- matrix(c(50, 1, 1, 50), nrow = 2, byrow = TRUE)
cat("Table:\n")
print(M5)

result5 <- fisher.test(M5)
cat("\nR Results:\n")
cat(sprintf("  p-value: %.20f\n", result5$p.value))
cat(sprintf("  odds ratio (CML): %.6f\n", result5$estimate))

sample_or5 <- (M5[1,1] * M5[2,2]) / (M5[1,2] * M5[2,1])
cat(sprintf("  sample odds ratio: %.6f\n", sample_or5))

cat("\nExpected Rust values:\n")
cat(sprintf("  p_value: < 1e-20 (highly significant)\n"))
cat(sprintf("  sample_odds_ratio: %.1f\n", sample_or5))
cat("\n")

# ============================================================================
# Summary Table for Documentation
# ============================================================================

cat("\n======================================\n")
cat("SUMMARY TABLE (for validation/stats/fisher.md)\n")
cat("======================================\n\n")

cat("| Test Case | Statistic | R Value | Tolerance |\n")
cat("|-----------|-----------|---------|----------|\n")
cat(sprintf("| Lady Tea | p-value | %.6f | 1e-4 |\n", result1$p.value))
cat(sprintf("| Lady Tea | sample OR | %.6f | 1e-6 |\n", sample_or))
cat(sprintf("| Significant | p-value | %.6f | 1e-4 |\n", result2$p.value))
cat(sprintf("| One-sided (greater) | p-value | %.6f | 1e-4 |\n", result3_greater$p.value))
cat(sprintf("| One-sided (less) | p-value | %.6f | 1e-4 |\n", result3_less$p.value))
cat(sprintf("| Zero cell | p-value | %.6f | 1e-4 |\n", result4$p.value))
cat(sprintf("| Extreme | p-value | %.2e | 1e-15 |\n", result5$p.value))

cat("\nNote: R's fisher.test uses Conditional Maximum Likelihood (CML) estimate for odds ratio.\n")
cat("Our Rust implementation uses Sample odds ratio = (a*d)/(b*c).\n")
cat("Both are valid; CML is asymptotically more efficient.\n")
