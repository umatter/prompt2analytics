# ANOVA Validation Script
# Compares R's aov/anova results with p2a-core output
#
# Usage: Rscript anova_validation.R

# ============================================================================
# Test Case 1: One-Way ANOVA (Classic Fertilizer Experiment)
# ============================================================================

cat("===============================================\n")
cat("Test Case 1: One-Way ANOVA\n")
cat("===============================================\n\n")

# Create test data: 3 groups with different means
set.seed(42)
data1 <- data.frame(
  yield = c(
    9.5, 10.2, 10.8, 9.8, 10.5,    # Group A: mean ~10
    14.2, 15.5, 14.8, 15.2, 15.8,  # Group B: mean ~15
    19.5, 20.2, 19.8, 20.5, 20.8   # Group C: mean ~20
  ),
  fertilizer = factor(rep(c("A", "B", "C"), each = 5))
)

cat("Data:\n")
print(data1)
cat("\n")

# Run one-way ANOVA
model1 <- aov(yield ~ fertilizer, data = data1)
result1 <- summary(model1)

cat("R One-Way ANOVA Results:\n")
cat("-----------------------\n")
print(result1)
cat("\n")

# Extract key statistics for comparison
anova_table1 <- result1[[1]]
cat("Key statistics for validation:\n")
cat(sprintf("  SS Between:     %.6f\n", anova_table1["fertilizer", "Sum Sq"]))
cat(sprintf("  SS Within:      %.6f\n", anova_table1["Residuals", "Sum Sq"]))
cat(sprintf("  DF Between:     %d\n", anova_table1["fertilizer", "Df"]))
cat(sprintf("  DF Within:      %d\n", anova_table1["Residuals", "Df"]))
cat(sprintf("  MS Between:     %.6f\n", anova_table1["fertilizer", "Mean Sq"]))
cat(sprintf("  MS Within:      %.6f\n", anova_table1["Residuals", "Mean Sq"]))
cat(sprintf("  F-statistic:    %.6f\n", anova_table1["fertilizer", "F value"]))
cat(sprintf("  p-value:        %.10f\n", anova_table1["fertilizer", "Pr(>F)"]))

# Group means
cat("\nGroup means:\n")
print(aggregate(yield ~ fertilizer, data = data1, mean))

# Grand mean
cat(sprintf("\nGrand mean: %.6f\n", mean(data1$yield)))

# Effect sizes
ss_b <- anova_table1["fertilizer", "Sum Sq"]
ss_total <- ss_b + anova_table1["Residuals", "Sum Sq"]
eta_sq <- ss_b / ss_total
cat(sprintf("Eta-squared: %.6f\n", eta_sq))

# Save expected results
write.csv(data.frame(
  ss_between = anova_table1["fertilizer", "Sum Sq"],
  ss_within = anova_table1["Residuals", "Sum Sq"],
  df_between = anova_table1["fertilizer", "Df"],
  df_within = anova_table1["Residuals", "Df"],
  ms_between = anova_table1["fertilizer", "Mean Sq"],
  ms_within = anova_table1["Residuals", "Mean Sq"],
  f_statistic = anova_table1["fertilizer", "F value"],
  p_value = anova_table1["fertilizer", "Pr(>F)"],
  eta_squared = eta_sq
), "validation/expected/anova_one_way_test1.csv", row.names = FALSE)

# ============================================================================
# Test Case 2: Two-Way ANOVA (2x2 Factorial)
# ============================================================================

cat("\n\n===============================================\n")
cat("Test Case 2: Two-Way ANOVA (with interaction)\n")
cat("===============================================\n\n")

# 2x2 factorial design: fertilizer (A, B) x water (Low, High)
data2 <- data.frame(
  yield = c(
    10.0, 11.0, 12.0,  # A-Low
    15.0, 16.0, 17.0,  # B-Low
    20.0, 21.0, 22.0,  # A-High
    30.0, 31.0, 32.0   # B-High
  ),
  fertilizer = factor(rep(c("A", "B", "A", "B"), each = 3)),
  water = factor(rep(c("Low", "Low", "High", "High"), each = 3))
)

cat("Data:\n")
print(data2)
cat("\n")

# Run two-way ANOVA with interaction
model2 <- aov(yield ~ fertilizer * water, data = data2)
result2 <- summary(model2)

cat("R Two-Way ANOVA Results (with interaction):\n")
cat("-------------------------------------------\n")
print(result2)
cat("\n")

# Extract key statistics
anova_table2 <- result2[[1]]
cat("Key statistics for validation:\n")
cat(sprintf("  SS Factor A:       %.6f\n", anova_table2["fertilizer", "Sum Sq"]))
cat(sprintf("  SS Factor B:       %.6f\n", anova_table2["water", "Sum Sq"]))
cat(sprintf("  SS Interaction:    %.6f\n", anova_table2["fertilizer:water", "Sum Sq"]))
cat(sprintf("  SS Error:          %.6f\n", anova_table2["Residuals", "Sum Sq"]))
cat(sprintf("  F (A):             %.6f (p=%.6f)\n",
    anova_table2["fertilizer", "F value"], anova_table2["fertilizer", "Pr(>F)"]))
cat(sprintf("  F (B):             %.6f (p=%.6f)\n",
    anova_table2["water", "F value"], anova_table2["water", "Pr(>F)"]))
cat(sprintf("  F (A:B):           %.6f (p=%.6f)\n",
    anova_table2["fertilizer:water", "F value"], anova_table2["fertilizer:water", "Pr(>F)"]))

# Save expected results
write.csv(data.frame(
  ss_a = anova_table2["fertilizer", "Sum Sq"],
  ss_b = anova_table2["water", "Sum Sq"],
  ss_ab = anova_table2["fertilizer:water", "Sum Sq"],
  ss_error = anova_table2["Residuals", "Sum Sq"],
  df_a = anova_table2["fertilizer", "Df"],
  df_b = anova_table2["water", "Df"],
  df_ab = anova_table2["fertilizer:water", "Df"],
  df_error = anova_table2["Residuals", "Df"],
  f_a = anova_table2["fertilizer", "F value"],
  f_b = anova_table2["water", "F value"],
  f_ab = anova_table2["fertilizer:water", "F value"],
  p_a = anova_table2["fertilizer", "Pr(>F)"],
  p_b = anova_table2["water", "Pr(>F)"],
  p_ab = anova_table2["fertilizer:water", "Pr(>F)"]
), "validation/expected/anova_two_way_test1.csv", row.names = FALSE)

# ============================================================================
# Test Case 3: Two-Way ANOVA (Additive Model)
# ============================================================================

cat("\n\n===============================================\n")
cat("Test Case 3: Two-Way ANOVA (additive, no interaction)\n")
cat("===============================================\n\n")

# Same data, additive model
model3 <- aov(yield ~ fertilizer + water, data = data2)
result3 <- summary(model3)

cat("R Two-Way ANOVA Results (additive):\n")
cat("-----------------------------------\n")
print(result3)

cat("\n\n===============================================\n")
cat("Validation Complete\n")
cat("===============================================\n")
cat("Expected values saved to validation/expected/\n")
