# E-Value Validation Script
# Compares Rust implementation against R EValue package
#
# Reference:
# VanderWeele, T. J., & Ding, P. (2017). Sensitivity Analysis in Observational
# Research: Introducing the E-Value. Annals of Internal Medicine, 167(4), 268-274.

# Install if needed
if (!requireNamespace("EValue", quietly = TRUE)) {
  install.packages("EValue")
}

library(EValue)

cat("E-Value Validation Tests\n")
cat("========================\n\n")

# Test 1: Basic Risk Ratio
cat("Test 1: Basic Risk Ratio\n")
cat("------------------------\n")

# RR = 2.5
ev_2.5 <- evalues.RR(est = 2.5)
cat(sprintf("RR = 2.5: E-value = %.4f\n", ev_2.5$point$e.value))
cat("Expected (Rust): 4.44\n\n")

# RR = 3.9 (paper example)
ev_3.9 <- evalues.RR(est = 3.9)
cat(sprintf("RR = 3.9: E-value = %.4f\n", ev_3.9$point$e.value))
cat("Expected (Rust): 7.26\n\n")

# Test 2: Risk Ratio with CI
cat("Test 2: Risk Ratio with Confidence Interval\n")
cat("--------------------------------------------\n")

ev_ci <- evalues.RR(est = 2.5, lo = 1.8, hi = 3.5)
cat(sprintf("RR = 2.5, CI = [1.8, 3.5]\n"))
cat(sprintf("  Point E-value: %.4f\n", ev_ci$point$e.value))
cat(sprintf("  CI E-value: %.4f\n", ev_ci$lower$e.value))
cat("Expected (Rust): Point = 4.44, CI = 3.0 (approx)\n\n")

# Test 3: Odds Ratio - Rare
cat("Test 3: Odds Ratio (Rare Outcome)\n")
cat("---------------------------------\n")

ev_or_rare <- evalues.OR(est = 2.5, rare = TRUE)
cat(sprintf("OR = 2.5 (rare): E-value = %.4f\n", ev_or_rare$point$e.value))
cat("Expected (Rust): 4.44 (same as RR)\n\n")

# Test 4: Odds Ratio - Common
cat("Test 4: Odds Ratio (Common Outcome)\n")
cat("-----------------------------------\n")

ev_or_common <- evalues.OR(est = 4, rare = FALSE)
cat(sprintf("OR = 4 (common): E-value = %.4f\n", ev_or_common$point$e.value))
cat(sprintf("  Transformed RR: %.4f (sqrt(4) = 2)\n", sqrt(4)))
cat("Expected (Rust): 3.41 (for RR = 2)\n\n")

# Test 5: Standardized Mean Difference
cat("Test 5: Standardized Mean Difference\n")
cat("------------------------------------\n")

# SMD conversion: RR = exp(0.91 * d)
d <- 0.5
rr_from_smd <- exp(0.91 * d)
cat(sprintf("SMD = %.2f -> RR_approx = exp(0.91 * %.2f) = %.4f\n", d, d, rr_from_smd))

# E-value for this RR
ev_smd <- evalues.RR(est = rr_from_smd)
cat(sprintf("E-value for SMD = 0.5: %.4f\n", ev_smd$point$e.value))
cat("Expected (Rust): ~2.53\n\n")

# Note: evalues.MD expects different parameters
# Direct calculation for comparison
rr_approx <- exp(0.91 * 0.5)
manual_evalue <- rr_approx + sqrt(rr_approx * (rr_approx - 1))
cat(sprintf("Manual calculation: %.4f\n", manual_evalue))
cat("\n")

# Test 6: Protective Effect
cat("Test 6: Protective Effect (RR < 1)\n")
cat("----------------------------------\n")

ev_0.5 <- evalues.RR(est = 0.5)
ev_2.0 <- evalues.RR(est = 2.0)
cat(sprintf("RR = 0.5: E-value = %.4f\n", ev_0.5$point$e.value))
cat(sprintf("RR = 2.0: E-value = %.4f\n", ev_2.0$point$e.value))
cat("Expected: Both should be equal (symmetric around 1)\n\n")

# Test 7: Hazard Ratio
cat("Test 7: Hazard Ratio\n")
cat("--------------------\n")

ev_hr_rare <- evalues.HR(est = 1.5, rare = TRUE)
cat(sprintf("HR = 1.5 (rare): E-value = %.4f\n", ev_hr_rare$point$e.value))

ev_hr_common <- evalues.HR(est = 4, rare = FALSE)
cat(sprintf("HR = 4 (common): E-value = %.4f\n", ev_hr_common$point$e.value))
cat("\n")

# Summary table
cat("Summary of Expected Values for Rust Implementation\n")
cat("===================================================\n")
cat(sprintf("%-30s %s\n", "Test Case", "Expected E-value"))
cat(sprintf("%-30s %s\n", "------", "----------------"))
cat(sprintf("%-30s %.4f\n", "evalue_rr(2.5)", ev_2.5$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_rr(3.9)", ev_3.9$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_rr_ci point", ev_ci$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_rr_ci lower", ev_ci$lower$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_or(2.5, rare=T)", ev_or_rare$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_or(4, rare=F)", ev_or_common$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_smd(0.5)", manual_evalue))
cat(sprintf("%-30s %.4f\n", "evalue_rr(0.5)", ev_0.5$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_hr(1.5, rare=T)", ev_hr_rare$point$e.value))
cat(sprintf("%-30s %.4f\n", "evalue_hr(4, rare=F)", ev_hr_common$point$e.value))
