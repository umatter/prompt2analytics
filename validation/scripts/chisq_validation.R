#!/usr/bin/env Rscript
# Chi-Squared Test Validation Script
# Compares R stats::chisq.test results with p2a-core output

cat("=== Chi-Squared Test Validation Script ===\n\n")

# Test Case 1: Goodness-of-fit with uniform distribution
cat("Test 1: Goodness-of-fit (Uniform Distribution)\n")
x1 <- c(89, 37, 30, 28, 2)
cat("Observed counts:", x1, "\n")
cat("Total:", sum(x1), "\n\n")

result1 <- chisq.test(x1)
print(result1)

cat("\nExpected values (uniform):", result1$expected, "\n")
cat("Residuals:", result1$residuals, "\n")
cat("Standardized residuals:", result1$stdres, "\n")

write.csv(
  data.frame(
    statistic = result1$statistic,
    df = result1$parameter,
    p_value = result1$p.value
  ),
  "validation/expected/chisq_gof_test1.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 2: Goodness-of-fit with specified probabilities
cat("\nTest 2: Goodness-of-fit (Specified Probabilities)\n")
x2 <- c(10, 20, 30, 40)
probs2 <- c(0.1, 0.2, 0.3, 0.4)
cat("Observed counts:", x2, "\n")
cat("Expected probabilities:", probs2, "\n\n")

result2 <- chisq.test(x2, p = probs2)
print(result2)

cat("\nExpected values:", result2$expected, "\n")

write.csv(
  data.frame(
    statistic = result2$statistic,
    df = result2$parameter,
    p_value = result2$p.value
  ),
  "validation/expected/chisq_gof_test2.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 3: Test of independence (2x3 table)
cat("\nTest 3: Test of Independence (2x3 Table)\n")
# Gender x Party preference data
M <- as.table(rbind(
  c(762, 327, 468),  # Female: Democrat, Independent, Republican
  c(484, 239, 477)   # Male: Democrat, Independent, Republican
))
dimnames(M) <- list(
  Gender = c("Female", "Male"),
  Party = c("Democrat", "Independent", "Republican")
)
cat("Contingency table:\n")
print(M)

result3 <- chisq.test(M)
print(result3)

cat("\nExpected values:\n")
print(round(result3$expected, 4))

cat("\nPearson residuals:\n")
print(round(result3$residuals, 4))

cat("\nStandardized residuals:\n")
print(round(result3$stdres, 4))

write.csv(
  data.frame(
    statistic = result3$statistic,
    df = result3$parameter,
    p_value = result3$p.value
  ),
  "validation/expected/chisq_independence_test1.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 4: 2x2 table with Yates' correction
cat("\nTest 4: 2x2 Table with Yates' Correction\n")
M2 <- matrix(c(12, 7, 5, 16), nrow = 2, byrow = FALSE)
dimnames(M2) <- list(
  Row = c("A", "B"),
  Col = c("X", "Y")
)
cat("Contingency table:\n")
print(M2)

# With Yates' correction (default for 2x2)
cat("\nWith Yates' correction:\n")
result4a <- chisq.test(M2, correct = TRUE)
print(result4a)

# Without Yates' correction
cat("\nWithout Yates' correction:\n")
result4b <- chisq.test(M2, correct = FALSE)
print(result4b)

write.csv(
  data.frame(
    test = c("with_yates", "without_yates"),
    statistic = c(result4a$statistic, result4b$statistic),
    df = c(result4a$parameter, result4b$parameter),
    p_value = c(result4a$p.value, result4b$p.value)
  ),
  "validation/expected/chisq_2x2_test.csv",
  row.names = FALSE
)

cat("\n-------------------------------------------\n")

# Test Case 5: Fair die test
cat("\nTest 5: Fair Die Test\n")
die_rolls <- c(16, 18, 22, 14, 15, 15)  # 100 rolls
cat("Observed counts:", die_rolls, "\n")
cat("Expected under fair die:", rep(100/6, 6), "\n\n")

result5 <- chisq.test(die_rolls)
print(result5)

write.csv(
  data.frame(
    statistic = result5$statistic,
    df = result5$parameter,
    p_value = result5$p.value
  ),
  "validation/expected/chisq_die_test.csv",
  row.names = FALSE
)

cat("\n=== Validation Complete ===\n")
cat("Expected values saved to validation/expected/\n")
