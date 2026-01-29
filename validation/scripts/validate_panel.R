#!/usr/bin/env Rscript
#
# R Validation Script for Panel Methods (FE, RE, Hausman)
# Generates reference values for p2a-core validation tests
#
# Dependencies: plm
#

cat("=== Panel Methods Validation Script ===\n\n")

# Install packages if needed
if (!requireNamespace("plm", quietly = TRUE)) {
  install.packages("plm", repos = "https://cloud.r-project.org")
}

library(plm)

# =============================================================================
# Test 1: Grunfeld Dataset - Fixed Effects
# =============================================================================
cat("--- Test 1: Grunfeld Fixed Effects ---\n")
data(Grunfeld)

pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

fe_fit <- plm(inv ~ value + capital, data = pdata, model = "within")

cat("\nFixed Effects Coefficients:\n")
print(coef(fe_fit))

cat("\nFixed Effects Standard Errors:\n")
print(sqrt(diag(vcov(fe_fit))))

cat("\nFixed Effects Summary:\n")
fe_sum <- summary(fe_fit)
cat("  Within R-squared:", fe_sum$r.squared["rsq"], "\n")
cat("  Adj. R-squared:", fe_sum$r.squared["adjrsq"], "\n")
cat("  F-statistic:", fe_sum$fstatistic$statistic, "\n")
cat("  df1:", fe_sum$fstatistic$parameter["df1"], "\n")
cat("  df2:", fe_sum$fstatistic$parameter["df2"], "\n")

# =============================================================================
# Test 2: Grunfeld Dataset - Random Effects
# =============================================================================
cat("\n--- Test 2: Grunfeld Random Effects ---\n")

re_fit <- plm(inv ~ value + capital, data = pdata, model = "random")

cat("\nRandom Effects Coefficients:\n")
print(coef(re_fit))

cat("\nRandom Effects Standard Errors:\n")
print(sqrt(diag(vcov(re_fit))))

cat("\nRandom Effects Summary:\n")
re_sum <- summary(re_fit)
cat("  R-squared:", re_sum$r.squared["rsq"], "\n")
cat("  Adj. R-squared:", re_sum$r.squared["adjrsq"], "\n")

# Variance components
cat("\nVariance Components:\n")
cat("  sigma_u (idiosyncratic):", sqrt(re_sum$ercomp$sigma2["idios"]), "\n")
cat("  sigma_e (individual):", sqrt(re_sum$ercomp$sigma2["id"]), "\n")
cat("  theta:", re_sum$ercomp$theta, "\n")

# =============================================================================
# Test 3: Hausman Test
# =============================================================================
cat("\n--- Test 3: Hausman Test ---\n")

hausman_result <- phtest(fe_fit, re_fit)

cat("\nHausman Test Results:\n")
print(hausman_result)

cat("\nHausman Test Statistics:\n")
cat("  Chi-squared statistic:", hausman_result$statistic, "\n")
cat("  Degrees of freedom:", hausman_result$parameter, "\n")
cat("  p-value:", hausman_result$p.value, "\n")

# =============================================================================
# Test 4: Synthetic Panel with Known DGP (FE Required)
# =============================================================================
cat("\n--- Test 4: Synthetic Panel (Entity Effects Correlated with X) ---\n")

set.seed(42)
n_entities <- 100
n_periods <- 10
n_total <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)

# Entity effects CORRELATED with x (FE is consistent, RE is inconsistent)
alpha <- rnorm(n_entities)
x <- alpha[entity] + rnorm(n_total)  # x is correlated with entity effect!
y <- alpha[entity] + 2.0 * x + rnorm(n_total, 0, 0.5)

synth_data <- data.frame(
  y = y, x = x,
  entity = factor(entity),
  time = factor(time)
)

pdata_synth <- pdata.frame(synth_data, index = c("entity", "time"))

fe_synth <- plm(y ~ x, data = pdata_synth, model = "within")
re_synth <- plm(y ~ x, data = pdata_synth, model = "random")

cat("\nFE Coefficient (should be ~2.0):\n")
print(coef(fe_synth))

cat("\nRE Coefficient (biased due to correlation):\n")
print(coef(re_synth))

cat("\nHausman Test (should reject H0 -> use FE):\n")
hausman_synth <- phtest(fe_synth, re_synth)
cat("  Chi-squared:", hausman_synth$statistic, "\n")
cat("  p-value:", hausman_synth$p.value, "\n")

# =============================================================================
# Test 5: Synthetic Panel with Known DGP (RE Valid)
# =============================================================================
cat("\n--- Test 5: Synthetic Panel (Entity Effects UNCORRELATED with X) ---\n")

set.seed(42)
n_entities <- 100
n_periods <- 10
n_total <- n_entities * n_periods

entity <- rep(1:n_entities, each = n_periods)
time <- rep(1:n_periods, n_entities)

# Entity effects UNCORRELATED with x (both FE and RE are consistent)
alpha <- rnorm(n_entities)
x_uncorr <- rnorm(n_total)  # x is independent of entity effect
y_uncorr <- alpha[entity] + 2.0 * x_uncorr + rnorm(n_total, 0, 0.5)

synth_uncorr <- data.frame(
  y = y_uncorr, x = x_uncorr,
  entity = factor(entity),
  time = factor(time)
)

pdata_uncorr <- pdata.frame(synth_uncorr, index = c("entity", "time"))

fe_uncorr <- plm(y ~ x, data = pdata_uncorr, model = "within")
re_uncorr <- plm(y ~ x, data = pdata_uncorr, model = "random")

cat("\nFE Coefficient (should be ~2.0):\n")
print(coef(fe_uncorr))

cat("\nRE Coefficient (should also be ~2.0):\n")
print(coef(re_uncorr))

cat("\nHausman Test (should fail to reject H0 -> RE acceptable):\n")
hausman_uncorr <- phtest(fe_uncorr, re_uncorr)
cat("  Chi-squared:", hausman_uncorr$statistic, "\n")
cat("  p-value:", hausman_uncorr$p.value, "\n")

# =============================================================================
# Write expected values to CSV
# =============================================================================
cat("\n--- Writing expected values to CSV ---\n")

# Grunfeld FE results
grunfeld_fe <- data.frame(
  variable = names(coef(fe_fit)),
  coefficient = coef(fe_fit),
  std_error = sqrt(diag(vcov(fe_fit)))
)
write.csv(grunfeld_fe, "validation/expected/panel_fe_grunfeld.csv", row.names = FALSE)

# Grunfeld RE results
grunfeld_re <- data.frame(
  variable = names(coef(re_fit)),
  coefficient = coef(re_fit),
  std_error = sqrt(diag(vcov(re_fit)))
)
write.csv(grunfeld_re, "validation/expected/panel_re_grunfeld.csv", row.names = FALSE)

# Hausman test results
hausman_csv <- data.frame(
  chi2_statistic = hausman_result$statistic,
  df = hausman_result$parameter,
  p_value = hausman_result$p.value
)
write.csv(hausman_csv, "validation/expected/hausman_grunfeld.csv", row.names = FALSE)

cat("\nExpected values written to validation/expected/\n")
cat("=== Panel Methods Validation Script Complete ===\n")
