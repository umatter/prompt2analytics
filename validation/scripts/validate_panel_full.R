#!/usr/bin/env Rscript
#
# R Validation Script for Full Panel Methods
# Generates reference values for p2a-core validation tests
#
# Methods covered:
# - Fixed Effects (plm model="within")
# - Random Effects (plm model="random")
# - Hausman Test (phtest)
# - Arellano-Bond GMM (pgmm)
# - Panel GLS - FEGLS (pggls model="within")
# - Panel GLS - Pooled (pggls model="pooling")
# - PVCM Within (pvcm model="within")
# - PVCM Random (pvcm model="random")
# - Mean Group/PMG (pmg)
#
# Dependencies: plm
#

cat("=== Full Panel Methods Validation Script ===\n\n")

# Install packages if needed
packages <- c("plm")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

library(plm)

# Create output directory for expected values
if (!dir.exists("validation/expected")) {
  dir.create("validation/expected", recursive = TRUE)
}

# =============================================================================
# Test Data: Grunfeld Investment Data (Classic Panel Dataset)
# =============================================================================
cat("--- Loading Grunfeld Dataset ---\n")
data(Grunfeld)
pdata <- pdata.frame(Grunfeld, index = c("firm", "year"))

cat(sprintf("Dataset: %d observations, %d firms, %d years\n",
            nrow(Grunfeld), length(unique(Grunfeld$firm)), length(unique(Grunfeld$year))))

# =============================================================================
# Test 1: Fixed Effects (Within)
# =============================================================================
cat("\n=== Test 1: Fixed Effects ===\n")

fe_result <- plm(inv ~ value + capital, data = pdata, model = "within")
fe_summary <- summary(fe_result)

cat("Coefficients:\n")
print(coef(fe_result))

cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(fe_result))))

cat("\nModel Statistics:\n")
cat(sprintf("  Within R-squared: %.6f\n", fe_summary$r.squared["rsq"]))
cat(sprintf("  Adj. R-squared: %.6f\n", fe_summary$r.squared["adjrsq"]))
cat(sprintf("  F-statistic: %.4f\n", fe_summary$fstatistic$statistic))
cat(sprintf("  df1: %d, df2: %d\n",
            fe_summary$fstatistic$parameter["df1"],
            fe_summary$fstatistic$parameter["df2"]))

# Save expected values
fe_expected <- data.frame(
  variable = names(coef(fe_result)),
  coefficient = as.numeric(coef(fe_result)),
  std_error = as.numeric(sqrt(diag(vcov(fe_result)))),
  r_squared = fe_summary$r.squared["rsq"],
  n_obs = nrow(Grunfeld),
  n_groups = length(unique(Grunfeld$firm))
)
write.csv(fe_expected, "validation/expected/panel_fe_full.csv", row.names = FALSE)

# =============================================================================
# Test 2: Random Effects (GLS)
# =============================================================================
cat("\n=== Test 2: Random Effects ===\n")

re_result <- plm(inv ~ value + capital, data = pdata, model = "random")
re_summary <- summary(re_result)

cat("Coefficients:\n")
print(coef(re_result))

cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(re_result))))

cat("\nVariance Components:\n")
cat(sprintf("  sigma_u (individual): %.6f\n", sqrt(re_summary$ercomp$sigma2["id"])))
cat(sprintf("  sigma_e (idiosyncratic): %.6f\n", sqrt(re_summary$ercomp$sigma2["idios"])))
cat(sprintf("  theta: %.6f\n", re_summary$ercomp$theta))

# Save expected values
re_expected <- data.frame(
  variable = names(coef(re_result)),
  coefficient = as.numeric(coef(re_result)),
  std_error = as.numeric(sqrt(diag(vcov(re_result)))),
  sigma_u = sqrt(re_summary$ercomp$sigma2["id"]),
  sigma_e = sqrt(re_summary$ercomp$sigma2["idios"]),
  theta = re_summary$ercomp$theta
)
write.csv(re_expected, "validation/expected/panel_re_full.csv", row.names = FALSE)

# =============================================================================
# Test 3: Hausman Test
# =============================================================================
cat("\n=== Test 3: Hausman Test ===\n")

hausman_result <- phtest(fe_result, re_result)

cat(sprintf("Chi-squared statistic: %.6f\n", hausman_result$statistic))
cat(sprintf("Degrees of freedom: %d\n", hausman_result$parameter))
cat(sprintf("p-value: %.6f\n", hausman_result$p.value))

hausman_expected <- data.frame(
  chi2_statistic = as.numeric(hausman_result$statistic),
  df = hausman_result$parameter,
  p_value = hausman_result$p.value
)
write.csv(hausman_expected, "validation/expected/hausman_full.csv", row.names = FALSE)

# =============================================================================
# Test 4: Arellano-Bond GMM (Dynamic Panel)
# =============================================================================
cat("\n=== Test 4: Arellano-Bond GMM ===\n")

# Use Grunfeld for dynamic panel (inv depends on lagged inv)
ab_result <- tryCatch({
  pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99),
       data = pdata,
       effect = "twoways",
       model = "onestep")
}, error = function(e) {
  cat("Arellano-Bond with twoways failed, trying individual effects only\n")
  pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99),
       data = pdata,
       effect = "individual",
       model = "onestep")
})

ab_summary <- summary(ab_result)

cat("\nCoefficients:\n")
print(coef(ab_result))

cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(ab_result))))

# Sargan test
cat("\nSargan Test:\n")
cat(sprintf("  Statistic: %.4f\n", ab_summary$sargan$statistic))
cat(sprintf("  df: %d\n", ab_summary$sargan$parameter))
cat(sprintf("  p-value: %.4f\n", ab_summary$sargan$p.value))

# AR tests
cat("\nAutocorrelation Tests:\n")
cat(sprintf("  AR(1) z: %.4f, p: %.4f\n",
            ab_summary$m1$statistic, ab_summary$m1$p.value))
cat(sprintf("  AR(2) z: %.4f, p: %.4f\n",
            ab_summary$m2$statistic, ab_summary$m2$p.value))

# Save expected values
ab_expected <- data.frame(
  variable = names(coef(ab_result)),
  coefficient = as.numeric(coef(ab_result)),
  std_error = as.numeric(sqrt(diag(vcov(ab_result)))),
  sargan_stat = ab_summary$sargan$statistic,
  sargan_df = ab_summary$sargan$parameter,
  sargan_p = ab_summary$sargan$p.value,
  ar1_z = ab_summary$m1$statistic,
  ar1_p = ab_summary$m1$p.value,
  ar2_z = ab_summary$m2$statistic,
  ar2_p = ab_summary$m2$p.value
)
write.csv(ab_expected, "validation/expected/arellano_bond_full.csv", row.names = FALSE)

# =============================================================================
# Test 5: Two-Step GMM
# =============================================================================
cat("\n=== Test 5: Two-Step GMM ===\n")

gmm2_result <- tryCatch({
  pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99),
       data = pdata,
       effect = "individual",
       model = "twosteps")
}, error = function(e) {
  cat("Two-step GMM failed:", e$message, "\n")
  NULL
})

if (!is.null(gmm2_result)) {
  gmm2_summary <- summary(gmm2_result)

  cat("\nCoefficients (Two-step):\n")
  print(coef(gmm2_result))

  cat("\nStandard Errors (Two-step):\n")
  print(sqrt(diag(vcov(gmm2_result))))

  gmm2_expected <- data.frame(
    variable = names(coef(gmm2_result)),
    coefficient = as.numeric(coef(gmm2_result)),
    std_error = as.numeric(sqrt(diag(vcov(gmm2_result)))),
    sargan_stat = gmm2_summary$sargan$statistic,
    sargan_p = gmm2_summary$sargan$p.value
  )
  write.csv(gmm2_expected, "validation/expected/gmm_twostep_full.csv", row.names = FALSE)
}

# =============================================================================
# Test 6: Panel GLS - Fixed Effects (FEGLS/pggls within)
# =============================================================================
cat("\n=== Test 6: Panel GLS - Fixed Effects ===\n")

fegls_result <- tryCatch({
  pggls(inv ~ value + capital, data = pdata, model = "within")
}, error = function(e) {
  cat("FEGLS failed:", e$message, "\n")
  NULL
})

if (!is.null(fegls_result)) {
  fegls_summary <- summary(fegls_result)

  cat("\nCoefficients:\n")
  print(coef(fegls_result))

  cat("\nStandard Errors:\n")
  print(sqrt(diag(vcov(fegls_result))))

  fegls_expected <- data.frame(
    variable = names(coef(fegls_result)),
    coefficient = as.numeric(coef(fegls_result)),
    std_error = as.numeric(sqrt(diag(vcov(fegls_result))))
  )
  write.csv(fegls_expected, "validation/expected/fegls_full.csv", row.names = FALSE)
}

# =============================================================================
# Test 7: Panel GLS - Pooled (pggls pooling)
# =============================================================================
cat("\n=== Test 7: Panel GLS - Pooled ===\n")

pggls_pool_result <- tryCatch({
  pggls(inv ~ value + capital, data = pdata, model = "pooling")
}, error = function(e) {
  cat("Pooled GLS failed:", e$message, "\n")
  NULL
})

if (!is.null(pggls_pool_result)) {
  pggls_pool_summary <- summary(pggls_pool_result)

  cat("\nCoefficients:\n")
  print(coef(pggls_pool_result))

  cat("\nStandard Errors:\n")
  print(sqrt(diag(vcov(pggls_pool_result))))

  pggls_pool_expected <- data.frame(
    variable = names(coef(pggls_pool_result)),
    coefficient = as.numeric(coef(pggls_pool_result)),
    std_error = as.numeric(sqrt(diag(vcov(pggls_pool_result))))
  )
  write.csv(pggls_pool_expected, "validation/expected/pggls_pooling_full.csv", row.names = FALSE)
}

# =============================================================================
# Test 8: PVCM Within (Variable Coefficients Model)
# =============================================================================
cat("\n=== Test 8: PVCM Within ===\n")

pvcm_within_result <- tryCatch({
  pvcm(inv ~ value + capital, data = pdata, model = "within")
}, error = function(e) {
  cat("PVCM Within failed:", e$message, "\n")
  NULL
})

if (!is.null(pvcm_within_result)) {
  pvcm_within_summary <- summary(pvcm_within_result)

  cat("\nIndividual Coefficients:\n")
  ind_coefs <- coef(pvcm_within_result)
  print(head(ind_coefs, 3))

  # For pvcm within, coefficients are individual-specific (N x K matrix)
  # Average across individuals for overall coefficients
  avg_coefs <- colMeans(ind_coefs)
  cat("\nAverage Coefficients:\n")
  print(avg_coefs)

  # Save individual coefficients
  write.csv(ind_coefs, "validation/expected/pvcm_within_individual.csv", row.names = TRUE)

  # Save average coefficients
  pvcm_within_expected <- data.frame(
    variable = names(avg_coefs),
    coefficient = as.numeric(avg_coefs),
    std_error = apply(ind_coefs, 2, sd) / sqrt(nrow(ind_coefs))
  )
  write.csv(pvcm_within_expected, "validation/expected/pvcm_within_full.csv", row.names = FALSE)
}

# =============================================================================
# Test 9: PVCM Random (Swamy estimator)
# =============================================================================
cat("\n=== Test 9: PVCM Random ===\n")

pvcm_random_result <- tryCatch({
  pvcm(inv ~ value + capital, data = pdata, model = "random")
}, error = function(e) {
  cat("PVCM Random failed:", e$message, "\n")
  NULL
})

if (!is.null(pvcm_random_result)) {
  pvcm_random_summary <- summary(pvcm_random_result)

  cat("\nGLS Coefficients (Swamy):\n")
  print(coef(pvcm_random_result))

  cat("\nStandard Errors:\n")
  print(sqrt(diag(vcov(pvcm_random_result))))

  pvcm_random_expected <- data.frame(
    variable = names(coef(pvcm_random_result)),
    coefficient = as.numeric(coef(pvcm_random_result)),
    std_error = as.numeric(sqrt(diag(vcov(pvcm_random_result))))
  )
  write.csv(pvcm_random_expected, "validation/expected/pvcm_random_full.csv", row.names = FALSE)
}

# =============================================================================
# Test 10: Mean Group Estimator (PMG)
# =============================================================================
cat("\n=== Test 10: Mean Group Estimator ===\n")

# PMG requires time series dimension - use simpler approach
# Mean group is average of individual OLS estimates
# plm::pmg() is available in some versions

pmg_result <- tryCatch({
  pmg(inv ~ value + capital, data = pdata, model = "mg")
}, error = function(e) {
  cat("PMG function not available, computing manually...\n")
  # Manual mean group estimation
  firms <- unique(pdata$firm)
  all_coefs <- list()

  for (f in firms) {
    firm_data <- pdata[pdata$firm == f, ]
    if (nrow(firm_data) > 3) {  # Need enough obs
      fit <- lm(inv ~ value + capital, data = firm_data)
      all_coefs[[as.character(f)]] <- coef(fit)
    }
  }

  # Average coefficients
  coef_mat <- do.call(rbind, all_coefs)
  mg_coef <- colMeans(coef_mat)
  mg_se <- apply(coef_mat, 2, sd) / sqrt(length(all_coefs))

  list(coefficients = mg_coef, std_errors = mg_se, individual = coef_mat)
})

if (!is.null(pmg_result)) {
  if (inherits(pmg_result, "list")) {
    cat("\nMean Group Coefficients:\n")
    print(pmg_result$coefficients)
    cat("\nStandard Errors:\n")
    print(pmg_result$std_errors)

    pmg_expected <- data.frame(
      variable = names(pmg_result$coefficients),
      coefficient = as.numeric(pmg_result$coefficients),
      std_error = as.numeric(pmg_result$std_errors)
    )
    write.csv(pmg_expected, "validation/expected/pmg_full.csv", row.names = FALSE)
    write.csv(pmg_result$individual, "validation/expected/pmg_individual.csv", row.names = TRUE)
  } else {
    cat("\nMean Group Coefficients:\n")
    print(coef(pmg_result))

    pmg_expected <- data.frame(
      variable = names(coef(pmg_result)),
      coefficient = as.numeric(coef(pmg_result)),
      std_error = as.numeric(sqrt(diag(vcov(pmg_result))))
    )
    write.csv(pmg_expected, "validation/expected/pmg_full.csv", row.names = FALSE)
  }
}

# =============================================================================
# Synthetic Panel Test: Known DGP for Precise Validation
# =============================================================================
cat("\n=== Synthetic Panel Test (Known DGP) ===\n")

set.seed(42)
n_firms <- 50
n_years <- 10
n_total <- n_firms * n_years

# Generate panel structure
firm_ids <- rep(1:n_firms, each = n_years)
year_ids <- rep(1:n_years, n_firms)

# True parameters
beta_value_true <- 0.11
beta_capital_true <- 0.31

# Fixed effects (firm-specific)
firm_fe <- rnorm(n_firms, mean = 50, sd = 100)

# Regressors
value <- abs(rnorm(n_total, mean = 2000, sd = 1500))
capital <- abs(rnorm(n_total, mean = 150, sd = 100))

# Generate y with known coefficients
y <- firm_fe[firm_ids] + beta_value_true * value + beta_capital_true * capital + rnorm(n_total, 0, 50)

synth_panel <- data.frame(
  firm = factor(firm_ids),
  year = year_ids,
  inv = y,
  value = value,
  capital = capital
)

pdata_synth <- pdata.frame(synth_panel, index = c("firm", "year"))

# Test FE on synthetic data
fe_synth <- plm(inv ~ value + capital, data = pdata_synth, model = "within")

cat("\nSynthetic Panel Fixed Effects:\n")
cat(sprintf("  True beta_value: %.4f\n", beta_value_true))
cat(sprintf("  Estimated beta_value: %.4f\n", coef(fe_synth)["value"]))
cat(sprintf("  True beta_capital: %.4f\n", beta_capital_true))
cat(sprintf("  Estimated beta_capital: %.4f\n", coef(fe_synth)["capital"]))

synth_fe_expected <- data.frame(
  variable = c("value", "capital"),
  true_coef = c(beta_value_true, beta_capital_true),
  estimated_coef = as.numeric(coef(fe_synth)),
  std_error = as.numeric(sqrt(diag(vcov(fe_synth))))
)
write.csv(synth_fe_expected, "validation/expected/panel_fe_synthetic.csv", row.names = FALSE)

# =============================================================================
# Summary
# =============================================================================
cat("\n=== Validation Script Complete ===\n")
cat("Expected values written to validation/expected/:\n")
cat("  - panel_fe_full.csv\n")
cat("  - panel_re_full.csv\n")
cat("  - hausman_full.csv\n")
cat("  - arellano_bond_full.csv\n")
cat("  - gmm_twostep_full.csv\n")
cat("  - fegls_full.csv\n")
cat("  - pggls_pooling_full.csv\n")
cat("  - pvcm_within_full.csv\n")
cat("  - pvcm_random_full.csv\n")
cat("  - pmg_full.csv\n")
cat("  - panel_fe_synthetic.csv\n")
