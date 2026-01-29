#!/usr/bin/env Rscript
#
# R Validation Script for OLS, Robust SEs, and Clustered SEs
# Generates reference values for p2a-core validation tests
#
# Dependencies: sandwich, lmtest, plm
#

cat("=== OLS Validation Script ===\n\n")

# Install packages if needed
required_packages <- c("sandwich", "lmtest")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(sandwich)
library(lmtest)

# =============================================================================
# Test 1: Longley Dataset - Classic OLS
# =============================================================================
cat("--- Test 1: Longley Dataset (Classic OLS) ---\n")
data(longley)

fit_longley <- lm(Employed ~ GNP.deflator + GNP + Unemployed + Armed.Forces + Population + Year,
                  data = longley)

cat("\nCoefficients:\n")
print(coef(fit_longley))

cat("\nStandard Errors:\n")
print(coef(summary(fit_longley))[, "Std. Error"])

cat("\nModel Summary:\n")
cat("  R-squared:", summary(fit_longley)$r.squared, "\n")
cat("  Adj. R-squared:", summary(fit_longley)$adj.r.squared, "\n")
cat("  F-statistic:", summary(fit_longley)$fstatistic[1], "\n")
cat("  df_resid:", summary(fit_longley)$df[2], "\n")
cat("  Residual SE:", summary(fit_longley)$sigma, "\n")

# =============================================================================
# Test 2: Simple Linear Regression with Known DGP
# =============================================================================
cat("\n--- Test 2: Simple Linear Regression (y = 2 + 1.5x + e) ---\n")
set.seed(42)
n <- 100
x <- runif(n, 0, 10)
y <- 2.0 + 1.5 * x + rnorm(n, 0, 0.5)  # Using 0.5 SD for reasonable noise

fit_simple <- lm(y ~ x)
cat("\nCoefficients:\n")
print(coef(fit_simple))

cat("\nStandard Errors:\n")
print(coef(summary(fit_simple))[, "Std. Error"])

cat("\nR-squared:", summary(fit_simple)$r.squared, "\n")

# =============================================================================
# Test 3: Robust Standard Errors (HC0-HC3) with mtcars
# =============================================================================
cat("\n--- Test 3: Robust Standard Errors (HC0-HC3) on mtcars ---\n")
data(mtcars)

fit_mtcars <- lm(mpg ~ wt + hp + disp, data = mtcars)

cat("\nCoefficients:\n")
print(coef(fit_mtcars))

cat("\nStandard SEs:\n")
print(sqrt(diag(vcov(fit_mtcars))))

cat("\nHC0 SEs:\n")
print(sqrt(diag(vcovHC(fit_mtcars, type = "HC0"))))

cat("\nHC1 SEs:\n")
print(sqrt(diag(vcovHC(fit_mtcars, type = "HC1"))))

cat("\nHC2 SEs:\n")
print(sqrt(diag(vcovHC(fit_mtcars, type = "HC2"))))

cat("\nHC3 SEs:\n")
print(sqrt(diag(vcovHC(fit_mtcars, type = "HC3"))))

# =============================================================================
# Test 4: Heteroskedastic Data for Robust SE Testing
# =============================================================================
cat("\n--- Test 4: Heteroskedastic Data ---\n")
set.seed(42)
n <- 100
x_het <- runif(n, -5, 5)
sigma <- 0.1 + 0.3 * abs(x_het)  # Variance increases with |x|
y_het <- 2.0 + 1.5 * x_het + rnorm(n, 0, sigma)

fit_het <- lm(y_het ~ x_het)

cat("\nCoefficients:\n")
print(coef(fit_het))

cat("\nStandard SEs (biased under heteroskedasticity):\n")
print(sqrt(diag(vcov(fit_het))))

cat("\nHC1 SEs (robust):\n")
print(sqrt(diag(vcovHC(fit_het, type = "HC1"))))

cat("\nHC3 SEs (most conservative):\n")
print(sqrt(diag(vcovHC(fit_het, type = "HC3"))))

# =============================================================================
# Test 5: Clustered Standard Errors
# =============================================================================
cat("\n--- Test 5: Clustered Standard Errors ---\n")
set.seed(42)
n_clusters <- 50
obs_per_cluster <- 10
n <- n_clusters * obs_per_cluster

cluster <- rep(1:n_clusters, each = obs_per_cluster)
u_cluster <- rnorm(n_clusters, 0, 1)  # Cluster-level shock
x_cl <- rnorm(n)
y_cl <- 1.0 + 2.0 * x_cl + u_cluster[cluster] + rnorm(n, 0, 0.5)

data_cl <- data.frame(y = y_cl, x = x_cl, cluster = factor(cluster))

fit_cl <- lm(y ~ x, data = data_cl)

cat("\nCoefficients:\n")
print(coef(fit_cl))

cat("\nStandard SEs (incorrect for clustered data):\n")
print(sqrt(diag(vcov(fit_cl))))

cat("\nCluster-Robust SEs:\n")
print(sqrt(diag(vcovCL(fit_cl, cluster = data_cl$cluster))))

cat("\nNumber of clusters:", n_clusters, "\n")

# =============================================================================
# Test 6: Grunfeld Data for Clustered SEs (Panel Data)
# =============================================================================
cat("\n--- Test 6: Grunfeld Dataset Clustered SEs ---\n")

# Create Grunfeld-like data manually (10 firms, 20 years)
set.seed(42)
n_firms <- 10
n_years <- 20
n_total <- n_firms * n_years

firm <- rep(1:n_firms, each = n_years)
year <- rep(1:n_years, n_firms)

# Investment, value, capital data (simplified)
value <- runif(n_total, 100, 500)
capital <- runif(n_total, 50, 300)
firm_effect <- rnorm(n_firms, 0, 50)[firm]
inv <- 50 + 0.11 * value + 0.31 * capital + firm_effect + rnorm(n_total, 0, 20)

grunfeld <- data.frame(
  firm = factor(firm),
  year = factor(year),
  inv = inv,
  value = value,
  capital = capital
)

fit_grunfeld <- lm(inv ~ value + capital, data = grunfeld)

cat("\nCoefficients:\n")
print(coef(fit_grunfeld))

cat("\nFirm-Clustered SEs:\n")
print(sqrt(diag(vcovCL(fit_grunfeld, cluster = grunfeld$firm))))

cat("\nYear-Clustered SEs:\n")
print(sqrt(diag(vcovCL(fit_grunfeld, cluster = grunfeld$year))))

# Two-way clustering (firm + year)
cat("\nTwo-Way Clustered SEs (firm + year):\n")
se_twoway <- sqrt(diag(vcovCL(fit_grunfeld, cluster = ~ firm + year, multi0 = TRUE)))
print(se_twoway)

# =============================================================================
# Write expected values to CSV
# =============================================================================
cat("\n--- Writing expected values to CSV ---\n")

# Longley results
longley_results <- data.frame(
  variable = names(coef(fit_longley)),
  coefficient = coef(fit_longley),
  std_error = coef(summary(fit_longley))[, "Std. Error"],
  t_stat = coef(summary(fit_longley))[, "t value"],
  p_value = coef(summary(fit_longley))[, "Pr(>|t|)"]
)
write.csv(longley_results, "validation/expected/ols_longley.csv", row.names = FALSE)

# mtcars HC results
mtcars_hc_results <- data.frame(
  variable = names(coef(fit_mtcars)),
  coefficient = coef(fit_mtcars),
  se_standard = sqrt(diag(vcov(fit_mtcars))),
  se_hc0 = sqrt(diag(vcovHC(fit_mtcars, type = "HC0"))),
  se_hc1 = sqrt(diag(vcovHC(fit_mtcars, type = "HC1"))),
  se_hc2 = sqrt(diag(vcovHC(fit_mtcars, type = "HC2"))),
  se_hc3 = sqrt(diag(vcovHC(fit_mtcars, type = "HC3")))
)
write.csv(mtcars_hc_results, "validation/expected/ols_robust_se_mtcars.csv", row.names = FALSE)

cat("\nExpected values written to validation/expected/\n")
cat("=== OLS Validation Script Complete ===\n")
