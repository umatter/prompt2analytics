#!/usr/bin/env Rscript
#
# R Validation Script for Regression Diagnostics & Methods
# Generates reference values for p2a-core validation tests
#
# Methods covered:
# - Quantile Regression (quantreg::rq)
# - GLS (nlme::gls)
# - Stepwise Selection (stats::step)
# - Breusch-Godfrey Test (lmtest::bgtest)
# - RESET Test (lmtest::resettest)
# - Harvey-Collier Test (lmtest::harvtest)
# - Wald Test (aod::wald.test)
# - HAC (sandwich::vcovHAC, NeweyWest)
# - Bootstrap (sandwich::vcovBS)
# - Driscoll-Kraay (sandwich::vcovPL)
# - Super Smoother (stats::supsmu)
# - Line (MASS::line)
#
# Dependencies: quantreg, nlme, lmtest, sandwich, aod, MASS
#

cat("=== Regression Diagnostics Validation Script ===\n\n")

# Install packages if needed
packages <- c("quantreg", "nlme", "lmtest", "sandwich", "aod", "MASS")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

library(quantreg)
library(nlme)
library(lmtest)
library(sandwich)
library(MASS)

# Create output directory for expected values
if (!dir.exists("validation/expected")) {
  dir.create("validation/expected", recursive = TRUE)
}

# =============================================================================
# Test 1: Quantile Regression
# =============================================================================
cat("\n=== Test 1: Quantile Regression ===\n")

set.seed(42)
n <- 100
x <- runif(n, 0, 10)
y <- 2 + 3 * x + rnorm(n, 0, 1 + 0.5 * x)  # heteroskedastic errors

# Median regression (tau = 0.5)
qr_median <- rq(y ~ x, tau = 0.5)
cat("\nMedian Regression (tau = 0.5):\n")
print(summary(qr_median))

# Multiple quantiles
qr_multi <- rq(y ~ x, tau = c(0.25, 0.5, 0.75))
cat("\nMultiple Quantiles (0.25, 0.5, 0.75):\n")
print(summary(qr_multi))

# Save results
qr_expected <- data.frame(
  tau = c(0.25, 0.5, 0.75),
  intercept = sapply(1:3, function(i) coef(qr_multi)[1, i]),
  slope = sapply(1:3, function(i) coef(qr_multi)[2, i])
)
write.csv(qr_expected, "validation/expected/quantreg_test.csv", row.names = FALSE)

cat("\nQuantile Regression Coefficients:\n")
print(qr_expected)

# =============================================================================
# Test 2: Generalized Least Squares
# =============================================================================
cat("\n=== Test 2: GLS ===\n")

set.seed(42)
n <- 50
t <- 1:n
# AR(1) error structure
rho <- 0.6
e <- numeric(n)
e[1] <- rnorm(1)
for (i in 2:n) {
  e[i] <- rho * e[i-1] + rnorm(1)
}
x <- runif(n, 0, 10)
y <- 5 + 2 * x + e

df_gls <- data.frame(y = y, x = x, t = t)

# GLS with AR(1) correlation
gls_ar1 <- gls(y ~ x, data = df_gls, correlation = corAR1(form = ~ t))
cat("\nGLS with AR(1) correlation:\n")
print(summary(gls_ar1))

# Extract AR(1) parameter
rho_est <- coef(gls_ar1$modelStruct$corStruct, unconstrained = FALSE)
cat(sprintf("\nEstimated rho: %.6f (true: %.6f)\n", rho_est, rho))

gls_expected <- data.frame(
  variable = c("(Intercept)", "x"),
  coefficient = coef(gls_ar1),
  std_error = sqrt(diag(vcov(gls_ar1))),
  rho_estimated = rho_est
)
write.csv(gls_expected, "validation/expected/gls_ar1_test.csv", row.names = FALSE)

# =============================================================================
# Test 3: Stepwise Selection
# =============================================================================
cat("\n=== Test 3: Stepwise Selection ===\n")

set.seed(42)
n <- 100
# True model: y = 2 + 3*x1 + 1.5*x2 (x3 is noise)
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)  # noise
y <- 2 + 3*x1 + 1.5*x2 + rnorm(n, 0, 0.5)

df_step <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)

# Full model
full_model <- lm(y ~ x1 + x2 + x3, data = df_step)

# Forward selection
null_model <- lm(y ~ 1, data = df_step)
step_forward <- step(null_model, scope = list(lower = null_model, upper = full_model),
                     direction = "forward", trace = 0)
cat("\nForward Selection Final Model:\n")
print(summary(step_forward))

# Backward selection
step_backward <- step(full_model, direction = "backward", trace = 0)
cat("\nBackward Selection Final Model:\n")
print(summary(step_backward))

# Both directions
step_both <- step(full_model, direction = "both", trace = 0)
cat("\nBidirectional Selection Final Model:\n")
print(summary(step_both))

step_expected <- data.frame(
  method = c("forward", "backward", "both"),
  aic = c(AIC(step_forward), AIC(step_backward), AIC(step_both)),
  n_vars = c(length(coef(step_forward)) - 1,
             length(coef(step_backward)) - 1,
             length(coef(step_both)) - 1),
  has_x1 = c("x1" %in% names(coef(step_forward)),
             "x1" %in% names(coef(step_backward)),
             "x1" %in% names(coef(step_both))),
  has_x2 = c("x2" %in% names(coef(step_forward)),
             "x2" %in% names(coef(step_backward)),
             "x2" %in% names(coef(step_both))),
  has_x3 = c("x3" %in% names(coef(step_forward)),
             "x3" %in% names(coef(step_backward)),
             "x3" %in% names(coef(step_both)))
)
write.csv(step_expected, "validation/expected/step_test.csv", row.names = FALSE)

cat("\nStepwise Selection Summary:\n")
print(step_expected)

# =============================================================================
# Test 4: Breusch-Godfrey Test
# =============================================================================
cat("\n=== Test 4: Breusch-Godfrey Test ===\n")

set.seed(42)
n <- 100
# Create AR(1) errors
e_ar <- numeric(n)
e_ar[1] <- rnorm(1)
for (i in 2:n) {
  e_ar[i] <- 0.5 * e_ar[i-1] + rnorm(1)
}
x <- rnorm(n)
y_autocor <- 2 + 3*x + e_ar

# Model with autocorrelated errors
model_autocor <- lm(y_autocor ~ x)

# BG test order 1
bg1 <- bgtest(model_autocor, order = 1)
cat("\nBreusch-Godfrey Test (order 1):\n")
print(bg1)

# BG test order 4
bg4 <- bgtest(model_autocor, order = 4)
cat("\nBreusch-Godfrey Test (order 4):\n")
print(bg4)

# F-type test
bg1_f <- bgtest(model_autocor, order = 1, type = "F")
cat("\nBreusch-Godfrey F-Test (order 1):\n")
print(bg1_f)

bg_expected <- data.frame(
  order = c(1, 4, 1),
  type = c("Chisq", "Chisq", "F"),
  statistic = c(bg1$statistic, bg4$statistic, bg1_f$statistic),
  df = c(bg1$parameter, bg4$parameter, paste(bg1_f$parameter[1], bg1_f$parameter[2], sep=",")),
  p_value = c(bg1$p.value, bg4$p.value, bg1_f$p.value)
)
write.csv(bg_expected, "validation/expected/bgtest_test.csv", row.names = FALSE)

cat("\nBG Test Summary:\n")
print(bg_expected)

# =============================================================================
# Test 5: RESET Test
# =============================================================================
cat("\n=== Test 5: RESET Test ===\n")

set.seed(42)
n <- 100
x <- runif(n, 1, 10)
# True model has quadratic term
y_quad <- 2 + 3*x + 0.5*x^2 + rnorm(n, 0, 2)

# Fit misspecified linear model
model_lin <- lm(y_quad ~ x)

# RESET test (should reject)
reset_fitted <- resettest(model_lin, power = 2:3, type = "fitted")
cat("\nRESET Test (fitted values, power 2:3):\n")
print(reset_fitted)

reset_regressor <- resettest(model_lin, power = 2:3, type = "regressor")
cat("\nRESET Test (regressors, power 2:3):\n")
print(reset_regressor)

reset_expected <- data.frame(
  type = c("fitted", "regressor"),
  statistic = c(reset_fitted$statistic, reset_regressor$statistic),
  df1 = c(reset_fitted$parameter[1], reset_regressor$parameter[1]),
  df2 = c(reset_fitted$parameter[2], reset_regressor$parameter[2]),
  p_value = c(reset_fitted$p.value, reset_regressor$p.value)
)
write.csv(reset_expected, "validation/expected/reset_test.csv", row.names = FALSE)

cat("\nRESET Test Summary:\n")
print(reset_expected)

# =============================================================================
# Test 6: Harvey-Collier Test
# =============================================================================
cat("\n=== Test 6: Harvey-Collier Test ===\n")

set.seed(42)
n <- 100
x <- seq(1, 10, length.out = n)
# Quadratic relationship (should fail linearity)
y_nonlin <- 5 + 2*x - 0.2*x^2 + rnorm(n, 0, 0.5)

model_nonlin <- lm(y_nonlin ~ x)

# Harvey-Collier test
hc_test <- harvtest(model_nonlin)
cat("\nHarvey-Collier Test:\n")
print(hc_test)

hc_expected <- data.frame(
  statistic = hc_test$statistic,
  df = hc_test$parameter,
  p_value = hc_test$p.value
)
write.csv(hc_expected, "validation/expected/harveycollier_test.csv", row.names = FALSE)

# =============================================================================
# Test 7: Wald Test
# =============================================================================
cat("\n=== Test 7: Wald Test ===\n")

set.seed(42)
n <- 100
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
y <- 2 + 3*x1 + 0.001*x2 + 1*x3 + rnorm(n, 0, 0.5)

model_wald <- lm(y ~ x1 + x2 + x3)
cat("\nModel for Wald Test:\n")
print(summary(model_wald))

# Test if x2 coefficient = 0 (single restriction)
wald_single <- aod::wald.test(
  b = coef(model_wald),
  Sigma = vcov(model_wald),
  Terms = 3  # x2 is the 3rd term
)
cat("\nWald Test (H0: x2 = 0):\n")
print(wald_single)

# Test if x2 and x3 both = 0 (multiple restrictions)
wald_multi <- aod::wald.test(
  b = coef(model_wald),
  Sigma = vcov(model_wald),
  Terms = c(3, 4)  # x2 and x3
)
cat("\nWald Test (H0: x2 = x3 = 0):\n")
print(wald_multi)

wald_expected <- data.frame(
  restriction = c("x2=0", "x2=x3=0"),
  chi2_statistic = c(wald_single$result$chi2["chi2"], wald_multi$result$chi2["chi2"]),
  df = c(wald_single$result$chi2["df"], wald_multi$result$chi2["df"]),
  p_value = c(wald_single$result$chi2["P(> Chi2)"], wald_multi$result$chi2["P(> Chi2)"])
)
write.csv(wald_expected, "validation/expected/wald_test.csv", row.names = FALSE)

cat("\nWald Test Summary:\n")
print(wald_expected)

# =============================================================================
# Test 8: HAC (Newey-West) Standard Errors
# =============================================================================
cat("\n=== Test 8: HAC (Newey-West) Standard Errors ===\n")

set.seed(42)
n <- 100
# Generate data with autocorrelated errors
e_hac <- numeric(n)
e_hac[1] <- rnorm(1)
for (i in 2:n) {
  e_hac[i] <- 0.6 * e_hac[i-1] + rnorm(1)
}
x <- rnorm(n)
y_hac <- 2 + 3*x + e_hac

model_hac <- lm(y_hac ~ x)

# Newey-West HAC
hac_nw <- NeweyWest(model_hac)
cat("\nNewey-West HAC Variance:\n")
print(hac_nw)

# vcovHAC with different kernels
hac_bartlett <- vcovHAC(model_hac, kernel = "Bartlett")
hac_parzen <- vcovHAC(model_hac, kernel = "Parzen")

hac_expected <- data.frame(
  kernel = c("Bartlett", "Parzen", "NeweyWest"),
  intercept_se = c(sqrt(hac_bartlett[1,1]), sqrt(hac_parzen[1,1]), sqrt(hac_nw[1,1])),
  x_se = c(sqrt(hac_bartlett[2,2]), sqrt(hac_parzen[2,2]), sqrt(hac_nw[2,2]))
)
write.csv(hac_expected, "validation/expected/hac_test.csv", row.names = FALSE)

cat("\nHAC Standard Errors:\n")
print(hac_expected)

# =============================================================================
# Test 9: Bootstrap Standard Errors
# =============================================================================
cat("\n=== Test 9: Bootstrap Standard Errors ===\n")

set.seed(42)
n <- 50  # smaller for speed
x <- rnorm(n)
y <- 2 + 3*x + rnorm(n, 0, 1 + abs(x))  # heteroskedastic

model_boot <- lm(y ~ x)

# Pairs bootstrap
boot_pairs <- vcovBS(model_boot, type = "xy", R = 200)
cat("\nPairs Bootstrap Variance:\n")
print(boot_pairs)

# Wild bootstrap
boot_wild <- vcovBS(model_boot, type = "wild", R = 200)
cat("\nWild Bootstrap Variance:\n")
print(boot_wild)

boot_expected <- data.frame(
  type = c("pairs", "wild"),
  intercept_se = c(sqrt(boot_pairs[1,1]), sqrt(boot_wild[1,1])),
  x_se = c(sqrt(boot_pairs[2,2]), sqrt(boot_wild[2,2]))
)
write.csv(boot_expected, "validation/expected/bootstrap_test.csv", row.names = FALSE)

cat("\nBootstrap Standard Errors:\n")
print(boot_expected)

# =============================================================================
# Test 10: Driscoll-Kraay Standard Errors
# =============================================================================
cat("\n=== Test 10: Driscoll-Kraay Standard Errors ===\n")

# Panel data structure
set.seed(42)
n_firms <- 10
n_periods <- 20
n_total <- n_firms * n_periods

firm <- rep(1:n_firms, each = n_periods)
time <- rep(1:n_periods, n_firms)
x <- rnorm(n_total)
# Firm fixed effects + correlated errors across firms
firm_fe <- rep(rnorm(n_firms), each = n_periods)
y <- 2 + 3*x + firm_fe + rnorm(n_total)

df_panel <- data.frame(y = y, x = x, firm = factor(firm), time = time)

# OLS on pooled data
model_pool <- lm(y ~ x, data = df_panel)

# Driscoll-Kraay SEs (panel-robust, accounts for cross-sectional dependence)
dk_vcov <- vcovPL(model_pool, cluster = ~ time, order.by = ~ time)
cat("\nDriscoll-Kraay Variance:\n")
print(dk_vcov)

dk_expected <- data.frame(
  variable = c("(Intercept)", "x"),
  coefficient = coef(model_pool),
  dk_se = sqrt(diag(dk_vcov)),
  ols_se = sqrt(diag(vcov(model_pool)))
)
write.csv(dk_expected, "validation/expected/driscoll_kraay_test.csv", row.names = FALSE)

cat("\nDriscoll-Kraay Standard Errors:\n")
print(dk_expected)

# =============================================================================
# Test 11: Super Smoother
# =============================================================================
cat("\n=== Test 11: Super Smoother ===\n")

set.seed(42)
n <- 100
x_ss <- sort(runif(n, 0, 2*pi))
y_ss <- sin(x_ss) + rnorm(n, 0, 0.3)

# Super smoother
ss_result <- supsmu(x_ss, y_ss)

# Save first 20 points for validation
supsmu_expected <- data.frame(
  x = head(ss_result$x, 20),
  y_smoothed = head(ss_result$y, 20),
  y_original = head(y_ss, 20)
)
write.csv(supsmu_expected, "validation/expected/supsmu_test.csv", row.names = FALSE)

cat("\nSuper Smoother Output (first 20 points):\n")
print(supsmu_expected)

# =============================================================================
# Test 12: Line (Resistant Line Fitting)
# =============================================================================
cat("\n=== Test 12: Line (Resistant Line Fitting) ===\n")

set.seed(42)
n <- 30
x_line <- 1:n
y_line <- 2 + 0.5*x_line + rnorm(n, 0, 0.5)
# Add some outliers
y_line[c(5, 15, 25)] <- y_line[c(5, 15, 25)] + c(10, -8, 12)

# Resistant line fitting
line_result <- line(x_line, y_line)
cat("\nResistant Line Coefficients:\n")
print(coef(line_result))

# OLS for comparison
ols_result <- lm(y_line ~ x_line)
cat("\nOLS Coefficients:\n")
print(coef(ols_result))

line_expected <- data.frame(
  method = c("line", "ols"),
  intercept = c(coef(line_result)[1], coef(ols_result)[1]),
  slope = c(coef(line_result)[2], coef(ols_result)[2])
)
write.csv(line_expected, "validation/expected/line_test.csv", row.names = FALSE)

cat("\nLine vs OLS (with outliers):\n")
print(line_expected)

# =============================================================================
# Summary
# =============================================================================
cat("\n=== Validation Script Complete ===\n")
cat("Expected values written to validation/expected/:\n")
cat("  - quantreg_test.csv\n")
cat("  - gls_ar1_test.csv\n")
cat("  - step_test.csv\n")
cat("  - bgtest_test.csv\n")
cat("  - reset_test.csv\n")
cat("  - harveycollier_test.csv\n")
cat("  - wald_test.csv\n")
cat("  - hac_test.csv\n")
cat("  - bootstrap_test.csv\n")
cat("  - driscoll_kraay_test.csv\n")
cat("  - supsmu_test.csv\n")
cat("  - line_test.csv\n")
