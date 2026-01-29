# R Validation Script for Time Series Methods (Phase 6)
# Generates expected values for Rust validation tests

library(stats)
library(vars)
library(lmtest)

cat("===========================================\n")
cat("Phase 6: Time Series Validation\n")
cat("===========================================\n\n")

# Use deterministic seed for reproducibility
set.seed(42)

# ============================================
# 1. VAR Model Validation
# ============================================
cat("1. VAR Model Validation\n")
cat("------------------------\n")

# Create bivariate time series
n <- 100
e1 <- rnorm(n, 0, 1)
e2 <- rnorm(n, 0, 1)

y1 <- numeric(n)
y2 <- numeric(n)
y1[1] <- e1[1]
y2[1] <- e2[1]

# VAR(1) DGP: y1_t = 0.5*y1_{t-1} + 0.3*y2_{t-1} + e1
#             y2_t = 0.2*y1_{t-1} + 0.4*y2_{t-1} + e2
for (t in 2:n) {
  y1[t] <- 0.5 * y1[t-1] + 0.3 * y2[t-1] + e1[t]
  y2[t] <- 0.2 * y1[t-1] + 0.4 * y2[t-1] + e2[t]
}

var_data <- data.frame(y1 = y1, y2 = y2)
var_model <- VAR(var_data, p = 1, type = "const")

cat("VAR(1) model fit:\n")
cat("Coefficients y1 equation:\n")
print(coef(var_model)$y1)
cat("\nCoefficients y2 equation:\n")
print(coef(var_model)$y2)
cat("\nAIC:", AIC(var_model), "\n")
cat("BIC:", BIC(var_model), "\n\n")

# ============================================
# 2. Granger Causality Validation
# ============================================
cat("2. Granger Causality Validation\n")
cat("--------------------------------\n")

# Test whether y2 Granger-causes y1
granger_result <- grangertest(y1 ~ y2, order = 2, data = var_data)
cat("Granger test: y2 -> y1\n")
cat("F statistic:", granger_result$F[2], "\n")
cat("p-value:", granger_result$`Pr(>F)`[2], "\n\n")

# Test whether y1 Granger-causes y2
granger_result2 <- grangertest(y2 ~ y1, order = 2, data = var_data)
cat("Granger test: y1 -> y2\n")
cat("F statistic:", granger_result2$F[2], "\n")
cat("p-value:", granger_result2$`Pr(>F)`[2], "\n\n")

# ============================================
# 3. AR Model Validation
# ============================================
cat("3. AR Model Validation\n")
cat("-----------------------\n")

# Generate AR(2) process
set.seed(42)
ar_data <- arima.sim(n = 100, model = list(ar = c(0.7, -0.2)))

# Fit AR using Yule-Walker
ar_yw <- ar(ar_data, method = "yule-walker")
cat("AR (Yule-Walker) results:\n")
cat("Order selected:", ar_yw$order, "\n")
cat("AR coefficients:", ar_yw$ar, "\n")
cat("Innovation variance:", ar_yw$var.pred, "\n\n")

# Fit AR using Burg
ar_burg <- ar(ar_data, method = "burg")
cat("AR (Burg) results:\n")
cat("Order selected:", ar_burg$order, "\n")
cat("AR coefficients:", ar_burg$ar, "\n")
cat("Innovation variance:", ar_burg$var.pred, "\n\n")

# Fit AR using OLS
ar_ols <- ar(ar_data, method = "ols")
cat("AR (OLS) results:\n")
cat("Order selected:", ar_ols$order, "\n")
cat("AR coefficients:", ar_ols$ar, "\n")
cat("Innovation variance:", ar_ols$var.pred, "\n\n")

# ============================================
# 4. ARIMA Model Validation
# ============================================
cat("4. ARIMA Model Validation\n")
cat("-------------------------\n")

# Use deterministic data for ARIMA
set.seed(42)
arima_data <- arima.sim(n = 100, model = list(ar = c(0.8, -0.3), ma = c(0.4)))

# Fit ARIMA(2,0,1)
arima_fit <- arima(arima_data, order = c(2, 0, 1))
cat("ARIMA(2,0,1) results:\n")
cat("AR coefficients:", coef(arima_fit)[c("ar1", "ar2")], "\n")
cat("MA coefficient:", coef(arima_fit)["ma1"], "\n")
cat("Intercept:", coef(arima_fit)["intercept"], "\n")
cat("AIC:", AIC(arima_fit), "\n\n")

# ============================================
# 5. Impulse Response Function Validation
# ============================================
cat("5. IRF Validation\n")
cat("-----------------\n")

irf_result <- irf(var_model, impulse = "y1", response = "y2", n.ahead = 5, ortho = TRUE)
cat("Orthogonalized IRF (y1 shock -> y2 response):\n")
cat("Steps 0-5:", irf_result$irf$y1[, "y2"], "\n\n")

# ============================================
# 6. GARCH Validation
# ============================================
cat("6. GARCH Validation\n")
cat("-------------------\n")

# Note: Using fGarch or rugarch would be needed for full validation
# For now, just document the expected structure
cat("GARCH(1,1) typical structure:\n")
cat("omega > 0, alpha >= 0, beta >= 0\n")
cat("Persistence = alpha + beta < 1 for stationarity\n\n")

# ============================================
# 7. Changepoint Detection Validation
# ============================================
cat("7. Changepoint Detection Validation\n")
cat("------------------------------------\n")

# Create data with clear changepoint
set.seed(42)
cp_data <- c(rnorm(50, mean = 0, sd = 1), rnorm(50, mean = 5, sd = 1))

# Note: changepoint package would be used for validation
cat("Test data: 50 points at mean=0, 50 points at mean=5\n")
cat("Expected changepoint: around index 50\n")
cat("Segment 1 mean:", mean(cp_data[1:50]), "\n")
cat("Segment 2 mean:", mean(cp_data[51:100]), "\n\n")

# ============================================
# Summary
# ============================================
cat("===========================================\n")
cat("Validation Reference Values Generated\n")
cat("===========================================\n")
cat("\nKey test patterns:\n")
cat("- VAR: Check coefficients match R's VAR()\n")
cat("- Granger: F-stat and p-value should be close\n")
cat("- AR: Order selection and coefficients match\n")
cat("- ARIMA: Coefficients and AIC match\n")
cat("- IRF: Response values match across horizons\n")
cat("- GARCH: Stationarity constraint satisfied\n")
cat("- Changepoint: Location detected correctly\n")
