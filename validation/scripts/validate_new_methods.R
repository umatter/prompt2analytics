#!/usr/bin/env Rscript
# Validation script for newly implemented methods
# Compares R results with expected Rust outputs

library(lmtest)
library(sandwich)
library(nnet)
library(MASS)

cat("============================================================\n")
cat("Validation of New Statistical Methods\n")
cat("============================================================\n\n")

# ==============================================================================
# 1. Breusch-Godfrey Test (bgtest)
# ==============================================================================
cat("1. BREUSCH-GODFREY TEST (bgtest)\n")
cat("----------------------------------------\n")

set.seed(123)
n <- 100
x <- 1:n
e <- numeric(n)
e[1] <- rnorm(1)
for(i in 2:n) {
  e[i] <- 0.7 * e[i-1] + rnorm(1)  # AR(1) errors with rho=0.7
}
y <- 2 + 0.5 * x + e

model <- lm(y ~ x)
bg_result <- bgtest(model, order = 1)

cat("R lmtest::bgtest results:\n")
cat(sprintf("  LM statistic: %.6f\n", bg_result$statistic))
cat(sprintf("  df: %d\n", bg_result$parameter))
cat(sprintf("  p-value: %.10f\n", bg_result$p.value))
cat("\n")

# ==============================================================================
# 2. RESET Test (resettest)
# ==============================================================================
cat("2. RESET TEST (resettest)\n")
cat("----------------------------------------\n")

set.seed(456)
x <- runif(50, 1, 10)
y <- 2 + 0.5 * x + 0.1 * x^2 + rnorm(50, sd = 0.5)  # Quadratic relationship

model <- lm(y ~ x)
reset_result <- resettest(model, power = 2:3)

cat("R lmtest::resettest results:\n")
cat(sprintf("  RESET statistic: %.6f\n", reset_result$statistic))
cat(sprintf("  df1: %d, df2: %d\n", reset_result$parameter[1], reset_result$parameter[2]))
cat(sprintf("  p-value: %.10f\n", reset_result$p.value))
cat("\n")

# ==============================================================================
# 3. Wald Test (waldtest)
# ==============================================================================
cat("3. WALD TEST (waldtest)\n")
cat("----------------------------------------\n")

set.seed(789)
x1 <- rnorm(100)
x2 <- rnorm(100)
y <- 1 + 2*x1 + 0.5*x2 + rnorm(100)

full_model <- lm(y ~ x1 + x2)
reduced_model <- lm(y ~ x1)
wald_result <- waldtest(reduced_model, full_model)

cat("R lmtest::waldtest results:\n")
cat(sprintf("  F statistic: %.6f\n", wald_result$F[2]))
cat(sprintf("  df1: %d, df2: %d\n", abs(wald_result$Df[2]), wald_result$Res.Df[2]))
cat(sprintf("  p-value: %.10f\n", wald_result$`Pr(>F)`[2]))
cat("\n")

# ==============================================================================
# 4. HAC Standard Errors (vcovHAC / Newey-West)
# ==============================================================================
cat("4. HAC STANDARD ERRORS (vcovHAC)\n")
cat("----------------------------------------\n")

set.seed(101)
n <- 100
x <- rnorm(n)
e <- numeric(n)
e[1] <- rnorm(1)
for(i in 2:n) e[i] <- 0.5 * e[i-1] + rnorm(1)
y <- 1 + 2*x + e

model <- lm(y ~ x)
hac_vcov <- vcovHAC(model)
hac_se <- sqrt(diag(hac_vcov))
ols_se <- sqrt(diag(vcov(model)))

cat("R sandwich::vcovHAC results:\n")
cat(sprintf("  OLS SE (Intercept): %.6f, HAC SE: %.6f\n", ols_se[1], hac_se[1]))
cat(sprintf("  OLS SE (x): %.6f, HAC SE: %.6f\n", ols_se[2], hac_se[2]))
cat("\n")

# Newey-West specific
nw_vcov <- NeweyWest(model, lag = 4)
nw_se <- sqrt(diag(nw_vcov))
cat("R sandwich::NeweyWest (lag=4) results:\n")
cat(sprintf("  NW SE (Intercept): %.6f\n", nw_se[1]))
cat(sprintf("  NW SE (x): %.6f\n", nw_se[2]))
cat("\n")

# ==============================================================================
# 5. Granger Causality Test (grangertest)
# ==============================================================================
cat("5. GRANGER CAUSALITY TEST (grangertest)\n")
cat("----------------------------------------\n")

set.seed(202)
n <- 100
x <- cumsum(rnorm(n))
y <- numeric(n)
y[1] <- rnorm(1)
for(i in 2:n) y[i] <- 0.3 * x[i-1] + 0.5 * y[i-1] + rnorm(1)

granger_result <- grangertest(y ~ x, order = 2)

cat("R lmtest::grangertest results:\n")
cat(sprintf("  F statistic: %.6f\n", granger_result$F[2]))
cat(sprintf("  df1: %d, df2: %d\n", abs(granger_result$Df[2]), granger_result$Res.Df[2]))
cat(sprintf("  p-value: %.10f\n", granger_result$`Pr(>F)`[2]))
cat("\n")

# ==============================================================================
# 6. Multinomial Logit (multinom)
# ==============================================================================
cat("6. MULTINOMIAL LOGIT (multinom)\n")
cat("----------------------------------------\n")

set.seed(404)
n <- 150
x <- rnorm(n)
# Generate categorical outcome
probs <- cbind(
  exp(0),
  exp(0.5 + 1*x),
  exp(1 + 2*x)
)
probs <- probs / rowSums(probs)
y <- apply(probs, 1, function(p) sample(c("A", "B", "C"), 1, prob = p))

data <- data.frame(y = factor(y), x = x)
multinom_result <- multinom(y ~ x, data = data, trace = FALSE)

cat("R nnet::multinom results:\n")
cat("  Coefficients (B vs A):\n")
cat(sprintf("    Intercept: %.6f\n", coef(multinom_result)[1, 1]))
cat(sprintf("    x: %.6f\n", coef(multinom_result)[1, 2]))
cat("  Coefficients (C vs A):\n")
cat(sprintf("    Intercept: %.6f\n", coef(multinom_result)[2, 1]))
cat(sprintf("    x: %.6f\n", coef(multinom_result)[2, 2]))
cat(sprintf("  Log-Likelihood: %.6f\n", logLik(multinom_result)))
cat(sprintf("  AIC: %.6f\n", AIC(multinom_result)))
cat("\n")

# ==============================================================================
# 7. Ordered Logit (polr)
# ==============================================================================
cat("7. ORDERED LOGIT (polr)\n")
cat("----------------------------------------\n")

set.seed(505)
n <- 200
x <- rnorm(n)
latent <- 1.5 * x + rlogis(n)
y <- cut(latent, breaks = c(-Inf, -1, 1, Inf), labels = c("Low", "Med", "High"))

data <- data.frame(y = ordered(y, levels = c("Low", "Med", "High")), x = x)
polr_result <- polr(y ~ x, data = data, method = "logistic")

cat("R MASS::polr results:\n")
cat(sprintf("  x coefficient: %.6f\n", coef(polr_result)))
cat(sprintf("  Threshold Low|Med: %.6f\n", polr_result$zeta[1]))
cat(sprintf("  Threshold Med|High: %.6f\n", polr_result$zeta[2]))
cat(sprintf("  Log-Likelihood: %.6f\n", logLik(polr_result)))
cat("\n")

# ==============================================================================
# 8. Negative Binomial (glm.nb)
# ==============================================================================
cat("8. NEGATIVE BINOMIAL (glm.nb)\n")
cat("----------------------------------------\n")

set.seed(606)
n <- 100
x <- runif(n, 0, 3)
mu <- exp(0.5 + 0.8 * x)
y <- rnbinom(n, size = 2, mu = mu)

data <- data.frame(y = y, x = x)
nb_result <- glm.nb(y ~ x, data = data)

cat("R MASS::glm.nb results:\n")
cat(sprintf("  Intercept: %.6f\n", coef(nb_result)[1]))
cat(sprintf("  x coefficient: %.6f\n", coef(nb_result)[2]))
cat(sprintf("  Theta (dispersion): %.6f\n", nb_result$theta))
cat(sprintf("  Log-Likelihood: %.6f\n", logLik(nb_result)))
cat(sprintf("  AIC: %.6f\n", AIC(nb_result)))
cat("\n")

cat("============================================================\n")
cat("Validation complete. Compare these values with Rust output.\n")
cat("============================================================\n")
