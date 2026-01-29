#!/usr/bin/env Rscript
#
# R Validation Script for Extended Discrete Choice Models
# Phase 3: Count models, ordered models, multinomial logit
# Generates reference values for p2a-core validation tests
#

cat("=== Extended Discrete Choice Validation Script (Phase 3) ===\n\n")

# Required packages
required_packages <- c("MASS", "pscl", "nnet")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing package:", pkg, "\n")
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
  library(pkg, character.only = TRUE)
}

# =============================================================================
# Test 1: Negative Binomial Regression
# =============================================================================
cat("--- Test 1: Negative Binomial (MASS::glm.nb) ---\n")

set.seed(42)
n <- 100

# Generate overdispersed count data
x <- runif(n, 0, 5)
mu <- exp(0.5 + 0.3 * x)  # log-linear mean
theta_true <- 2.0  # dispersion parameter
y <- rnbinom(n, mu = mu, size = theta_true)

data_nb <- data.frame(y = y, x = x)

nb_fit <- glm.nb(y ~ x, data = data_nb)

cat("\nNegative Binomial Coefficients:\n")
print(coef(nb_fit))

cat("\nStandard Errors:\n")
print(sqrt(diag(vcov(nb_fit))))

cat("\nTheta (dispersion):", nb_fit$theta, "\n")
cat("Log-Likelihood:", logLik(nb_fit), "\n")
cat("AIC:", AIC(nb_fit), "\n")

# Save results
nb_results <- data.frame(
  variable = names(coef(nb_fit)),
  coefficient = as.numeric(coef(nb_fit)),
  std_error = sqrt(diag(vcov(nb_fit))),
  theta = nb_fit$theta,
  log_likelihood = as.numeric(logLik(nb_fit)),
  aic = AIC(nb_fit),
  n_obs = n
)
write.csv(nb_results, "validation/expected/negbin_test.csv", row.names = FALSE)

# =============================================================================
# Test 2: Zero-Inflated Poisson (ZIP)
# =============================================================================
cat("\n--- Test 2: Zero-Inflated Poisson (pscl::zeroinfl) ---\n")

set.seed(42)
n <- 150

# Generate ZIP data: excess zeros + Poisson counts
x <- runif(n, 0, 4)

# Zero-inflation probability (logistic)
pi_zero <- plogis(-1 + 0.5 * x)  # Higher x -> more zeros
zero_state <- rbinom(n, 1, pi_zero)

# Poisson mean (for count part)
lambda <- exp(1.0 + 0.4 * x)

# Generate y: 0 if zero_state, else Poisson
y <- ifelse(zero_state == 1, 0, rpois(n, lambda))

data_zip <- data.frame(y = y, x = x)

zip_fit <- zeroinfl(y ~ x | x, data = data_zip, dist = "poisson")

cat("\nZIP Count Model Coefficients:\n")
print(coef(zip_fit, "count"))

cat("\nZIP Zero-Inflation Coefficients:\n")
print(coef(zip_fit, "zero"))

cat("\nZIP Standard Errors (Count):\n")
se <- sqrt(diag(vcov(zip_fit)))
count_se <- se[1:2]
zero_se <- se[3:4]
print(count_se)

cat("\nZIP Standard Errors (Zero):\n")
print(zero_se)

cat("\nLog-Likelihood:", logLik(zip_fit), "\n")
cat("Number of zeros:", sum(y == 0), "\n")

# Save results
zip_results <- data.frame(
  type = c("count", "count", "zero", "zero"),
  variable = c("(Intercept)", "x", "(Intercept)", "x"),
  coefficient = c(coef(zip_fit, "count"), coef(zip_fit, "zero")),
  std_error = c(count_se, zero_se),
  log_likelihood = as.numeric(logLik(zip_fit)),
  n_obs = n,
  n_zeros = sum(y == 0)
)
write.csv(zip_results, "validation/expected/zip_test.csv", row.names = FALSE)

# =============================================================================
# Test 3: Zero-Inflated Negative Binomial (ZINB)
# =============================================================================
cat("\n--- Test 3: Zero-Inflated Negative Binomial (pscl::zeroinfl) ---\n")

set.seed(42)
n <- 150

# Similar structure but with overdispersion
x <- runif(n, 0, 4)

# Zero-inflation probability
pi_zero <- plogis(-0.5 + 0.3 * x)
zero_state <- rbinom(n, 1, pi_zero)

# Negative binomial for count part
mu <- exp(0.8 + 0.3 * x)
theta_nb <- 1.5

y <- ifelse(zero_state == 1, 0, rnbinom(n, mu = mu, size = theta_nb))

data_zinb <- data.frame(y = y, x = x)

zinb_fit <- zeroinfl(y ~ x | x, data = data_zinb, dist = "negbin")

cat("\nZINB Count Model Coefficients:\n")
print(coef(zinb_fit, "count"))

cat("\nZINB Zero-Inflation Coefficients:\n")
print(coef(zinb_fit, "zero"))

cat("\nZINB Theta:", zinb_fit$theta, "\n")
cat("Log-Likelihood:", logLik(zinb_fit), "\n")

# Save results
se_zinb <- sqrt(diag(vcov(zinb_fit)))
zinb_results <- data.frame(
  type = c("count", "count", "zero", "zero"),
  variable = c("(Intercept)", "x", "(Intercept)", "x"),
  coefficient = c(coef(zinb_fit, "count"), coef(zinb_fit, "zero")),
  std_error = se_zinb[1:4],
  theta = zinb_fit$theta,
  log_likelihood = as.numeric(logLik(zinb_fit)),
  n_obs = n,
  n_zeros = sum(y == 0)
)
write.csv(zinb_results, "validation/expected/zinb_test.csv", row.names = FALSE)

# =============================================================================
# Test 4: Hurdle Poisson Model
# =============================================================================
cat("\n--- Test 4: Hurdle Poisson (pscl::hurdle) ---\n")

set.seed(42)
n <- 150

x <- runif(n, 0, 5)

# Two-part model:
# 1. Binary: P(y > 0) = logistic(-1 + 0.5*x)
# 2. Truncated Poisson for positive counts

prob_pos <- plogis(-1 + 0.5 * x)
is_positive <- rbinom(n, 1, prob_pos)

# Truncated Poisson for positive y
lambda <- exp(1.5 + 0.2 * x)
y <- rep(0, n)
for (i in 1:n) {
  if (is_positive[i] == 1) {
    # Truncated Poisson (rejection sampling)
    repeat {
      tmp <- rpois(1, lambda[i])
      if (tmp > 0) {
        y[i] <- tmp
        break
      }
    }
  }
}

data_hurdle <- data.frame(y = y, x = x)

hurdle_fit <- hurdle(y ~ x | x, data = data_hurdle, dist = "poisson")

cat("\nHurdle Poisson Count Coefficients:\n")
print(coef(hurdle_fit, "count"))

cat("\nHurdle Poisson Zero Coefficients:\n")
print(coef(hurdle_fit, "zero"))

cat("\nLog-Likelihood:", logLik(hurdle_fit), "\n")

# Save results
se_hurdle <- sqrt(diag(vcov(hurdle_fit)))
hurdle_results <- data.frame(
  type = c("count", "count", "zero", "zero"),
  variable = c("(Intercept)", "x", "(Intercept)", "x"),
  coefficient = c(coef(hurdle_fit, "count"), coef(hurdle_fit, "zero")),
  std_error = se_hurdle[1:4],
  log_likelihood = as.numeric(logLik(hurdle_fit)),
  n_obs = n,
  n_zeros = sum(y == 0),
  n_positive = sum(y > 0)
)
write.csv(hurdle_results, "validation/expected/hurdle_poisson_test.csv", row.names = FALSE)

# =============================================================================
# Test 5: Hurdle Negative Binomial Model
# =============================================================================
cat("\n--- Test 5: Hurdle Negative Binomial (pscl::hurdle) ---\n")

set.seed(42)
n <- 150

x <- runif(n, 0, 5)

prob_pos <- plogis(-0.5 + 0.4 * x)
is_positive <- rbinom(n, 1, prob_pos)

mu <- exp(1.0 + 0.25 * x)
theta_h <- 2.0

y <- rep(0, n)
for (i in 1:n) {
  if (is_positive[i] == 1) {
    repeat {
      tmp <- rnbinom(1, mu = mu[i], size = theta_h)
      if (tmp > 0) {
        y[i] <- tmp
        break
      }
    }
  }
}

data_hurdle_nb <- data.frame(y = y, x = x)

hurdle_nb_fit <- hurdle(y ~ x | x, data = data_hurdle_nb, dist = "negbin")

cat("\nHurdle NegBin Count Coefficients:\n")
print(coef(hurdle_nb_fit, "count"))

cat("\nHurdle NegBin Zero Coefficients:\n")
print(coef(hurdle_nb_fit, "zero"))

cat("\nHurdle NegBin Theta:", hurdle_nb_fit$theta, "\n")
cat("Log-Likelihood:", logLik(hurdle_nb_fit), "\n")

# Save results
se_hurdle_nb <- sqrt(diag(vcov(hurdle_nb_fit)))
hurdle_nb_results <- data.frame(
  type = c("count", "count", "zero", "zero"),
  variable = c("(Intercept)", "x", "(Intercept)", "x"),
  coefficient = c(coef(hurdle_nb_fit, "count"), coef(hurdle_nb_fit, "zero")),
  std_error = se_hurdle_nb[1:4],
  theta = hurdle_nb_fit$theta,
  log_likelihood = as.numeric(logLik(hurdle_nb_fit)),
  n_obs = n,
  n_zeros = sum(y == 0),
  n_positive = sum(y > 0)
)
write.csv(hurdle_nb_results, "validation/expected/hurdle_negbin_test.csv", row.names = FALSE)

# =============================================================================
# Test 6: Ordered Logit (MASS::polr)
# =============================================================================
cat("\n--- Test 6: Ordered Logit (MASS::polr) ---\n")

set.seed(42)
n <- 200

x <- rnorm(n)

# Latent variable: y* = 0.5 + 1.2*x + e (logistic error)
latent <- 0.5 + 1.2 * x + rlogis(n)

# Cut into categories
y <- cut(latent, breaks = c(-Inf, -1, 1, Inf), labels = c("Low", "Medium", "High"))
y <- factor(y, levels = c("Low", "Medium", "High"), ordered = TRUE)

data_ordered <- data.frame(y = y, x = x)

ologit_fit <- polr(y ~ x, data = data_ordered, method = "logistic")

cat("\nOrdered Logit Coefficients:\n")
print(coef(ologit_fit))

cat("\nOrdered Logit Thresholds:\n")
print(ologit_fit$zeta)

cat("\nOrdered Logit Standard Errors:\n")
se_ologit <- sqrt(diag(vcov(ologit_fit)))
print(se_ologit)

cat("\nLog-Likelihood:", logLik(ologit_fit), "\n")
cat("AIC:", AIC(ologit_fit), "\n")

# Category counts
cat("\nCategory counts:\n")
print(table(y))

# Save results
ologit_results <- data.frame(
  type = c("coef", "threshold", "threshold"),
  variable = c("x", "Low|Medium", "Medium|High"),
  estimate = c(coef(ologit_fit), ologit_fit$zeta),
  std_error = se_ologit,
  log_likelihood = as.numeric(logLik(ologit_fit)),
  aic = AIC(ologit_fit),
  n_obs = n
)
write.csv(ologit_results, "validation/expected/ordered_logit_test.csv", row.names = FALSE)

# =============================================================================
# Test 7: Ordered Probit (MASS::polr)
# =============================================================================
cat("\n--- Test 7: Ordered Probit (MASS::polr) ---\n")

set.seed(42)
n <- 200

x <- rnorm(n)

# Latent variable: y* = 0.3 + 0.8*x + e (normal error)
latent <- 0.3 + 0.8 * x + rnorm(n)

# Cut into categories
y <- cut(latent, breaks = c(-Inf, -0.5, 0.5, Inf), labels = c("Low", "Medium", "High"))
y <- factor(y, levels = c("Low", "Medium", "High"), ordered = TRUE)

data_oprobit <- data.frame(y = y, x = x)

oprobit_fit <- polr(y ~ x, data = data_oprobit, method = "probit")

cat("\nOrdered Probit Coefficients:\n")
print(coef(oprobit_fit))

cat("\nOrdered Probit Thresholds:\n")
print(oprobit_fit$zeta)

cat("\nOrdered Probit Standard Errors:\n")
se_oprobit <- sqrt(diag(vcov(oprobit_fit)))
print(se_oprobit)

cat("\nLog-Likelihood:", logLik(oprobit_fit), "\n")

# Save results
oprobit_results <- data.frame(
  type = c("coef", "threshold", "threshold"),
  variable = c("x", "Low|Medium", "Medium|High"),
  estimate = c(coef(oprobit_fit), oprobit_fit$zeta),
  std_error = se_oprobit,
  log_likelihood = as.numeric(logLik(oprobit_fit)),
  aic = AIC(oprobit_fit),
  n_obs = n
)
write.csv(oprobit_results, "validation/expected/ordered_probit_test.csv", row.names = FALSE)

# =============================================================================
# Test 8: Multinomial Logit (nnet::multinom)
# =============================================================================
cat("\n--- Test 8: Multinomial Logit (nnet::multinom) ---\n")

set.seed(42)
n <- 300

x1 <- rnorm(n)
x2 <- rnorm(n)

# Generate multinomial outcome with 3 categories
# True model: P(Y=j) = exp(Vj) / sum(exp(Vk))
# V_A = 0 (reference)
# V_B = 0.5 + 1.0*x1 - 0.5*x2
# V_C = -0.3 + 0.3*x1 + 0.8*x2

V_A <- rep(0, n)
V_B <- 0.5 + 1.0 * x1 - 0.5 * x2
V_C <- -0.3 + 0.3 * x1 + 0.8 * x2

exp_V <- cbind(exp(V_A), exp(V_B), exp(V_C))
probs <- exp_V / rowSums(exp_V)

y <- apply(probs, 1, function(p) sample(c("A", "B", "C"), 1, prob = p))
y <- factor(y, levels = c("A", "B", "C"))

data_multinom <- data.frame(y = y, x1 = x1, x2 = x2)

multinom_fit <- multinom(y ~ x1 + x2, data = data_multinom, trace = FALSE)

cat("\nMultinomial Logit Coefficients:\n")
print(coef(multinom_fit))

cat("\nMultinomial Logit Standard Errors:\n")
se_multinom <- summary(multinom_fit)$standard.errors
print(se_multinom)

cat("\nLog-Likelihood:", logLik(multinom_fit), "\n")
cat("AIC:", AIC(multinom_fit), "\n")

# Category counts
cat("\nCategory counts:\n")
print(table(y))

# Save results - flatten the coefficient matrix
coefs <- coef(multinom_fit)
ses <- se_multinom
multinom_results <- data.frame(
  category = c(rep("B", 3), rep("C", 3)),
  variable = rep(c("(Intercept)", "x1", "x2"), 2),
  coefficient = c(coefs["B", ], coefs["C", ]),
  std_error = c(ses["B", ], ses["C", ]),
  reference = "A",
  log_likelihood = as.numeric(logLik(multinom_fit)),
  aic = AIC(multinom_fit),
  n_obs = n
)
write.csv(multinom_results, "validation/expected/multinom_test.csv", row.names = FALSE)

# =============================================================================
# Test 9: Multinomial with 4 categories
# =============================================================================
cat("\n--- Test 9: Multinomial with 4 Categories ---\n")

set.seed(42)
n <- 400

x <- rnorm(n)

# 4 categories
V_1 <- rep(0, n)
V_2 <- 0.2 + 0.5 * x
V_3 <- -0.1 + 0.8 * x
V_4 <- 0.3 + 1.2 * x

exp_V <- cbind(exp(V_1), exp(V_2), exp(V_3), exp(V_4))
probs <- exp_V / rowSums(exp_V)

y <- apply(probs, 1, function(p) sample(1:4, 1, prob = p))
y <- factor(y)

data_multi4 <- data.frame(y = y, x = x)

multi4_fit <- multinom(y ~ x, data = data_multi4, trace = FALSE)

cat("\nMultinomial (4 cat) Coefficients:\n")
print(coef(multi4_fit))

# Save results
coefs4 <- coef(multi4_fit)
ses4 <- summary(multi4_fit)$standard.errors
multi4_results <- data.frame(
  category = c(rep("2", 2), rep("3", 2), rep("4", 2)),
  variable = rep(c("(Intercept)", "x"), 3),
  coefficient = c(coefs4["2", ], coefs4["3", ], coefs4["4", ]),
  std_error = c(ses4["2", ], ses4["3", ], ses4["4", ]),
  reference = "1",
  log_likelihood = as.numeric(logLik(multi4_fit)),
  n_obs = n
)
write.csv(multi4_results, "validation/expected/multinom_4cat_test.csv", row.names = FALSE)

# =============================================================================
# Test 10: Ordered Logit with multiple predictors
# =============================================================================
cat("\n--- Test 10: Ordered Logit with Multiple Predictors ---\n")

set.seed(42)
n <- 300

x1 <- rnorm(n)
x2 <- rnorm(n)

latent <- 0.5 * x1 + 0.8 * x2 + rlogis(n)

y <- cut(latent, breaks = c(-Inf, -1.5, 0, 1.5, Inf),
         labels = c("Very Low", "Low", "High", "Very High"))
y <- factor(y, levels = c("Very Low", "Low", "High", "Very High"), ordered = TRUE)

data_ologit_multi <- data.frame(y = y, x1 = x1, x2 = x2)

ologit_multi_fit <- polr(y ~ x1 + x2, data = data_ologit_multi, method = "logistic")

cat("\nOrdered Logit (Multi) Coefficients:\n")
print(coef(ologit_multi_fit))

cat("\nOrdered Logit (Multi) Thresholds:\n")
print(ologit_multi_fit$zeta)

se_multi <- sqrt(diag(vcov(ologit_multi_fit)))

# Save results
ologit_multi_results <- data.frame(
  type = c("coef", "coef", "threshold", "threshold", "threshold"),
  variable = c("x1", "x2", "Very Low|Low", "Low|High", "High|Very High"),
  estimate = c(coef(ologit_multi_fit), ologit_multi_fit$zeta),
  std_error = se_multi,
  log_likelihood = as.numeric(logLik(ologit_multi_fit)),
  n_obs = n
)
write.csv(ologit_multi_results, "validation/expected/ordered_logit_multi_test.csv", row.names = FALSE)

# =============================================================================
# Summary
# =============================================================================
cat("\n=== Summary of Generated Files ===\n")
cat("1. validation/expected/negbin_test.csv\n")
cat("2. validation/expected/zip_test.csv\n")
cat("3. validation/expected/zinb_test.csv\n")
cat("4. validation/expected/hurdle_poisson_test.csv\n")
cat("5. validation/expected/hurdle_negbin_test.csv\n")
cat("6. validation/expected/ordered_logit_test.csv\n")
cat("7. validation/expected/ordered_probit_test.csv\n")
cat("8. validation/expected/multinom_test.csv\n")
cat("9. validation/expected/multinom_4cat_test.csv\n")
cat("10. validation/expected/ordered_logit_multi_test.csv\n")

cat("\n=== Extended Discrete Choice Validation Script Complete ===\n")
