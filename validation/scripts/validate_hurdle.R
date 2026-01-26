#!/usr/bin/env Rscript
# Hurdle Model Validation Script
# Compares R's pscl::hurdle() with p2a-core Rust implementation

cat("=== Hurdle Model Validation ===\n\n")

# Install pscl if not available
if (!requireNamespace("pscl", quietly = TRUE)) {
  cat("Installing pscl package...\n")
  install.packages("pscl", repos = "https://cloud.r-project.org/")
}

library(pscl)

# Test 1: Basic hurdle Poisson model
cat("--- Test 1: Hurdle Poisson ---\n")
set.seed(42)
n <- 200

# Generate data with zeros and positive counts
x <- rnorm(n)
# Binary part: probability of positive count depends on x
prob_positive <- plogis(0.5 + 1.0 * x)
is_positive <- rbinom(n, 1, prob_positive)

# Count part: Poisson for positive values
lambda <- exp(0.8 + 0.5 * x)
y_count <- rpois(n, lambda)
# Ensure positive when is_positive=1
y_count[y_count == 0] <- 1

# Final y: zero if not positive, else the count
y <- ifelse(is_positive == 1, y_count, 0)

data1 <- data.frame(y = y, x = x)

# Fit hurdle Poisson
hurdle_pois <- hurdle(y ~ x, data = data1, dist = "poisson", zero.dist = "binomial")

cat("N observations:", n, "\n")
cat("N zeros:", sum(y == 0), "\n")
cat("N positive:", sum(y > 0), "\n\n")

cat("Binary (zero-hurdle) coefficients:\n")
print(summary(hurdle_pois)$coefficients$zero)

cat("\nCount (truncated Poisson) coefficients:\n")
print(summary(hurdle_pois)$coefficients$count)

cat("\nLog-likelihood:", logLik(hurdle_pois), "\n")
cat("AIC:", AIC(hurdle_pois), "\n")
cat("BIC:", BIC(hurdle_pois), "\n")

# Test 2: Hurdle Negative Binomial
cat("\n--- Test 2: Hurdle Negative Binomial ---\n")
set.seed(123)
n <- 200

# Generate overdispersed count data
x <- rnorm(n)
prob_positive <- plogis(-0.3 + 0.8 * x)
is_positive <- rbinom(n, 1, prob_positive)

# Negative binomial for positive values
mu <- exp(1.0 + 0.4 * x)
theta <- 2.0  # Shape parameter
y_count <- rnbinom(n, mu = mu, size = theta)
y_count[y_count == 0] <- 1

y <- ifelse(is_positive == 1, y_count, 0)

data2 <- data.frame(y = y, x = x)

# Fit hurdle negative binomial
hurdle_nb <- hurdle(y ~ x, data = data2, dist = "negbin", zero.dist = "binomial")

cat("N observations:", n, "\n")
cat("N zeros:", sum(y == 0), "\n")
cat("N positive:", sum(y > 0), "\n\n")

cat("Binary (zero-hurdle) coefficients:\n")
print(summary(hurdle_nb)$coefficients$zero)

cat("\nCount (truncated NegBin) coefficients:\n")
print(summary(hurdle_nb)$coefficients$count)

cat("\nTheta (dispersion):", hurdle_nb$theta, "\n")
cat("Log-likelihood:", logLik(hurdle_nb), "\n")
cat("AIC:", AIC(hurdle_nb), "\n")
cat("BIC:", BIC(hurdle_nb), "\n")

# Test 3: Different covariates for binary and count parts
cat("\n--- Test 3: Different Covariates (Z ≠ X) ---\n")
set.seed(456)
n <- 250

x1 <- rnorm(n)
x2 <- rnorm(n)
z <- rnorm(n)  # Different variable for binary part

prob_positive <- plogis(-0.2 + 0.6 * z)
is_positive <- rbinom(n, 1, prob_positive)

mu <- exp(0.5 + 0.3 * x1 + 0.2 * x2)
y_count <- rpois(n, mu)
y_count[y_count == 0] <- 1

y <- ifelse(is_positive == 1, y_count, 0)

data3 <- data.frame(y = y, x1 = x1, x2 = x2, z = z)

# Fit with different formulas for count and zero parts
hurdle_diff <- hurdle(y ~ x1 + x2 | z, data = data3, dist = "poisson", zero.dist = "binomial")

cat("N observations:", n, "\n")
cat("N zeros:", sum(y == 0), "\n")

cat("\nBinary coefficients (z only):\n")
print(summary(hurdle_diff)$coefficients$zero)

cat("\nCount coefficients (x1, x2):\n")
print(summary(hurdle_diff)$coefficients$count)

cat("\nLog-likelihood:", logLik(hurdle_diff), "\n")

# Performance benchmarks
cat("\n=== Performance Benchmarks ===\n\n")

library(microbenchmark)

# Benchmark function
benchmark_hurdle <- function(n, type = "poisson") {
  set.seed(42)
  x <- rnorm(n)
  prob_positive <- plogis(0.5 + 1.0 * x)
  is_positive <- rbinom(n, 1, prob_positive)

  if (type == "poisson") {
    lambda <- exp(0.8 + 0.5 * x)
    y_count <- rpois(n, lambda)
  } else {
    mu <- exp(0.8 + 0.5 * x)
    y_count <- rnbinom(n, mu = mu, size = 2)
  }
  y_count[y_count == 0] <- 1
  y <- ifelse(is_positive == 1, y_count, 0)

  data.frame(y = y, x = x)
}

# Benchmark different sizes
for (size in c(100, 500, 1000, 2000)) {
  data_bench <- benchmark_hurdle(size, "poisson")

  bm <- microbenchmark(
    hurdle(y ~ x, data = data_bench, dist = "poisson", zero.dist = "binomial"),
    times = 20,
    unit = "milliseconds"
  )

  med <- median(bm$time) / 1e6  # Convert to milliseconds
  cat(sprintf("Hurdle Poisson n=%d: %.2f ms (median)\n", size, med))
}

cat("\n")
for (size in c(100, 500, 1000, 2000)) {
  data_bench <- benchmark_hurdle(size, "negbin")

  bm <- microbenchmark(
    hurdle(y ~ x, data = data_bench, dist = "negbin", zero.dist = "binomial"),
    times = 20,
    unit = "milliseconds"
  )

  med <- median(bm$time) / 1e6
  cat(sprintf("Hurdle NegBin n=%d: %.2f ms (median)\n", size, med))
}

# Save expected results for Rust comparison
cat("\n=== Saving Expected Results ===\n")

# Create validation data
set.seed(42)
n <- 100
x <- rnorm(n)
prob_positive <- plogis(0.5 + 1.0 * x)
is_positive <- rbinom(n, 1, prob_positive)
lambda <- exp(0.8 + 0.5 * x)
y_count <- rpois(n, lambda)
y_count[y_count == 0] <- 1
y <- ifelse(is_positive == 1, y_count, 0)
data_val <- data.frame(y = y, x = x)

model_val <- hurdle(y ~ x, data = data_val, dist = "poisson", zero.dist = "binomial")

write.csv(data.frame(
  part = c("zero", "zero", "count", "count"),
  variable = c("(Intercept)", "x", "(Intercept)", "x"),
  coefficient = c(coef(model_val, "zero"), coef(model_val, "count")),
  std_error = c(
    sqrt(diag(vcov(model_val, "zero"))),
    sqrt(diag(vcov(model_val, "count")))
  )
), "validation/expected/hurdle_poisson.csv", row.names = FALSE)

cat("Saved expected results to validation/expected/hurdle_poisson.csv\n")
cat("Done!\n")
