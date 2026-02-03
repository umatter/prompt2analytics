#!/usr/bin/env Rscript
# Harvey-Collier Test Validation Script
# Compares R's lmtest::harvtest() with p2a-core Rust implementation

cat("=== Harvey-Collier Test Validation ===\n\n")

# Install lmtest if not available
if (!requireNamespace("lmtest", quietly = TRUE)) {
  cat("Installing lmtest package...\n")
  install.packages("lmtest", repos = "https://cloud.r-project.org/")
}

library(lmtest)

# Test 1: Linear relationship (should NOT reject)
cat("--- Test 1: Linear Relationship (H0: linear is correct) ---\n")
set.seed(42)
n <- 50
x <- 1:n
y <- 2 + 3 * x + rnorm(n, 0, 2)

model1 <- lm(y ~ x)
hc1 <- harvtest(model1)

cat("Model: y ~ x (truly linear)\n")
cat("t-statistic:", hc1$statistic, "\n")
cat("df:", hc1$parameter, "\n")
cat("p-value:", hc1$p.value, "\n")
cat("Interpretation: p > 0.05 => fail to reject linearity\n\n")

# Test 2: Quadratic relationship (should reject)
cat("--- Test 2: Quadratic Relationship (H1: nonlinear) ---\n")
set.seed(123)
n <- 50
x <- 1:n
y <- 1 + x + 0.05 * x^2 + rnorm(n, 0, 1)

model2 <- lm(y ~ x)  # Misspecified model
hc2 <- harvtest(model2)

cat("Model: y ~ x (true DGP: y = 1 + x + 0.05*x^2 + noise)\n")
cat("t-statistic:", hc2$statistic, "\n")
cat("df:", hc2$parameter, "\n")
cat("p-value:", hc2$p.value, "\n")
cat("Interpretation: p < 0.05 => detect nonlinearity\n\n")

# Test 3: Multiple regressors
cat("--- Test 3: Multiple Regressors ---\n")
set.seed(456)
n <- 60
x1 <- rnorm(n)
x2 <- rnorm(n)
y <- 1 + 2*x1 + 3*x2 + rnorm(n, 0, 0.5)

model3 <- lm(y ~ x1 + x2)
hc3 <- harvtest(model3)

cat("Model: y ~ x1 + x2 (truly linear)\n")
cat("t-statistic:", hc3$statistic, "\n")
cat("df:", hc3$parameter, "\n")
cat("p-value:", hc3$p.value, "\n\n")

# Test 4: Strong nonlinearity
cat("--- Test 4: Strong Nonlinearity (sin function) ---\n")
set.seed(789)
n <- 100
x <- seq(0, 4*pi, length.out = n)
y <- sin(x) + rnorm(n, 0, 0.1)

model4 <- lm(y ~ x)
hc4 <- harvtest(model4)

cat("Model: y ~ x (true DGP: y = sin(x) + noise)\n")
cat("t-statistic:", hc4$statistic, "\n")
cat("df:", hc4$parameter, "\n")
cat("p-value:", hc4$p.value, "\n\n")

# Performance benchmarks
cat("=== Performance Benchmarks ===\n\n")

library(microbenchmark)

for (n in c(50, 100, 500, 1000)) {
  set.seed(42)
  x <- rnorm(n)
  y <- 1 + 2*x + rnorm(n, 0, 0.5)
  model <- lm(y ~ x)

  bm <- microbenchmark(
    harvtest(model),
    times = 50,
    unit = "microseconds"
  )

  med <- median(bm$time) / 1000  # Convert to microseconds
  cat(sprintf("harvtest n=%d: %.2f us (median)\n", n, med))
}

# Save expected results
cat("\n=== Saving Expected Results ===\n")

# Test case 1: Linear
set.seed(42)
n <- 50
x <- 1:n
y <- 2 + 3 * x + rnorm(n, 0, 2)
model <- lm(y ~ x)
hc <- harvtest(model)

write.csv(data.frame(
  statistic = as.numeric(hc$statistic),
  df = as.numeric(hc$parameter),
  p_value = hc$p.value
), "validation/expected/harvtest_linear.csv", row.names = FALSE)

cat("Saved expected results to validation/expected/harvtest_linear.csv\n")
cat("Done!\n")
