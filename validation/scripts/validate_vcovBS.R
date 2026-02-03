#!/usr/bin/env Rscript
# Bootstrap Covariance (vcovBS) Validation Script
# Compares R package sandwich's vcovBS results with p2a-core output

# Install packages if needed
packages <- c("sandwich", "microbenchmark")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

library(sandwich)
library(microbenchmark)

cat("=== Bootstrap Covariance (vcovBS) Validation ===\n\n")

# Test Case 1: Simple OLS with heteroskedastic errors
set.seed(42)
n <- 100
x <- rnorm(n)
# Heteroskedastic errors: variance increases with x
errors <- rnorm(n, 0, 0.5 + abs(x))
y <- 5 + 2 * x + errors

df <- data.frame(y = y, x = x)
model <- lm(y ~ x, data = df)

cat("Test Case 1: OLS with Heteroskedastic Errors (n=100)\n")
cat(sprintf("True beta0: 5.0, True beta1: 2.0\n"))
cat(sprintf("OLS estimates: beta0 = %.4f, beta1 = %.4f\n",
            coef(model)[1], coef(model)[2]))

# Standard OLS standard errors
cat("\nStandard OLS Standard Errors:\n")
ols_se <- sqrt(diag(vcov(model)))
cat(sprintf("  intercept SE: %.6f\n", ols_se[1]))
cat(sprintf("  x SE: %.6f\n", ols_se[2]))

# Pairs bootstrap (xy)
cat("\n--- Pairs Bootstrap (xy) ---\n")
set.seed(42)
bs_pairs <- vcovBS(model, R = 999, type = "xy")
bs_pairs_se <- sqrt(diag(bs_pairs))
cat(sprintf("  intercept SE: %.6f\n", bs_pairs_se[1]))
cat(sprintf("  x SE: %.6f\n", bs_pairs_se[2]))

# Residual bootstrap
cat("\n--- Residual Bootstrap ---\n")
set.seed(42)
bs_resid <- vcovBS(model, R = 999, type = "residual")
bs_resid_se <- sqrt(diag(bs_resid))
cat(sprintf("  intercept SE: %.6f\n", bs_resid_se[1]))
cat(sprintf("  x SE: %.6f\n", bs_resid_se[2]))

# Wild bootstrap
cat("\n--- Wild Bootstrap (Rademacher) ---\n")
set.seed(42)
bs_wild <- vcovBS(model, R = 999, type = "wild")
bs_wild_se <- sqrt(diag(bs_wild))
cat(sprintf("  intercept SE: %.6f\n", bs_wild_se[1]))
cat(sprintf("  x SE: %.6f\n", bs_wild_se[2]))

# Compare with HC standard errors
cat("\n--- Comparison with HC (sandwich) ---\n")
hc_se <- sqrt(diag(vcovHC(model, type = "HC1")))
cat(sprintf("  intercept SE (HC1): %.6f\n", hc_se[1]))
cat(sprintf("  x SE (HC1): %.6f\n", hc_se[2]))

cat("\n=== Expected Values for Rust Validation ===\n")
cat(sprintf("\nPairs Bootstrap:\n"))
cat(sprintf("  intercept SE: %.6f (should be similar)\n", bs_pairs_se[1]))
cat(sprintf("  x SE: %.6f (should be similar)\n", bs_pairs_se[2]))
cat(sprintf("\nResidual Bootstrap:\n"))
cat(sprintf("  intercept SE: %.6f (should be similar)\n", bs_resid_se[1]))
cat(sprintf("  x SE: %.6f (should be similar)\n", bs_resid_se[2]))
cat(sprintf("\nWild Bootstrap:\n"))
cat(sprintf("  intercept SE: %.6f (should be similar)\n", bs_wild_se[1]))
cat(sprintf("  x SE: %.6f (should be similar)\n", bs_wild_se[2]))

# Performance Benchmarks
cat("\n\n=== Performance Benchmarks ===\n")

benchmark_vcovBS <- function(n, n_boot = 999) {
  set.seed(42)
  x <- rnorm(n)
  y <- 5 + 2 * x + rnorm(n, 0, 0.5 + abs(x))
  model <- lm(y ~ x)

  bm <- microbenchmark(
    vcovBS(model, R = n_boot, type = "xy"),
    times = 20,
    unit = "microseconds"
  )

  return(median(bm$time) / 1000)  # Convert to microseconds
}

sizes <- c(100, 500, 1000)

cat("\n| Dataset Size | R vcovBS (µs) |\n")
cat("|--------------|---------------|\n")

for (n in sizes) {
  time_us <- tryCatch({
    benchmark_vcovBS(n, 200)  # Use 200 replications for speed
  }, error = function(e) {
    NA
  })

  if (!is.na(time_us)) {
    cat(sprintf("| n=%d        | %.2f          |\n", n, time_us))
  } else {
    cat(sprintf("| n=%d        | Error         |\n", n))
  }
}

cat("\n=== Validation Complete ===\n")
