#!/usr/bin/env Rscript
# Regression Benchmarks for Cross-Language Comparison
# Compares R's lm() and sandwich package against p2a Rust implementation

library(microbenchmark)
library(sandwich)

# Ensure reproducibility
set.seed(42)

# Generate data with same DGP as Rust benchmarks
generate_regression_data <- function(n, k = 5) {
  X <- matrix(runif(n * k, -1, 1), nrow = n, ncol = k)
  colnames(X) <- paste0("x", 1:k)

  # y = sum(x_i) + noise
  y <- rowSums(X) + runif(n, 0, 0.5)

  data.frame(y = y, X)
}

# Benchmark OLS with standard errors
benchmark_ols <- function() {
  results <- list()

  for (n in c(100, 1000, 10000)) {
    cat(sprintf("Benchmarking OLS with n=%d\n", n))
    data <- generate_regression_data(n)

    # Standard OLS
    bm_standard <- microbenchmark(
      lm(y ~ x1 + x2 + x3 + x4 + x5, data = data),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("ols_standard_", n)]] <- summary(bm_standard)
  }

  results
}

# Benchmark Robust Standard Errors (HC0-HC3)
# NOTE: This benchmarks the FULL regression + vcovHC to match Rust benchmark
benchmark_robust_se <- function() {
  results <- list()

  n <- 1000
  data <- generate_regression_data(n)

  for (hc_type in c("HC0", "HC1", "HC2", "HC3")) {
    cat(sprintf("Benchmarking %s (full regression + vcov)\n", hc_type))

    # Time the FULL workflow: fit + robust vcov (matches Rust benchmark)
    bm <- microbenchmark(
      {
        fit <- lm(y ~ x1 + x2 + x3 + x4 + x5, data = data)
        vcovHC(fit, type = hc_type)
      },
      times = 100,
      unit = "microseconds"
    )

    results[[hc_type]] <- summary(bm)
  }

  results
}

# Run benchmarks
cat("=== OLS Benchmarks ===\n")
ols_results <- benchmark_ols()

cat("\n=== Robust SE Benchmarks ===\n")
robust_results <- benchmark_robust_se()

# Save results
save_results <- function(results, filename) {
  df <- do.call(rbind, lapply(names(results), function(name) {
    r <- results[[name]]
    data.frame(
      method = name,
      mean_us = r$mean,
      median_us = r$median,
      min_us = r$min,
      max_us = r$max,
      n_eval = r$neval
    )
  }))

  write.csv(df, filename, row.names = FALSE)
  cat(sprintf("Results saved to %s\n", filename))
}

# Create results directory if needed
dir.create("results", showWarnings = FALSE)

save_results(ols_results, "results/regression_ols.csv")
save_results(robust_results, "results/regression_robust_se.csv")

# Print summary
cat("\n=== Summary ===\n")
print(do.call(rbind, ols_results))
print(do.call(rbind, robust_results))
