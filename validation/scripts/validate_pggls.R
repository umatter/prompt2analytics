#!/usr/bin/env Rscript
# Panel GLS (pggls) Validation Script
# Compares R package plm's pggls results with p2a-core output

# Install packages if needed
packages <- c("plm", "microbenchmark")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org/")
  }
}

library(plm)
library(microbenchmark)

cat("=== Panel GLS (pggls) Validation ===\n\n")

# Test Case 1: Balanced panel with firm-level heteroskedasticity
# Generate synthetic data similar to Rust test
set.seed(42)

# Parameters
n_firms <- 10
n_periods <- 5
n_obs <- n_firms * n_periods

# Generate panel structure
firm_ids <- rep(1:n_firms, each = n_periods)
time_ids <- rep(1:n_periods, times = n_firms)

# True parameters
beta0_true <- 5.0
beta1_true <- 2.0

# Fixed effects for each firm
firm_fe <- rnorm(n_firms, 0, 1)

# Generate X and Y
x <- rnorm(n_obs, 0, 1)
errors <- rnorm(n_obs, 0, 0.5)
y <- beta0_true + beta1_true * x + firm_fe[firm_ids] + errors

# Create data frame
panel_df <- data.frame(
  firm = factor(firm_ids),
  time = time_ids,
  y = y,
  x = x
)

# Convert to pdata.frame
pdata <- pdata.frame(panel_df, index = c("firm", "time"))

cat("Test Case 1: Balanced Panel (N=10, T=5)\n")
cat(sprintf("True beta0: %.2f, True beta1: %.2f\n", beta0_true, beta1_true))

# Run FGLS with fixed effects model
cat("\n--- Fixed Effects GLS (FEGLS) ---\n")
result_fe <- tryCatch({
  pggls(y ~ x, data = pdata, model = "within")
}, error = function(e) {
  cat("FEGLS failed:", e$message, "\n")
  NULL
})

if (!is.null(result_fe)) {
  cat("Coefficients:\n")
  print(coef(result_fe))
  cat("\nSummary:\n")
  print(summary(result_fe))
}

# Run pooled GLS
cat("\n--- Pooled GLS ---\n")
result_pool <- tryCatch({
  pggls(y ~ x, data = pdata, model = "pooling")
}, error = function(e) {
  cat("Pooled GLS failed:", e$message, "\n")
  NULL
})

if (!is.null(result_pool)) {
  cat("Coefficients:\n")
  print(coef(result_pool))
  cat("\nSummary:\n")
  print(summary(result_pool))
}

# Run first-difference GLS
cat("\n--- First Difference GLS ---\n")
result_fd <- tryCatch({
  pggls(y ~ x, data = pdata, model = "fd")
}, error = function(e) {
  cat("First Difference GLS failed:", e$message, "\n")
  NULL
})

if (!is.null(result_fd)) {
  cat("Coefficients:\n")
  print(coef(result_fd))
  cat("\nSummary:\n")
  print(summary(result_fd))
}

# Expected values for Rust validation
cat("\n=== Expected Values for Rust Validation ===\n")

if (!is.null(result_fe)) {
  fe_coef <- coef(result_fe)
  fe_se <- sqrt(diag(vcov(result_fe)))
  cat("\nFixed Effects GLS:\n")
  cat(sprintf("  x coefficient: %.6f\n", fe_coef["x"]))
  cat(sprintf("  x std error: %.6f\n", fe_se["x"]))
}

if (!is.null(result_pool)) {
  pool_coef <- coef(result_pool)
  pool_se <- sqrt(diag(vcov(result_pool)))
  cat("\nPooled GLS:\n")
  cat(sprintf("  intercept: %.6f\n", pool_coef["(Intercept)"]))
  cat(sprintf("  x coefficient: %.6f\n", pool_coef["x"]))
  cat(sprintf("  intercept std error: %.6f\n", pool_se["(Intercept)"]))
  cat(sprintf("  x std error: %.6f\n", pool_se["x"]))
}

if (!is.null(result_fd)) {
  fd_coef <- coef(result_fd)
  fd_se <- sqrt(diag(vcov(result_fd)))
  cat("\nFirst Difference GLS:\n")
  for (name in names(fd_coef)) {
    cat(sprintf("  %s coefficient: %.6f\n", name, fd_coef[name]))
    cat(sprintf("  %s std error: %.6f\n", name, fd_se[name]))
  }
}

# Test Case 2: Larger panel for benchmarking
cat("\n\n=== Performance Benchmarks ===\n")

# Benchmark function
benchmark_pggls <- function(n_firms, n_periods) {
  n_obs <- n_firms * n_periods
  firm_ids <- rep(1:n_firms, each = n_periods)
  time_ids <- rep(1:n_periods, times = n_firms)
  x <- rnorm(n_obs)
  y <- 5 + 2 * x + rnorm(n_obs, 0, 0.5)

  df <- data.frame(
    firm = factor(firm_ids),
    time = time_ids,
    y = y,
    x = x
  )
  pdata <- pdata.frame(df, index = c("firm", "time"))

  bm <- microbenchmark(
    pggls(y ~ x, data = pdata, model = "within"),
    times = 50,
    unit = "microseconds"
  )

  return(median(bm$time) / 1000)  # Convert to microseconds
}

# Run benchmarks at different sizes
sizes <- list(
  c(10, 10),     # n=100
  c(50, 20),     # n=1000
  c(100, 100),   # n=10000
  c(316, 316)    # n~100000
)

cat("\n| Dataset Size | R pggls (µs) |\n")
cat("|--------------|-------------|\n")

for (size in sizes) {
  n_firms <- size[1]
  n_periods <- size[2]
  n_obs <- n_firms * n_periods

  time_us <- tryCatch({
    benchmark_pggls(n_firms, n_periods)
  }, error = function(e) {
    NA
  })

  if (!is.na(time_us)) {
    cat(sprintf("| n=%d        | %.2f        |\n", n_obs, time_us))
  } else {
    cat(sprintf("| n=%d        | Error       |\n", n_obs))
  }
}

cat("\n=== Validation Complete ===\n")
