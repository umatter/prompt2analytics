#!/usr/bin/env Rscript
# spline and approx R Benchmarks

library(microbenchmark)

set.seed(42)

cat("=== spline/approx R Benchmarks ===\n")

# spline tests
cat("\n--- spline ---\n")
sizes <- c(10, 50, 100, 500)

for (n in sizes) {
  x <- sort(runif(n, 0, 10))
  y <- sin(x) + rnorm(n, sd = 0.1)
  xout <- seq(min(x), max(x), length.out = 100)

  bm_natural <- microbenchmark(
    spline(x, y, xout = xout, method = "natural"),
    times = 100,
    unit = "microseconds"
  )

  bm_fmm <- microbenchmark(
    spline(x, y, xout = xout, method = "fmm"),
    times = 100,
    unit = "microseconds"
  )

  cat(sprintf("  n=%d (100 output pts):\n", n))
  cat(sprintf("    natural: %.2f us (median)\n", median(bm_natural$time) / 1000))
  cat(sprintf("    fmm:     %.2f us (median)\n", median(bm_fmm$time) / 1000))
}

# approx tests
cat("\n--- approx ---\n")
for (n in sizes) {
  x <- sort(runif(n, 0, 10))
  y <- x^2 + rnorm(n, sd = 0.1)
  xout <- seq(min(x), max(x), length.out = 1000)

  bm_linear <- microbenchmark(
    approx(x, y, xout = xout, method = "linear"),
    times = 100,
    unit = "microseconds"
  )

  bm_const <- microbenchmark(
    approx(x, y, xout = xout, method = "constant"),
    times = 100,
    unit = "microseconds"
  )

  cat(sprintf("  n=%d (1000 output pts):\n", n))
  cat(sprintf("    linear:   %.2f us (median)\n", median(bm_linear$time) / 1000))
  cat(sprintf("    constant: %.2f us (median)\n", median(bm_const$time) / 1000))
}

# Validation
cat("\n=== Validation ===\n")
x <- c(1, 2, 3, 4, 5)
y <- c(1, 4, 9, 16, 25)  # y = x^2

# Linear interpolation
result_approx <- approx(x, y, xout = c(1.5, 2.5, 3.5, 4.5))
cat(sprintf("approx(linear): %s\n", paste(round(result_approx$y, 4), collapse = ", ")))

# Spline interpolation
result_spline <- spline(x, y, xout = c(1.5, 2.5, 3.5, 4.5))
cat(sprintf("spline(natural): %s\n", paste(round(result_spline$y, 4), collapse = ", ")))
