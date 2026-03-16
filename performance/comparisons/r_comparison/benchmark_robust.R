#!/usr/bin/env Rscript
# density, ecdf, fivenum, IQR, mad R Benchmarks

library(microbenchmark)

set.seed(42)

cat("=== Robust Statistics R Benchmarks ===\n")

sizes <- c(100, 1000, 10000)

for (n in sizes) {
  x <- rnorm(n)

  cat(sprintf("\nn=%d:\n", n))

  # density
  bm_density <- microbenchmark(
    density(x),
    times = 100,
    unit = "microseconds"
  )
  cat(sprintf("  density:  %.2f us (median)\n", median(bm_density$time) / 1000))

  # ecdf
  bm_ecdf <- microbenchmark(
    ecdf(x),
    times = 100,
    unit = "microseconds"
  )
  cat(sprintf("  ecdf:     %.2f us (median)\n", median(bm_ecdf$time) / 1000))

  # fivenum
  bm_fivenum <- microbenchmark(
    fivenum(x),
    times = 100,
    unit = "microseconds"
  )
  cat(sprintf("  fivenum:  %.2f us (median)\n", median(bm_fivenum$time) / 1000))

  # IQR
  bm_iqr <- microbenchmark(
    IQR(x),
    times = 100,
    unit = "microseconds"
  )
  cat(sprintf("  IQR:      %.2f us (median)\n", median(bm_iqr$time) / 1000))

  # mad
  bm_mad <- microbenchmark(
    mad(x),
    times = 100,
    unit = "microseconds"
  )
  cat(sprintf("  mad:      %.2f us (median)\n", median(bm_mad$time) / 1000))
}

# Validation
cat("\n=== Validation ===\n")
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
cat(sprintf("fivenum: %s\n", paste(fivenum(x), collapse = ", ")))
cat(sprintf("IQR: %.6f\n", IQR(x)))
cat(sprintf("mad: %.6f\n", mad(x)))

# density at specific points
d <- density(x)
cat(sprintf("density n: %d, bw: %.6f\n", d$n, d$bw))
