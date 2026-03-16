#!/usr/bin/env Rscript
# medpolish, cmdscale, cutree, isoreg, loglin R Benchmarks
# Compares R implementations against p2a-core Rust

set.seed(42)

# Simple benchmark function using system.time
benchmark_fn <- function(expr, times = 100) {
  results <- numeric(times)
  for (i in 1:times) {
    results[i] <- system.time(eval(expr))[["elapsed"]]
  }
  median(results) * 1e6  # Convert to microseconds
}

# ============================================================================
# medpolish benchmarks
# ============================================================================
cat("=== medpolish R Benchmarks ===\n")

for (size in c(10, 32, 100)) {
  nrow <- size
  ncol <- size
  data <- matrix(rnorm(nrow * ncol, mean = 50, sd = 10), nrow = nrow, ncol = ncol)

  cat(sprintf("\n%dx%d matrix:\n", nrow, ncol))
  med <- benchmark_fn(quote(medpolish(data)), times = 50)
  cat(sprintf("  medpolish: %.2f us (median)\n", med))
}

# ============================================================================
# isoreg benchmarks
# ============================================================================
cat("\n=== isoreg R Benchmarks ===\n")

for (n in c(100, 1000, 10000)) {
  x <- 1:n
  y <- x * 0.5 + rnorm(n, sd = 5)

  cat(sprintf("\nn=%d:\n", n))
  med <- benchmark_fn(quote(isoreg(x, y)), times = 50)
  cat(sprintf("  isoreg: %.2f us (median)\n", med))
}

# ============================================================================
# loglin benchmarks
# ============================================================================
cat("\n=== loglin R Benchmarks ===\n")

# 2x2 table
cat("\n2x2 table:\n")
table_2x2 <- array(c(10, 15, 20, 25), dim = c(2, 2))
med <- benchmark_fn(quote(loglin(table_2x2, list(1, 2))), times = 50)
cat(sprintf("  loglin: %.2f us (median)\n", med))

# 2x3 table
cat("\n2x3 table:\n")
table_2x3 <- array(c(10, 15, 20, 25, 30, 35), dim = c(2, 3))
med <- benchmark_fn(quote(loglin(table_2x3, list(1, 2))), times = 50)
cat(sprintf("  loglin: %.2f us (median)\n", med))

# 2x2x2 table
cat("\n2x2x2 table:\n")
table_2x2x2 <- array(c(10, 15, 20, 25, 30, 35, 40, 45), dim = c(2, 2, 2))
med <- benchmark_fn(quote(loglin(table_2x2x2, list(c(1, 2), c(1, 3), c(2, 3)))), times = 50)
cat(sprintf("  loglin: %.2f us (median)\n", med))

# ============================================================================
# cmdscale benchmarks
# ============================================================================
cat("\n=== cmdscale R Benchmarks ===\n")

for (n in c(100, 1000)) {
  # Generate random points and compute distance matrix
  # Cap at n=1000: cmdscale is O(n^3) eigendecomposition, n=10000 is impractical
  points <- matrix(rnorm(n * 2), ncol = 2)
  d <- dist(points)

  cat(sprintf("\nn=%d:\n", n))
  med <- benchmark_fn(quote(cmdscale(d, k = 2, eig = TRUE)), times = 50)
  cat(sprintf("  cmdscale: %.2f us (median)\n", med))
}

# ============================================================================
# cutree benchmarks
# ============================================================================
cat("\n=== cutree R Benchmarks ===\n")

for (n in c(100, 1000)) {
  # Cap at n=1000: hclust requires O(n^2) distance matrix, impractical at n=10000
  # Generate random points and perform hierarchical clustering
  points <- matrix(rnorm(n * 2), ncol = 2)
  hc <- hclust(dist(points), method = "complete")

  cat(sprintf("\nn=%d:\n", n))
  med <- benchmark_fn(quote(cutree(hc, k = 5)), times = 50)
  cat(sprintf("  cutree: %.2f us (median)\n", med))
}

# ============================================================================
# Validation section
# ============================================================================
cat("\n=== Validation Results ===\n")

# medpolish validation
cat("\nmedpolish validation:\n")
test_data <- matrix(c(
  8.0, 6.0, 7.5,
  5.2, 4.0, 5.5,
  6.8, 5.3, 6.5
), nrow = 3, byrow = TRUE)
mp <- medpolish(test_data)
cat(sprintf("  Overall: %.6f\n", mp$overall))
cat(sprintf("  Row effects: %s\n", paste(round(mp$row, 4), collapse = ", ")))
cat(sprintf("  Col effects: %s\n", paste(round(mp$col, 4), collapse = ", ")))

# isoreg validation
cat("\nisoreg validation:\n")
y <- c(1, 0, 4, 3, 3, 5, 4, 2, 0)
ir <- isoreg(y)
cat(sprintf("  y:  %s\n", paste(y, collapse = ", ")))
cat(sprintf("  yf: %s\n", paste(round(ir$yf, 4), collapse = ", ")))

# loglin validation
cat("\nloglin validation (2x2 independence):\n")
table_val <- array(c(10, 20, 30, 40), dim = c(2, 2))
ll <- loglin(table_val, list(1, 2))
cat(sprintf("  LRT: %.6f\n", ll$lrt))
cat(sprintf("  df: %d\n", ll$df))

# cmdscale validation
cat("\ncmdscale validation:\n")
points <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
d <- dist(points)
mds <- cmdscale(d, k = 2, eig = TRUE)
cat(sprintf("  GOF: %.6f, %.6f\n", mds$GOF[1], mds$GOF[2]))

# cutree validation
cat("\ncutree validation:\n")
hc_val <- hclust(dist(matrix(c(1, 2, 3, 1.1, 2.1, 3.1), ncol = 2, byrow = TRUE)), method = "complete")
ct <- cutree(hc_val, k = 2)
cat(sprintf("  Cluster assignments: %s\n", paste(ct, collapse = ", ")))
