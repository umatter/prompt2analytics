#!/usr/bin/env Rscript
# Cochran-Mantel-Haenszel Test R Benchmark
# Compares R implementation performance against p2a Rust

# Try to load microbenchmark, fall back to system.time
use_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Cochran-Mantel-Haenszel Test R Benchmarks ===\n\n")

# Generate stratified 2x2 table data
generate_cmh_array <- function(n_strata) {
  # Create array of n_strata 2x2 tables
  data <- array(0, dim = c(2, 2, n_strata))

  for (k in 1:n_strata) {
    base <- runif(1, 10, 60)
    data[1, 1, k] <- round(base * runif(1, 0.5, 1.5))
    data[1, 2, k] <- round(base * runif(1, 0.8, 1.8))
    data[2, 1, k] <- round(base * runif(1, 0.3, 1.3))
    data[2, 2, k] <- round(base * runif(1, 1.0, 2.0))
  }

  data
}

# Test configurations: number of strata
configs <- c(5, 10, 50, 100)

cat("Benchmarking mantelhaen.test with varying number of strata\n")
cat("-------------------------------------------------------------------\n\n")

for (n_strata in configs) {
  data <- generate_cmh_array(n_strata)

  cat(sprintf("n_strata=%d:\n", n_strata))

  if (use_microbenchmark) {
    # Run benchmark with microbenchmark
    bm <- microbenchmark(
      mantelhaen.test(data),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
    cat(sprintf("  Median: %.2f µs\n", med))
    cat(sprintf("  Mean:   %.2f µs\n", mean(bm$time) / 1000))
    cat(sprintf("  Min:    %.2f µs\n", min(bm$time) / 1000))
    cat(sprintf("  Max:    %.2f µs\n\n", max(bm$time) / 1000))
  } else {
    # Fallback: use system.time with 100 iterations
    n_iter <- 100
    timing <- system.time(for(i in 1:n_iter) {
      mantelhaen.test(data)
    })
    avg_us <- (timing["elapsed"] / n_iter) * 1e6
    cat(sprintf("  Mean (100 iter): %.2f µs\n\n", avg_us))
  }
}

cat("\n=== Validation Output ===\n\n")

# Classic Rabbits example from R documentation
Rabbits <- array(c(
  0, 0, 6, 5,
  3, 0, 3, 6,
  6, 2, 0, 4,
  5, 6, 1, 0,
  2, 5, 0, 0
), dim = c(2, 2, 5),
dimnames = list(
  Delay = c("None", "1.5h"),
  Response = c("Cured", "Died"),
  Penicillin.Level = c("1/8", "1/4", "1/2", "1", "4")
))

cat("Rabbits data (5 strata):\n")
print(Rabbits)
cat("\n")

result <- mantelhaen.test(Rabbits)
cat("mantelhaen.test result:\n")
print(result)

# Show intermediate computations
cat("\n\nIntermediate values:\n")
for (k in 1:5) {
  cat(sprintf("Stratum %d:\n", k))
  cat(sprintf("  Table: %s\n", paste(Rabbits[,,k], collapse=", ")))

  table_k <- Rabbits[,,k]
  a <- table_k[1,1]
  b <- table_k[1,2]
  c <- table_k[2,1]
  d <- table_k[2,2]
  n <- sum(table_k)

  if (n > 1) {
    exp_a <- (a + b) * (a + c) / n
    var_a <- (a + b) * (c + d) * (a + c) * (b + d) / (n^2 * (n - 1))
    cat(sprintf("  a=%d, b=%d, c=%d, d=%d, n=%d\n", a, b, c, d, n))
    cat(sprintf("  E[a] = %.4f, Var(a) = %.4f\n", exp_a, var_a))
    cat(sprintf("  OR = (a*d)/(b*c) = %.4f\n", (a*d)/(b*c)))
  }
  cat("\n")
}
