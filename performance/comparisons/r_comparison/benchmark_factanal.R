#!/usr/bin/env Rscript
# Factor Analysis R Benchmark
# Compares R factanal() performance against p2a Rust implementation

library(microbenchmark)

set.seed(42)

# Function to generate test data with factor structure
generate_factor_data <- function(n, p, k) {
  # Generate k latent factors
  factors <- matrix(rnorm(n * k), n, k)

  # Generate loading matrix (each var loads on one factor)
  loadings <- matrix(0, p, k)
  vars_per_factor <- p %/% k
  for (j in 1:k) {
    start <- (j - 1) * vars_per_factor + 1
    end <- min(j * vars_per_factor, p)
    loadings[start:end, j] <- runif(end - start + 1, 0.6, 0.9)
  }

  # Generate data: X = F * L' + e
  noise <- matrix(rnorm(n * p, sd = 0.3), n, p)
  data <- factors %*% t(loadings) + noise

  as.data.frame(data)
}

cat("=== Factor Analysis R Benchmarks ===\n\n")

# Different dataset sizes
configs <- list(
  list(n = 100, p = 6, k = 2),
  list(n = 500, p = 10, k = 3),
  list(n = 1000, p = 15, k = 4),
  list(n = 5000, p = 20, k = 5)
)

for (cfg in configs) {
  n <- cfg$n
  p <- cfg$p
  k <- cfg$k

  # Generate data
  data <- generate_factor_data(n, p, k)

  cat(sprintf("Dataset: n=%d, p=%d, k=%d\n", n, p, k))

  # Benchmark factanal with different rotations
  tryCatch({
    bm <- microbenchmark(
      no_rotation = factanal(data, factors = k, rotation = "none"),
      varimax = factanal(data, factors = k, rotation = "varimax"),
      promax = factanal(data, factors = k, rotation = "promax"),
      times = 50,
      unit = "microseconds"
    )

    cat("  No rotation: ", sprintf("%.2f", median(bm$time[bm$expr == "no_rotation"]) / 1000), " us (median)\n")
    cat("  Varimax:     ", sprintf("%.2f", median(bm$time[bm$expr == "varimax"]) / 1000), " us (median)\n")
    cat("  Promax:      ", sprintf("%.2f", median(bm$time[bm$expr == "promax"]) / 1000), " us (median)\n")
    cat("\n")

  }, error = function(e) {
    cat("  Error: ", conditionMessage(e), "\n\n")
  })
}

cat("=== Benchmark Complete ===\n")

# Output summary table format for documentation
cat("\n=== Summary Table (copy to validation doc) ===\n")
cat("| Dataset | R no_rotation (µs) | R varimax (µs) | R promax (µs) |\n")
cat("|---------|--------------------|-----------------|-----------------|\n")

for (cfg in configs) {
  n <- cfg$n
  p <- cfg$p
  k <- cfg$k

  data <- generate_factor_data(n, p, k)

  tryCatch({
    bm <- microbenchmark(
      no_rotation = factanal(data, factors = k, rotation = "none"),
      varimax = factanal(data, factors = k, rotation = "varimax"),
      promax = factanal(data, factors = k, rotation = "promax"),
      times = 20,
      unit = "microseconds"
    )

    t_none <- median(bm$time[bm$expr == "no_rotation"]) / 1000
    t_var <- median(bm$time[bm$expr == "varimax"]) / 1000
    t_pro <- median(bm$time[bm$expr == "promax"]) / 1000

    cat(sprintf("| n=%d, p=%d, k=%d | %.2f | %.2f | %.2f |\n", n, p, k, t_none, t_var, t_pro))

  }, error = function(e) {
    cat(sprintf("| n=%d, p=%d, k=%d | ERROR | ERROR | ERROR |\n", n, p, k))
  })
}
