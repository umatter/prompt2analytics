#!/usr/bin/env Rscript
# Bartlett's Test R Benchmark
# Compares R implementation performance against p2a Rust

# Check if microbenchmark is available
has_microbenchmark <- requireNamespace("microbenchmark", quietly = TRUE)

if (has_microbenchmark) {
    library(microbenchmark)

    set.seed(42)

    # Benchmark at different dataset sizes
    sizes <- c(100, 1000, 10000)
    k_groups <- c(5, 10, 20)  # Number of groups for each size

    cat("=== Bartlett's Test R Benchmarks (microbenchmark) ===\n\n")

    for (i in seq_along(sizes)) {
        n <- sizes[i]
        k <- k_groups[i]

        # Generate data with k groups
        # Each group has roughly n/k observations
        n_per_group <- n %/% k

        # Generate data with varying variances across groups
        values <- numeric(0)
        groups <- character(0)

        for (g in 1:k) {
            # Each group has different variance: var ~ g^2
            group_vals <- rnorm(n_per_group, mean = 10 * g, sd = g)
            values <- c(values, group_vals)
            groups <- c(groups, rep(paste0("G", g), n_per_group))
        }

        # Convert to factor
        groups <- factor(groups)

        bm <- microbenchmark(
            bartlett.test(values ~ groups),
            times = 100,
            unit = "microseconds"
        )

        med <- median(bm$time) / 1000  # Convert nanoseconds to microseconds
        cat(sprintf("  n=%d, k=%d: %.2f us (median of 100 runs)\n", n, k, med))
    }
} else {
    cat("=== Bartlett's Test R Benchmarks (system.time fallback) ===\n\n")
    cat("Note: microbenchmark not available, using system.time\n\n")

    set.seed(42)

    sizes <- c(100, 1000, 10000)
    k_groups <- c(5, 10, 20)

    for (i in seq_along(sizes)) {
        n <- sizes[i]
        k <- k_groups[i]

        n_per_group <- n %/% k

        values <- numeric(0)
        groups <- character(0)

        for (g in 1:k) {
            group_vals <- rnorm(n_per_group, mean = 10 * g, sd = g)
            values <- c(values, group_vals)
            groups <- c(groups, rep(paste0("G", g), n_per_group))
        }

        groups <- factor(groups)

        # Warm-up
        invisible(bartlett.test(values ~ groups))

        # Time 50 iterations
        timing <- system.time(replicate(50, { bartlett.test(values ~ groups) }))
        avg_us <- (timing["elapsed"] * 1000 * 1000) / 50  # Convert to microseconds

        cat(sprintf("  n=%d, k=%d: %.2f us (avg of 50 runs)\n", n, k, avg_us))
    }
}

cat("\n=== Validation Test ===\n")
# Run the validation case from the Rust test
x <- c(1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 5, 10, 15, 20, 25)
g <- factor(c(rep("A", 5), rep("B", 5), rep("C", 5)))
result <- bartlett.test(x ~ g)
cat("\nTest case: x ~ g\n")
cat(sprintf("K-squared = %.6f\n", result$statistic))
cat(sprintf("df = %d\n", result$parameter))
cat(sprintf("p-value = %.6f\n", result$p.value))
cat("\nGroup variances:\n")
print(tapply(x, g, var))
