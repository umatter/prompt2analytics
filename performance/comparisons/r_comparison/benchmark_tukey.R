#!/usr/bin/env Rscript
# Tukey HSD R Benchmark
# Compares R implementation performance against p2a Rust

# Check if microbenchmark is available
has_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== Tukey HSD R Benchmarks ===\n\n")

# Helper function to generate grouped data
generate_tukey_data <- function(n_per_group, n_groups) {
  n_total <- n_per_group * n_groups

  # Generate response with group effects
  group <- rep(1:n_groups, each = n_per_group)
  y <- rnorm(n_total) + (group - 1) * 2  # 2 units shift per group

  data.frame(
    y = y,
    group = factor(group)
  )
}

# Benchmark configurations
configs <- list(
  list(n_per_group = 50, n_groups = 2, label = "n=100, k=2"),
  list(n_per_group = 200, n_groups = 5, label = "n=1000, k=5"),
  list(n_per_group = 1000, n_groups = 10, label = "n=10000, k=10")
)

cat("Tukey HSD Performance Benchmarks:\n")
cat("----------------------------------\n")

for (cfg in configs) {
  data <- generate_tukey_data(cfg$n_per_group, cfg$n_groups)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      {
        fit <- aov(y ~ group, data = data)
        TukeyHSD(fit)
      },
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert from nanoseconds to microseconds
    cat(sprintf("  %s: %.2f us (median of 100)\n", cfg$label, med))
  } else {
    # Fallback without microbenchmark
    timing <- system.time(replicate(50, {
      fit <- aov(y ~ group, data = data)
      TukeyHSD(fit)
    }))
    avg_us <- timing["elapsed"] * 1000000 / 50
    cat(sprintf("  %s: %.2f us (average of 50)\n", cfg$label, avg_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test Case 1: Basic three groups
y <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
group <- factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C"))

fit <- aov(y ~ group)
tukey <- TukeyHSD(fit)

cat("Test 1 - Three groups (equal sizes):\n")
cat("ANOVA Summary:\n")
print(summary(fit))
cat("\nTukey HSD Results:\n")
print(tukey)

# Test Case 2: Unequal sample sizes
y2 <- c(
  10, 11, 12,           # A: n=3
  20, 21, 19, 20, 21,   # B: n=5
  30, 31, 29, 30        # C: n=4
)
group2 <- factor(c("A", "A", "A", "B", "B", "B", "B", "B", "C", "C", "C", "C"))

fit2 <- aov(y2 ~ group2)
tukey2 <- TukeyHSD(fit2)

cat("\nTest 2 - Unequal sample sizes:\n")
cat("Group sizes: A=3, B=5, C=4\n")
print(tukey2)

cat("\n=== Done ===\n")
