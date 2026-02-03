#!/usr/bin/env Rscript
# MANOVA R Benchmark
# Compares R implementation performance against p2a Rust

# Check if microbenchmark is available
has_microbenchmark <- require(microbenchmark, quietly = TRUE)

set.seed(42)

cat("=== MANOVA R Benchmarks ===\n\n")

# Helper function to generate multivariate data with group structure
generate_manova_data <- function(n_per_group, n_groups, n_vars) {
  n_total <- n_per_group * n_groups

  # Generate response matrix
  y <- matrix(rnorm(n_total * n_vars), nrow = n_total, ncol = n_vars)

  # Add group effects (shift means)
  group <- rep(1:n_groups, each = n_per_group)
  for (g in 1:n_groups) {
    idx <- which(group == g)
    for (v in 1:n_vars) {
      y[idx, v] <- y[idx, v] + (g - 1) * 2 + (v - 1) * 0.5
    }
  }

  # Add some within-group correlation
  for (i in 1:n_total) {
    noise <- rnorm(1) * 0.3
    y[i, ] <- y[i, ] + noise * seq(1, n_vars) / n_vars
  }

  group <- factor(group)

  list(y = y, group = group)
}

# Benchmark configurations
configs <- list(
  list(n_per_group = 33, n_groups = 3, n_vars = 2, label = "n=100, p=2, g=3"),
  list(n_per_group = 250, n_groups = 4, n_vars = 3, label = "n=1000, p=3, g=4"),
  list(n_per_group = 2000, n_groups = 5, n_vars = 5, label = "n=10000, p=5, g=5")
)

cat("MANOVA Performance Benchmarks:\n")
cat("------------------------------\n")

for (cfg in configs) {
  data <- generate_manova_data(cfg$n_per_group, cfg$n_groups, cfg$n_vars)

  if (has_microbenchmark) {
    bm <- microbenchmark(
      manova(data$y ~ data$group),
      times = 100,
      unit = "microseconds"
    )
    med <- median(bm$time) / 1000  # Convert from nanoseconds to microseconds
    cat(sprintf("  %s: %.2f us (median of 100)\n", cfg$label, med))
  } else {
    # Fallback without microbenchmark
    timing <- system.time(replicate(50, { manova(data$y ~ data$group) }))
    avg_us <- timing["elapsed"] * 1000000 / 50
    cat(sprintf("  %s: %.2f us (average of 50)\n", cfg$label, avg_us))
  }
}

cat("\n=== Validation Tests ===\n\n")

# Test Case 1: Three groups with clear separation
y1 <- c(1.0, 1.2, 0.8, 5.0, 5.2, 4.8, 9.0, 9.2, 8.8)
y2 <- c(8.0, 7.8, 8.2, 4.0, 4.2, 3.8, 1.0, 1.2, 0.8)
group <- factor(c("A", "A", "A", "B", "B", "B", "C", "C", "C"))

fit <- manova(cbind(y1, y2) ~ group)
s <- summary(fit, test = "Pillai")

cat("Test 1 - Three groups with clear separation:\n")
cat(sprintf("  Pillai's Trace: %.6f\n", s$stats[1, "Pillai"]))
cat(sprintf("  F-value: %.4f\n", s$stats[1, "approx F"]))
cat(sprintf("  df1: %d, df2: %d\n", s$stats[1, "num Df"], s$stats[1, "den Df"]))
cat(sprintf("  p-value: %.6f\n\n", s$stats[1, "Pr(>F)"]))

# All four tests
cat("All four test statistics:\n")
for (test in c("Pillai", "Wilks", "Hotelling-Lawley", "Roy")) {
  s <- summary(fit, test = test)
  # Get the column name for the statistic
  stat_col <- switch(test,
    "Pillai" = "Pillai",
    "Wilks" = "Wilks",
    "Hotelling-Lawley" = "Hotelling",
    "Roy" = "Roy"
  )
  stat_val <- tryCatch(s$stats[1, stat_col], error = function(e) NA)
  p_val <- s$stats[1, "Pr(>F)"]
  cat(sprintf("  %s: %.6f, p = %.6f\n", test, stat_val, p_val))
}

cat("\n")

# Test Case 2: No difference between groups
y <- matrix(c(
  1.0, 5.0,
  1.5, 4.5,
  2.0, 6.0,
  0.8, 5.2,
  1.2, 4.8,
  1.7, 5.5,
  1.3, 4.7,
  1.1, 5.3
), ncol = 2, byrow = TRUE)
group2 <- factor(c("A", "A", "A", "A", "B", "B", "B", "B"))

fit2 <- manova(y ~ group2)
s2 <- summary(fit2, test = "Wilks")

cat("Test 2 - No difference (overlapping groups):\n")
cat(sprintf("  Wilks' Lambda: %.6f\n", s2$stats[1, "Wilks"]))
cat(sprintf("  p-value: %.6f\n", s2$stats[1, "Pr(>F)"]))
cat(sprintf("  (Should not be significant: p > 0.05)\n\n"))

cat("=== Done ===\n")
