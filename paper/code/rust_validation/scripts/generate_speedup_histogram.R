#!/usr/bin/env Rscript
# generate_speedup_histogram.R - Generate speedup histogram from benchmark results
# Usage: Rscript scripts/generate_speedup_histogram.R

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

# Configuration
args <- commandArgs(trailingOnly = FALSE)
script_dir <- dirname(sub("^--file=", "", args[grep("^--file=", args)]))
if (length(script_dir) == 0) script_dir <- "scripts"
base_dir <- dirname(script_dir)
results_dir <- file.path(base_dir, "results", "benchmarks")
figures_dir <- file.path(base_dir, "figures")

dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

# Categories and colors
category_colors <- c(
  "regression" = "#1f77b4",
  "econometrics" = "#ff7f0e",
  "ml" = "#2ca02c",
  "munging" = "#9467bd"
)

# Method to category mapping
method_categories <- list(
  ols = "regression",
  ols_hc0 = "regression",
  ols_hc1 = "regression",
  ols_hc2 = "regression",
  ols_hc3 = "regression",
  ols_robust = "regression",
  panel_fe = "econometrics",
  panel_re = "econometrics",
  logit = "econometrics",
  probit = "econometrics",
  kmeans = "ml",
  pca = "ml",
  dbscan = "ml",
  hierarchical = "ml",
  sort = "munging",
  filter = "munging",
  group_by = "munging",
  select = "munging",
  standardize = "munging",
  lag = "munging"
)

# Collect speedup data
collect_speedups <- function(target_n = 100000) {
  r_files <- list.files(results_dir, pattern = sprintf("^r_.*_n%d\\.json$", target_n), full.names = TRUE)

  speedups <- data.frame(
    method = character(),
    speedup = numeric(),
    category = character(),
    r_us = numeric(),
    rust_us = numeric(),
    stringsAsFactors = FALSE
  )

  for (r_file in r_files) {
    tryCatch({
      r_data <- fromJSON(r_file)

      # Extract method from filename
      basename <- basename(r_file)
      parts <- strsplit(gsub("\\.json$", "", basename), "_")[[1]]
      method <- paste(parts[2:(length(parts)-1)], collapse = "_")

      # Find Rust file
      rust_file <- gsub("/r_", "/rust_", r_file)
      if (!file.exists(rust_file)) {
        cat(sprintf("Warning: No Rust file for %s\n", method))
        next
      }

      rust_data <- fromJSON(rust_file)

      # Parse timing
      r_median <- r_data$timing$median_us
      if (is.null(r_median)) r_median <- NA

      # Handle hyperfine format
      if ("results" %in% names(rust_data)) {
        rust_median <- rust_data$results[[1]]$median * 1e6
      } else {
        rust_median <- rust_data$timing$median_us
      }
      if (is.null(rust_median)) rust_median <- NA

      if (!is.na(r_median) && !is.na(rust_median) && rust_median > 0) {
        speedup <- r_median / rust_median
        category <- method_categories[[method]]
        if (is.null(category)) category <- "other"

        speedups <- rbind(speedups, data.frame(
          method = method,
          speedup = speedup,
          category = category,
          r_us = r_median,
          rust_us = rust_median,
          stringsAsFactors = FALSE
        ))

        cat(sprintf("  %s: %.2fx (R: %.0fus, Rust: %.0fus)\n",
                    method, speedup, r_median, rust_median))
      }

    }, error = function(e) {
      cat(sprintf("Warning: Failed to parse %s: %s\n", r_file, e$message))
    })
  }

  return(speedups)
}

# Generate histogram
cat("Collecting speedup data from benchmark results...\n")
speedups <- collect_speedups(100000)

if (nrow(speedups) == 0) {
  cat("No benchmark results found. Run ./scripts/run_benchmark.sh first.\n")
  quit(status = 1)
}

# Sort by speedup
speedups <- speedups[order(-speedups$speedup), ]

# Create plot
cat("\nGenerating histogram...\n")

# Histogram plot
p1 <- ggplot(speedups, aes(x = speedup)) +
  geom_histogram(bins = 15, fill = "#3498db", color = "white", alpha = 0.8) +
  geom_vline(xintercept = 1, color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = median(speedups$speedup), color = "darkgreen",
             linetype = "solid", linewidth = 1) +
  annotate("rect", xmin = 0, xmax = 1, ymin = -Inf, ymax = Inf,
           alpha = 0.1, fill = "red") +
  annotate("rect", xmin = 1, xmax = Inf, ymin = -Inf, ymax = Inf,
           alpha = 0.1, fill = "green") +
  annotate("text", x = 0.5, y = Inf, label = "R faster", vjust = 2,
           color = "darkred", size = 3) +
  annotate("text", x = max(speedups$speedup) * 0.7, y = Inf,
           label = "Rust faster", vjust = 2, color = "darkgreen", size = 3) +
  labs(
    x = "Speedup Factor (R time / Rust time)",
    y = "Number of Methods",
    title = sprintf("Distribution of Rust vs R Speedups (n=100,000, %d methods)",
                    nrow(speedups)),
    subtitle = sprintf("Median: %.1fx | Mean: %.1fx",
                       median(speedups$speedup), mean(speedups$speedup))
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title = element_text(size = 11)
  )

# Bar chart by method
speedups$method_label <- gsub("_", " ", speedups$method)
speedups$method_label <- factor(speedups$method_label,
                                levels = rev(speedups$method_label))

p2 <- ggplot(speedups, aes(x = method_label, y = speedup, fill = category)) +
  geom_col(alpha = 0.9) +
  geom_hline(yintercept = 1, color = "red", linetype = "dashed", linewidth = 1) +
  geom_text(aes(label = sprintf("%.1fx", speedup)),
            hjust = -0.1, size = 3) +
  scale_fill_manual(values = category_colors, name = "Category") +
  coord_flip(clip = "off") +
  labs(
    x = "",
    y = "Speedup Factor",
    title = "Speedup by Method"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title = element_text(size = 11),
    legend.position = "bottom"
  ) +
  expand_limits(y = max(speedups$speedup) * 1.15)

# Save plots
ggsave(file.path(figures_dir, "speedup_histogram.png"), p1,
       width = 8, height = 5, dpi = 150)
ggsave(file.path(figures_dir, "speedup_histogram.pdf"), p1,
       width = 8, height = 5)
cat(sprintf("Histogram saved to: %s\n", file.path(figures_dir, "speedup_histogram.png")))

ggsave(file.path(figures_dir, "speedup_by_method.png"), p2,
       width = 10, height = 8, dpi = 150)
ggsave(file.path(figures_dir, "speedup_by_method.pdf"), p2,
       width = 10, height = 8)
cat(sprintf("Bar chart saved to: %s\n", file.path(figures_dir, "speedup_by_method.png")))

# Print summary
cat("\nSummary Statistics:\n")
cat(sprintf("  Total methods: %d\n", nrow(speedups)))
cat(sprintf("  Mean speedup: %.2fx\n", mean(speedups$speedup)))
cat(sprintf("  Median speedup: %.2fx\n", median(speedups$speedup)))
cat(sprintf("  Min speedup: %.2fx (%s)\n",
            min(speedups$speedup), speedups$method[which.min(speedups$speedup)]))
cat(sprintf("  Max speedup: %.2fx (%s)\n",
            max(speedups$speedup), speedups$method[which.max(speedups$speedup)]))
cat(sprintf("  Methods faster in Rust: %d\n", sum(speedups$speedup > 1)))
cat(sprintf("  Methods faster in R: %d\n", sum(speedups$speedup < 1)))
