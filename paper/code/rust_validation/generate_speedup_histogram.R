#!/usr/bin/env Rscript
# generate_speedup_histogram.R - Generate histogram of Rust/R speedup ratios
# This script reads benchmark_summary.json and creates a histogram figure for the paper

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
})

# Read benchmark summary
summary_path <- "results/summaries/benchmark_summary.json"
if (!file.exists(summary_path)) {
  stop("Benchmark summary not found. Run benchmarks first.")
}

data <- fromJSON(summary_path)

# Extract speedups at n=100,000 (computation-dominated regime)
# Also include panel methods at n=100,000
speedups_100k <- data.frame(
  method = character(),
  category = character(),
  speedup = numeric(),
  stringsAsFactors = FALSE
)

# Method category mapping
categories <- list(
  ols = "Regression",
  ols_hc0 = "Regression",
  ols_hc1 = "Regression",
  ols_hc2 = "Regression",
  ols_hc3 = "Regression",
  ols_robust = "Regression",
  panel_fe = "Panel",
  panel_re = "Panel",
  logit = "Discrete",
  probit = "Discrete",
  kmeans = "ML",
  pca = "ML",
  dbscan = "ML",
  hierarchical = "ML",
  sort = "Munging",
  filter = "Munging",
  group_by = "Munging",
  select = "Munging",
  standardize = "Munging",
  lag = "Munging"
)

for (method_name in names(data$methods)) {
  method_data <- data$methods[[method_name]]

  # Find the largest sample size entry (usually 100000)
  if (is.data.frame(method_data)) {
    # Get the row with largest n
    max_idx <- which.max(method_data$n)
    if (length(max_idx) > 0) {
      speedup <- method_data$speedup[max_idx]
      n <- method_data$n[max_idx]

      # Only include if n >= 100000 or it's a panel method at n >= 100000
      if (n >= 100000 || (grepl("panel", method_name) && n >= 100000)) {
        cat_name <- categories[[method_name]]
        if (is.null(cat_name)) cat_name <- "Other"

        speedups_100k <- rbind(speedups_100k, data.frame(
          method = method_name,
          category = cat_name,
          speedup = speedup,
          stringsAsFactors = FALSE
        ))
      }
    }
  }
}

if (nrow(speedups_100k) == 0) {
  stop("No benchmark data found at n=100,000")
}

cat("Speedup data at n=100,000:\n")
print(speedups_100k)

# Create a nice method name mapping
method_labels <- c(
  "ols" = "OLS",
  "ols_hc0" = "OLS (HC0)",
  "ols_hc1" = "OLS (HC1)",
  "ols_hc2" = "OLS (HC2)",
  "ols_hc3" = "OLS (HC3)",
  "ols_robust" = "OLS (Robust)",
  "panel_fe" = "Panel FE",
  "panel_re" = "Panel RE",
  "logit" = "Logit",
  "probit" = "Probit",
  "kmeans" = "K-Means",
  "pca" = "PCA",
  "dbscan" = "DBSCAN",
  "hierarchical" = "Hierarchical",
  "sort" = "Sort",
  "filter" = "Filter",
  "group_by" = "Group By",
  "select" = "Select",
  "standardize" = "Standardize",
  "lag" = "Lag"
)

speedups_100k$method_label <- sapply(speedups_100k$method, function(m) {
  if (m %in% names(method_labels)) method_labels[m] else m
})

# Calculate summary statistics
cat("\n\nSummary Statistics (n=100,000):\n")
cat(sprintf("  Methods benchmarked: %d\n", nrow(speedups_100k)))
cat(sprintf("  Mean speedup: %.2fx\n", mean(speedups_100k$speedup)))
cat(sprintf("  Median speedup: %.2fx\n", median(speedups_100k$speedup)))
cat(sprintf("  Min speedup: %.2fx (%s)\n", min(speedups_100k$speedup),
            speedups_100k$method_label[which.min(speedups_100k$speedup)]))
cat(sprintf("  Max speedup: %.2fx (%s)\n", max(speedups_100k$speedup),
            speedups_100k$method_label[which.max(speedups_100k$speedup)]))
cat(sprintf("  Methods faster in Rust (>1x): %d (%.0f%%)\n",
            sum(speedups_100k$speedup > 1),
            100 * sum(speedups_100k$speedup > 1) / nrow(speedups_100k)))

# Calculate log2 speedup for better visualization (centered at 0 = equal performance)
speedups_100k$log2_speedup <- log2(speedups_100k$speedup)

# Filter out methods with speedup of 0 or near-0 (I/O dominated, not meaningful comparison)
# Note: speedup < 0.01 means Rust took >100x longer - likely CLI overhead dominated
speedups_plot <- speedups_100k[speedups_100k$speedup >= 0.01, ]

cat(sprintf("\nFiltered to %d methods with speedup >= 0.01\n", nrow(speedups_plot)))

# Create histogram of speedups
p <- ggplot(speedups_plot, aes(x = speedup, fill = category)) +
  geom_histogram(bins = 12, color = "black", alpha = 0.7) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
  scale_x_log10(
    breaks = c(0.1, 0.2, 0.5, 1, 2, 5, 10, 15),
    labels = c("0.1x", "0.2x", "0.5x", "1x", "2x", "5x", "10x", "15x")
  ) +
  scale_fill_brewer(palette = "Set2", name = "Category") +
  labs(
    title = "Distribution of Rust/R Speedup Ratios (n = 100,000)",
    subtitle = sprintf("Median: %.1fx, %d/%d methods faster in Rust",
                       median(speedups_plot$speedup),
                       sum(speedups_plot$speedup > 1),
                       nrow(speedups_plot)),
    x = "Speedup (R time / Rust time)",
    y = "Number of Methods"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  annotate("text", x = 0.15, y = Inf, label = "R Faster", vjust = 2, hjust = 0, size = 3) +
  annotate("text", x = 6, y = Inf, label = "Rust Faster", vjust = 2, hjust = 1, size = 3)

# Save the plot
ggsave("figures/benchmark_speedup_histogram.pdf", p, width = 8, height = 5)
ggsave("figures/benchmark_speedup_histogram.png", p, width = 8, height = 5, dpi = 300)

cat("\n\nHistogram saved to:\n")
cat("  figures/benchmark_speedup_histogram.pdf\n")
cat("  figures/benchmark_speedup_histogram.png\n")

# Also create a bar chart showing individual method speedups
speedups_plot$method_label <- factor(speedups_plot$method_label,
                                     levels = speedups_plot$method_label[order(speedups_plot$speedup)])

p2 <- ggplot(speedups_plot, aes(x = method_label, y = speedup, fill = category)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.8) +
  coord_flip() +
  scale_fill_brewer(palette = "Set2", name = "Category") +
  labs(
    title = "Rust/R Speedup by Method (n = 100,000)",
    x = NULL,
    y = "Speedup (R time / Rust time)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "right",
    panel.grid.minor = element_blank()
  ) +
  annotate("text", x = 0.5, y = 1.2, label = "Rust Faster", size = 3, hjust = 0)

ggsave("figures/benchmark_speedup_bars.pdf", p2, width = 8, height = 6)
ggsave("figures/benchmark_speedup_bars.png", p2, width = 8, height = 6, dpi = 300)

cat("  figures/benchmark_speedup_bars.pdf\n")
cat("  figures/benchmark_speedup_bars.png\n")

# Output summary data as JSON for paper integration
summary_out <- list(
  generated = Sys.time(),
  n_methods = nrow(speedups_100k),
  n_100k = 100000,
  summary = list(
    mean_speedup = mean(speedups_100k$speedup),
    median_speedup = median(speedups_100k$speedup),
    min_speedup = min(speedups_100k$speedup),
    max_speedup = max(speedups_100k$speedup),
    pct_rust_faster = 100 * sum(speedups_100k$speedup > 1) / nrow(speedups_100k)
  ),
  by_category = aggregate(speedup ~ category, speedups_100k, function(x) {
    list(mean = mean(x), median = median(x), n = length(x))
  }),
  methods = speedups_100k[, c("method", "category", "speedup")]
)

write(toJSON(summary_out, auto_unbox = TRUE, pretty = TRUE),
      "results/summaries/speedup_histogram_data.json")
cat("\nSummary data saved to: results/summaries/speedup_histogram_data.json\n")
