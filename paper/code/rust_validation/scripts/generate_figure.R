#!/usr/bin/env Rscript
# generate_figure.R - Generate benchmark comparison figure from validation results
#
# This creates a bar chart showing speedup factors at different sample sizes

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})

# Find script directory
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  return(getwd())
}

script_dir <- get_script_dir()
base_dir <- dirname(script_dir)
summary_file <- file.path(base_dir, "results/summaries/benchmark_summary.json")
output_file <- file.path(base_dir, "figures/benchmark_speedup.pdf")

cat("=== Generating Benchmark Figure ===\n")
cat("Reading:", summary_file, "\n")

# Read benchmark data
data <- fromJSON(summary_file)

# Extract and reshape data
method_data <- lapply(names(data$methods), function(method) {
  method_results <- data$methods[[method]]
  # Handle both list and data.frame formats from jsonlite
  if (is.data.frame(method_results)) {
    data.frame(
      method = toupper(method),
      n = method_results$n,
      r_ms = method_results$r_median_us / 1000,
      rust_ms = method_results$rust_median_us / 1000,
      speedup = method_results$speedup
    )
  } else {
    data.frame(
      method = toupper(method),
      n = sapply(method_results, function(x) x$n),
      r_ms = sapply(method_results, function(x) x$r_median_us) / 1000,
      rust_ms = sapply(method_results, function(x) x$rust_median_us) / 1000,
      speedup = sapply(method_results, function(x) x$speedup)
    )
  }
})

df <- do.call(rbind, method_data)

# Rename methods for display
df$method <- factor(df$method,
                    levels = c("OLS", "PANEL_FE", "LOGIT", "KMEANS", "PCA", "SORT", "FILTER", "GROUP_BY"),
                    labels = c("OLS", "Panel FE", "Logit", "K-Means", "PCA", "Sort", "Filter", "Group By"))

# Filter to largest sample sizes only (most meaningful comparisons)
df_large <- df %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup()

cat("\nLarge-scale benchmark results:\n")
print(df_large)

# Create speedup bar chart
colors <- c("R" = "#0097A7", "Rust" = "#E65100")

# Reshape for stacked bars
df_long <- df_large %>%
  select(method, n, r_ms, rust_ms) %>%
  pivot_longer(cols = c(r_ms, rust_ms),
               names_to = "implementation",
               values_to = "time_ms") %>%
  mutate(implementation = ifelse(implementation == "r_ms", "R", "Rust"))

# Create comparison figure
p <- ggplot(df_long, aes(x = method, y = time_ms, fill = implementation)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(data = df_large, aes(x = method, y = pmax(r_ms, rust_ms) + 10,
                                   label = sprintf("%.1fx", speedup),
                                   fill = NULL),
            vjust = 0, size = 3.5) +
  scale_fill_manual(values = colors, name = "Implementation") +
  scale_y_log10(name = "Execution Time (ms, log scale)",
                labels = scales::comma) +
  labs(x = "Method",
       title = "R vs Rust Performance Comparison",
       subtitle = sprintf("Largest sample sizes (n=%s)",
                         format(max(df_large$n), big.mark = ","))) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 10)
  )

# Save figure
dir.create(dirname(output_file), showWarnings = FALSE, recursive = TRUE)
ggsave(output_file, p, width = 8, height = 5, dpi = 300)
cat("\nFigure saved to:", output_file, "\n")

# Also save as PNG
png_file <- sub("\\.pdf$", ".png", output_file)
ggsave(png_file, p, width = 8, height = 5, dpi = 300)
cat("Also saved as:", png_file, "\n")

# Create a detailed timing table for paper
cat("\n=== Detailed Results for Paper ===\n")
df_table <- df %>%
  mutate(
    speedup_label = ifelse(speedup >= 1,
                           sprintf("%.1fx faster", speedup),
                           sprintf("%.1fx slower", 1/speedup))
  ) %>%
  select(method, n, r_ms, rust_ms, speedup_label)
print(df_table, n = 30)
