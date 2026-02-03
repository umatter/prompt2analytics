# ============================================================================
# fig_benchmark_histogram.R
# Histogram showing distribution of speedup factors across all methods
# ============================================================================

## SETUP ----
library(tidyverse)
library(jsonlite)

INPUT <- "rust_validation/results/summaries/"
OUTPUT <- "../figures/"

## DATA IMPORT AND PREPARATION ----
summary_data <- fromJSON(paste0(INPUT, "benchmark_summary.json"), simplifyVector = FALSE)

# Extract methods data and flatten (handle type inconsistencies)
methods_list <- summary_data$methods
benchmark_df <- map_dfr(names(methods_list), function(method) {
  method_results <- methods_list[[method]]
  map_dfr(method_results, function(r) {
    tibble(
      method = method,
      n = r$n,
      speedup = r$speedup
    )
  })
})

# Filter to n=100,000 (or max n for each method)
plot_data <- benchmark_df %>%
  group_by(method) %>%
  filter(n == max(n)) %>%
  ungroup() %>%
  filter(speedup > 0) %>%
  mutate(
    rust_faster = speedup > 1,
    speedup_label = case_when(
      speedup >= 1 ~ sprintf("%.1f×", speedup),
      TRUE ~ sprintf("%.2f×", speedup)
    )
  )

## PLOT ----
# Create histogram with log-scale x-axis centered on 1×
p <- ggplot(plot_data, aes(x = speedup, fill = rust_faster)) +
  geom_histogram(bins = 15, color = "white", linewidth = 0.3) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  scale_x_log10(
    name = "Speedup Factor (Rust / R)",
    breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 6),
    labels = c("0.1×", "0.25×", "0.5×", "1×", "2×", "4×", "6×")
  ) +
  scale_y_continuous(name = "Number of Methods") +
  scale_fill_manual(
    values = c("FALSE" = "#0097A7", "TRUE" = "#E65100"),
    labels = c("FALSE" = "R faster", "TRUE" = "Rust faster"),
    name = NULL
  ) +
  annotate(
    "text", x = 0.15, y = Inf, label = "R faster",
    vjust = 2, hjust = 0.5, color = "#0097A7", fontface = "bold", size = 3.5
  ) +
  annotate(
    "text", x = 4, y = Inf, label = "Rust faster",
    vjust = 2, hjust = 0.5, color = "#E65100", fontface = "bold", size = 3.5
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.margin = margin(15, 15, 10, 10)
  )

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "benchmark_histogram.pdf"),
  plot = p,
  width = 7,
  height = 4
)

ggsave(
  paste0(OUTPUT, "benchmark_histogram.png"),
  plot = p,
  width = 7,
  height = 4,
  dpi = 300
)

message("Created: ", OUTPUT, "benchmark_histogram.pdf")
message("Created: ", OUTPUT, "benchmark_histogram.png")

# Print summary stats
cat("\nSpeedup Distribution Summary:\n")
cat(sprintf("  Methods with Rust faster (>1×): %d\n", sum(plot_data$rust_faster)))
cat(sprintf("  Methods with R faster (<1×): %d\n", sum(!plot_data$rust_faster)))
cat(sprintf("  Median speedup: %.2f×\n", median(plot_data$speedup)))
cat(sprintf("  Mean speedup: %.2f×\n", mean(plot_data$speedup)))
cat(sprintf("  Max speedup: %.2f×\n", max(plot_data$speedup)))
