#!/usr/bin/env Rscript
# generate_benchmark_figure_n10000.R
#
# Generates a two-panel box plot figure comparing R and Rust benchmark times
# at consistent N=10,000 for all methods
#
# Panel A: R execution times
# Panel B: p2a (Rust) execution times
#
# Usage: Rscript generate_benchmark_figure_n10000.R
#
# Output: paper/figures/benchmark_boxplots.pdf

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(scales)
})

# Paths - detect script location
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("--file=", "", file_arg))))
  }
  return(getwd())
}

script_dir <- get_script_dir()
paper_dir <- dirname(script_dir)
project_root <- dirname(paper_dir)

r_results_dir <- file.path(project_root, "performance/comparisons/r_comparison/results")
rust_results_dir <- file.path(project_root, "performance/results")
output_file <- file.path(paper_dir, "figures/benchmark_boxplots.pdf")

cat("=== Generating Benchmark Box Plot Figure (N=10,000) ===\n\n")

# Find the N=10000 benchmark files
find_latest_file <- function(dir, pattern) {
  files <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[which.max(file.info(files)$mtime)]
}

r_file <- find_latest_file(r_results_dir, "r_n10000_.*\\.csv$")
rust_file <- find_latest_file(rust_results_dir, "rust_n10000_.*\\.json$")

if (is.null(r_file)) stop("No R N=10000 benchmark file found")
if (is.null(rust_file)) stop("No Rust N=10000 benchmark file found")

cat("R benchmark file:", r_file, "\n")
cat("Rust benchmark file:", rust_file, "\n\n")

# Read R benchmarks
r_data <- read.csv(r_file, stringsAsFactors = FALSE) %>%
  mutate(
    implementation = "R",
    time_min_ms = time_min_us / 1000,
    time_p25_ms = time_p25_us / 1000,
    time_median_ms = time_median_us / 1000,
    time_p75_ms = time_p75_us / 1000,
    time_max_ms = time_max_us / 1000,
    method_std = case_when(
      method == "FE_plm" ~ "FE",
      TRUE ~ method
    )
  )

# Read Rust benchmarks
rust_json <- fromJSON(rust_file)

if (!"variant" %in% names(rust_json)) {
  rust_json$variant <- NA_character_
}

rust_data <- rust_json %>%
  mutate(
    implementation = "p2a (Rust)",
    time_min_ms = time_min_us / 1000,
    time_p25_ms = time_p25_us / 1000,
    time_median_ms = time_median_us / 1000,
    time_p75_ms = time_p75_us / 1000,
    time_max_ms = time_max_us / 1000,
    method_std = case_when(
      method == "OLS" & (is.na(variant) | variant == "standard") ~ "OLS",
      method == "OLS" & variant == "HC1" ~ "OLS+HC1",
      method == "FixedEffects" ~ "FE",
      method == "KMeans" | method == "K-Means" ~ "K-Means",
      TRUE ~ method
    )
  )

# Define method order by category
method_order <- c("OLS", "OLS+HC1", "FE", "Logit", "ARIMA", "MSTL", "K-Means", "PCA")

# Combine data
r_filtered <- r_data %>%
  filter(method_std %in% method_order) %>%
  select(implementation, method = method_std, n,
         time_min_ms, time_p25_ms, time_median_ms, time_p75_ms, time_max_ms,
         time_mean_us, time_std_us)

rust_filtered <- rust_data %>%
  filter(method_std %in% method_order) %>%
  select(implementation, method = method_std, n,
         time_min_ms, time_p25_ms, time_median_ms, time_p75_ms, time_max_ms,
         time_mean_us, time_std_us)

plot_data <- bind_rows(r_filtered, rust_filtered)

# Print summary
cat("Benchmarks at N=10,000:\n")
comparison <- plot_data %>%
  select(implementation, method, time_median_ms) %>%
  spread(implementation, time_median_ms) %>%
  mutate(speedup = round(R / `p2a (Rust)`, 1))
print(comparison)

# Set factor levels
plot_data$method <- factor(plot_data$method, levels = method_order)
plot_data$implementation <- factor(plot_data$implementation, levels = c("R", "p2a (Rust)"))

# Define colorblind-friendly colors (Okabe-Ito palette)
colors <- c("R" = "#0072B2", "p2a (Rust)" = "#E69F00")

# Use median with IQR (p25-p75) - more robust to outliers than mean/SD
# Create the plot - dots for median with IQR error bars
p <- ggplot(plot_data, aes(x = method, y = time_median_ms, color = implementation)) +
  geom_errorbar(
    aes(ymin = time_p25_ms, ymax = time_p75_ms),
    position = position_dodge(width = 0.6),
    width = 0.3,
    linewidth = 1.2
  ) +
  geom_point(
    position = position_dodge(width = 0.6),
    size = 4
  ) +
  scale_y_log10(name = "Execution Time\n(milliseconds, log scale)",
                labels = scales::comma) +
  coord_cartesian(ylim = c(0.3, 110)) +
  scale_x_discrete(name = "Method") +
  scale_color_manual(values = colors, name = NULL) +
  theme_bw(base_size = 16) +
  theme(
    legend.position = c(0.98, 0.98),
    legend.justification = c("right", "top"),
    legend.background = element_rect(fill = "white", color = "grey40", linewidth = 0.5),
    legend.text = element_text(size = 15),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 15),
    axis.text.y = element_text(size = 15),
    axis.title = element_text(size = 17),
    panel.grid.minor = element_blank(),
    plot.margin = margin(10, 15, 10, 10)
  )

# Save the figure
cat("\nSaving figure to:", output_file, "\n")
ggsave(output_file, p, width = 10, height = 5, dpi = 300)

png_file <- sub("\\.pdf$", ".png", output_file)
ggsave(png_file, p, width = 10, height = 5, dpi = 300)
cat("Also saved as:", png_file, "\n")

cat("\nDone!\n")
