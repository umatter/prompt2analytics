#!/usr/bin/env Rscript
# generate_benchmark_figure.R
#
# Generates a two-panel box plot figure comparing R and Rust benchmark times
# Panel A: R execution times
# Panel B: p2a (Rust) execution times
#
# Usage: Rscript generate_benchmark_figure.R
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
output_file_main <- file.path(paper_dir, "figures/benchmark_boxplots.pdf")
output_file_appendix <- file.path(paper_dir, "figures/benchmark_boxplots_appendix.pdf")

cat("=== Generating Benchmark Box Plot Figure ===\n\n")

# Find the most recent benchmark files
find_latest_file <- function(dir, pattern) {
  files <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[which.max(file.info(files)$mtime)]
}

r_file <- find_latest_file(r_results_dir, "r_comprehensive_.*\\.csv$")
rust_file <- find_latest_file(rust_results_dir, "rust_comprehensive_.*\\.json$")

if (is.null(r_file)) stop("No R benchmark file found")
if (is.null(rust_file)) stop("No Rust benchmark file found")

cat("R benchmark file:", r_file, "\n")
cat("Rust benchmark file:", rust_file, "\n\n")

# Read R benchmarks
r_data <- read.csv(r_file, stringsAsFactors = FALSE) %>%
  mutate(
    implementation = "R",
    # Convert microseconds to milliseconds
    time_min_ms = time_min_us / 1000,
    time_p25_ms = time_p25_us / 1000,
    time_median_ms = time_median_us / 1000,
    time_p75_ms = time_p75_us / 1000,
    time_max_ms = time_max_us / 1000
  ) %>%
  # Create clean labels
  mutate(
    category = case_when(
      grepl("^OLS", method) ~ "Regression",
      grepl("^FE", method) ~ "Panel",
      grepl("Logit|Probit", method) ~ "Discrete",
      grepl("K-Means|PCA", method) ~ "ML",
      grepl("ARIMA|MSTL", method) ~ "TimeSeries",
      TRUE ~ "Other"
    ),
    label = paste0(method, "\n(n=", n, ")")
  )

# Read Rust benchmarks
rust_json <- fromJSON(rust_file)

# Handle variant column which may not exist
if (!"variant" %in% names(rust_json)) {
  rust_json$variant <- NA_character_
}

rust_data <- rust_json %>%
  mutate(
    implementation = "p2a (Rust)",
    # Convert microseconds to milliseconds
    time_min_ms = time_min_us / 1000,
    time_p25_ms = time_p25_us / 1000,
    time_median_ms = time_median_us / 1000,
    time_p75_ms = time_p75_us / 1000,
    time_max_ms = time_max_us / 1000,
    # Standardize method names to match R
    method_std = case_when(
      method == "OLS" & (is.na(variant) | variant == "standard") ~ "OLS",
      method == "OLS" & variant == "HC1" ~ "OLS+HC1",
      method == "Panel_FE" ~ "FE_plm",
      method == "FixedEffects" ~ "FE_plm",
      method == "KMeans" ~ "K-Means",
      TRUE ~ method
    )
  ) %>%
  mutate(
    category = case_when(
      grepl("^OLS", method_std) ~ "Regression",
      grepl("^FE|Panel", method_std) ~ "Panel",
      grepl("Logit|Probit", method_std) ~ "Discrete",
      grepl("K-Means|KMeans|PCA", method_std) ~ "ML",
      grepl("ARIMA|MSTL", method_std) ~ "TimeSeries",
      TRUE ~ "Other"
    ),
    label = paste0(method_std, "\n(n=", n, ")")
  )

# Filter to common benchmarks
# Use method names that exist in both R and Rust benchmarks
selected_methods <- c(
  "OLS", "OLS+HC1", "FE_plm", "Logit", "PCA", "K-Means", "ARIMA", "MSTL"
)

# Get all available (method, n) combinations from R
r_available <- r_data %>%
  filter(method %in% selected_methods) %>%
  select(method, n) %>%
  distinct()

# Get all available (method, n) combinations from Rust
rust_available <- rust_data %>%
  filter(method_std %in% selected_methods) %>%
  select(method_std, n) %>%
  distinct() %>%
  rename(method = method_std)

# Find common (method, n) combinations
common_combos <- inner_join(r_available, rust_available, by = c("method", "n"))

cat("\nCommon benchmark combinations found:\n")
print(common_combos)

# For R data - filter to common combinations
r_filtered <- r_data %>%
  filter(method %in% selected_methods) %>%
  inner_join(common_combos, by = c("method", "n")) %>%
  select(implementation, category, method, n, label,
         time_min_ms, time_p25_ms, time_median_ms, time_p75_ms, time_max_ms)

# For Rust data - filter to common combinations
rust_filtered <- rust_data %>%
  filter(method_std %in% selected_methods) %>%
  inner_join(common_combos, by = c("method_std" = "method", "n" = "n")) %>%
  mutate(method = method_std) %>%
  select(implementation, category, method, n, label,
         time_min_ms, time_p25_ms, time_median_ms, time_p75_ms, time_max_ms)

# Combine data
plot_data <- bind_rows(r_filtered, rust_filtered)

# Print summary
cat("All benchmarks:\n")
print(plot_data %>% select(implementation, method, n, time_median_ms) %>%
        spread(implementation, time_median_ms))

# Split into main figure (largest N per method) and appendix (other N values)
largest_n_per_method <- plot_data %>%
  group_by(method) %>%
  summarise(max_n = max(n), .groups = "drop")

plot_data_main <- plot_data %>%
  inner_join(largest_n_per_method, by = "method") %>%
  filter(n == max_n) %>%
  select(-max_n)

plot_data_appendix <- plot_data %>%
  inner_join(largest_n_per_method, by = "method") %>%
  filter(n != max_n) %>%
  select(-max_n)

cat("\n=== Main figure (largest N per method) ===\n")
print(plot_data_main %>%
        filter(implementation == "R") %>%
        select(method, n) %>%
        distinct())

cat("\n=== Appendix figure (other N values) ===\n")
print(plot_data_appendix %>%
        filter(implementation == "R") %>%
        select(method, n) %>%
        distinct() %>%
        arrange(method, n))

# Helper function to create box plot
create_boxplot <- function(data, title_suffix = "") {
  # Create simpler labels without n for main figure
  data <- data %>%
    mutate(label_simple = method)

  # Order methods by category
  method_order <- data %>%
    arrange(category, method) %>%
    pull(label_simple) %>%
    unique()

  data$label_simple <- factor(data$label_simple, levels = method_order)
  data$implementation <- factor(data$implementation, levels = c("R", "p2a (Rust)"))

  ggplot(data, aes(x = label_simple, fill = implementation)) +
    geom_boxplot(
      aes(
        ymin = time_min_ms,
        lower = time_p25_ms,
        middle = time_median_ms,
        upper = time_p75_ms,
        ymax = time_max_ms
      ),
      stat = "identity",
      width = 0.6
    ) +
    facet_wrap(~ implementation, ncol = 2, scales = "free_y",
               labeller = labeller(implementation = c("R" = "A: R Reference Implementations",
                                                      "p2a (Rust)" = "B: prompt2analytics (Rust)"))) +
    scale_y_log10(name = "Execution Time (milliseconds, log scale)",
                  labels = scales::comma) +
    scale_x_discrete(name = "Method") +
    scale_fill_manual(values = c("R" = "#4292C6", "p2a (Rust)" = "#EF6548")) +
    theme_bw(base_size = 12) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
      axis.text.y = element_text(size = 11),
      axis.title = element_text(size = 12),
      strip.text = element_text(size = 12, face = "bold"),
      strip.background = element_rect(fill = "gray95"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 15, 10, 10)
    )
}

# Helper function for appendix with n in labels
create_boxplot_appendix <- function(data) {
  # Order methods by category, then method, then n
  method_order <- data %>%
    arrange(category, method, n) %>%
    pull(label) %>%
    unique()

  data$label <- factor(data$label, levels = method_order)
  data$implementation <- factor(data$implementation, levels = c("R", "p2a (Rust)"))

  ggplot(data, aes(x = label, fill = implementation)) +
    geom_boxplot(
      aes(
        ymin = time_min_ms,
        lower = time_p25_ms,
        middle = time_median_ms,
        upper = time_p75_ms,
        ymax = time_max_ms
      ),
      stat = "identity",
      width = 0.6
    ) +
    facet_wrap(~ implementation, ncol = 2, scales = "free_y",
               labeller = labeller(implementation = c("R" = "A: R Reference Implementations",
                                                      "p2a (Rust)" = "B: prompt2analytics (Rust)"))) +
    scale_y_log10(name = "Execution Time (milliseconds, log scale)",
                  labels = scales::comma) +
    scale_x_discrete(name = "Method (sample size)") +
    scale_fill_manual(values = c("R" = "#4292C6", "p2a (Rust)" = "#EF6548")) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
      axis.text.y = element_text(size = 10),
      axis.title = element_text(size = 11),
      strip.text = element_text(size = 11, face = "bold"),
      strip.background = element_rect(fill = "gray95"),
      panel.grid.minor = element_blank(),
      plot.margin = margin(10, 15, 10, 10)
    )
}

# Create and save main figure (largest N only)
p_main <- create_boxplot(plot_data_main)

cat("\nSaving main figure to:", output_file_main, "\n")
ggsave(output_file_main, p_main, width = 10, height = 5, dpi = 300)

png_file_main <- sub("\\.pdf$", ".png", output_file_main)
ggsave(png_file_main, p_main, width = 10, height = 5, dpi = 300)
cat("Also saved as:", png_file_main, "\n")

# Create and save appendix figure (other N values)
if (nrow(plot_data_appendix) > 0) {
  p_appendix <- create_boxplot_appendix(plot_data_appendix)

  cat("\nSaving appendix figure to:", output_file_appendix, "\n")
  ggsave(output_file_appendix, p_appendix, width = 12, height = 5, dpi = 300)

  png_file_appendix <- sub("\\.pdf$", ".png", output_file_appendix)
  ggsave(png_file_appendix, p_appendix, width = 12, height = 5, dpi = 300)
  cat("Also saved as:", png_file_appendix, "\n")
} else {
  cat("\nNo data for appendix figure.\n")
}

cat("\nDone!\n")
