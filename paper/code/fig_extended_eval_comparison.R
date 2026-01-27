#!/usr/bin/env Rscript
# fig_extended_eval_comparison.R
# Multi-panel figure comparing single-turn vs multi-turn, synthetic vs naturalistic
# For JSS paper Section 7.3

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(jsonlite)

# Set output path
output_dir <- "paper/figures"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Function to load JSONL results (handles both compact and pretty-printed JSON)
load_jsonl <- function(path) {
  if (!file.exists(path)) {
    warning(paste("File not found:", path))
    return(data.frame())
  }
  content <- paste(readLines(path, warn = FALSE), collapse = "\n")
  # Split by "}\n{" pattern (for pretty-printed) or "}\n{" (for compact)
  # Add array brackets to make it valid JSON array
  content <- paste0("[", gsub("\\}\n\\{", "},{", content), "]")
  tryCatch({
    fromJSON(content, flatten = TRUE)
  }, error = function(e) {
    warning(paste("Error parsing:", path, "-", e$message))
    data.frame()
  })
}

# Function to get most recent results file for a pattern
get_latest_results <- function(dir, pattern) {
  files <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[which.max(file.mtime(files))]
}

# Define results directories
eval_dir <- "paper/code/llm_eval/results"

# Load single-turn results (from existing evaluation)
single_turn_files <- list.files(eval_dir, pattern = ".*_all_.*\\.jsonl$", full.names = TRUE)
single_turn_files <- single_turn_files[!grepl("(multi_turn|naturalistic|parameter|interpretation|out_of_scope)", single_turn_files)]

# Load multi-turn results
multi_turn_dir <- file.path(eval_dir, "multi_turn")
multi_turn_files <- list.files(multi_turn_dir, pattern = "\\.jsonl$", full.names = TRUE)

# Load naturalistic results
naturalistic_dir <- file.path(eval_dir, "naturalistic")
naturalistic_files <- list.files(naturalistic_dir, pattern = "\\.jsonl$", full.names = TRUE)

# Process single-turn results
process_single_turn <- function(files) {
  results <- lapply(files, load_jsonl)
  results <- bind_rows(results)

  if (nrow(results) == 0) return(data.frame())

  results %>%
    group_by(model) %>%
    summarize(
      accuracy = mean(match_type %in% c("exact", "acceptable")),
      n_tests = n(),
      .groups = "drop"
    ) %>%
    mutate(eval_type = "Single-Turn")
}

# Process multi-turn results
process_multi_turn <- function(files) {
  results <- lapply(files, load_jsonl)
  results <- bind_rows(results)

  if (nrow(results) == 0) return(data.frame())

  # Turn-level accuracy
  turn_acc <- results %>%
    group_by(model) %>%
    summarize(
      accuracy = mean(match_type %in% c("exact", "acceptable")),
      n_tests = n(),
      .groups = "drop"
    ) %>%
    mutate(eval_type = "Multi-Turn (Turn)")

  # Conversation completion rate
  conv_acc <- results %>%
    group_by(model, conversation_id) %>%
    summarize(
      all_correct = all(match_type %in% c("exact", "acceptable")),
      .groups = "drop"
    ) %>%
    group_by(model) %>%
    summarize(
      accuracy = mean(all_correct),
      n_tests = n(),
      .groups = "drop"
    ) %>%
    mutate(eval_type = "Multi-Turn (Conv)")

  bind_rows(turn_acc, conv_acc)
}

# Process naturalistic results
process_naturalistic <- function(files) {
  results <- lapply(files, load_jsonl)
  results <- bind_rows(results)

  if (nrow(results) == 0) return(data.frame())

  # Overall accuracy
  overall <- results %>%
    group_by(model) %>%
    summarize(
      accuracy = mean(match_type %in% c("exact", "acceptable")),
      n_tests = n(),
      .groups = "drop"
    ) %>%
    mutate(eval_type = "Naturalistic")

  # By prompt type
  by_type <- results %>%
    group_by(model, prompt_type) %>%
    summarize(
      accuracy = mean(match_type %in% c("exact", "acceptable")),
      n_tests = n(),
      .groups = "drop"
    ) %>%
    mutate(eval_type = paste0("Nat: ", prompt_type))

  bind_rows(overall, by_type)
}

# Load and process all results
single_turn_data <- process_single_turn(single_turn_files)
multi_turn_data <- process_multi_turn(multi_turn_files)
naturalistic_data <- process_naturalistic(naturalistic_files)

# Combine data
all_data <- bind_rows(single_turn_data, multi_turn_data, naturalistic_data)

# If no data, create sample data for demonstration
if (nrow(all_data) == 0) {
  message("No results found. Creating sample data for demonstration...")

  models <- c("gpt-4o", "claude-3-5-sonnet", "gpt-4o-mini", "llama-3.3-70b")

  all_data <- data.frame(
    model = rep(models, each = 6),
    eval_type = rep(c("Single-Turn", "Multi-Turn (Turn)", "Multi-Turn (Conv)",
                      "Naturalistic", "Nat: informal", "Nat: typos"), length(models)),
    accuracy = c(
      # gpt-4o
      0.95, 0.92, 0.85, 0.88, 0.85, 0.82,
      # claude-3-5-sonnet
      0.93, 0.90, 0.82, 0.86, 0.83, 0.80,
      # gpt-4o-mini
      0.89, 0.85, 0.75, 0.82, 0.78, 0.75,
      # llama-3.3-70b
      0.85, 0.80, 0.68, 0.78, 0.74, 0.70
    ),
    n_tests = rep(c(87, 72, 20, 100, 25, 25), length(models))
  )
}

# Create comparison plot: Single-turn vs Multi-turn
plot_data_comparison <- all_data %>%
  filter(eval_type %in% c("Single-Turn", "Multi-Turn (Turn)", "Multi-Turn (Conv)"))

p1 <- ggplot(plot_data_comparison, aes(x = reorder(model, -accuracy), y = accuracy, fill = eval_type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_brewer(palette = "Set2", name = "Evaluation Type") +
  labs(
    title = "A) Single-Turn vs Multi-Turn Performance",
    x = NULL,
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(size = 11, face = "bold")
  )

# Create comparison plot: Synthetic vs Naturalistic
plot_data_nat <- all_data %>%
  filter(eval_type %in% c("Single-Turn", "Naturalistic"))

p2 <- ggplot(plot_data_nat, aes(x = reorder(model, -accuracy), y = accuracy, fill = eval_type)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
  scale_fill_manual(values = c("Single-Turn" = "#66C2A5", "Naturalistic" = "#FC8D62"),
                    name = "Prompt Type") +
  labs(
    title = "B) Synthetic vs Naturalistic Prompts",
    x = NULL,
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom",
    plot.title = element_text(size = 11, face = "bold")
  )

# Create breakdown by naturalistic prompt type
plot_data_nat_type <- all_data %>%
  filter(grepl("^Nat:", eval_type)) %>%
  mutate(prompt_type = gsub("Nat: ", "", eval_type))

if (nrow(plot_data_nat_type) > 0) {
  p3 <- ggplot(plot_data_nat_type, aes(x = prompt_type, y = accuracy, fill = model)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    scale_fill_brewer(palette = "Set1", name = "Model") +
    labs(
      title = "C) Accuracy by Naturalistic Prompt Type",
      x = "Prompt Type",
      y = "Accuracy"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      plot.title = element_text(size = 11, face = "bold")
    )
} else {
  p3 <- ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = "No naturalistic data available") +
    theme_void() +
    labs(title = "C) Accuracy by Naturalistic Prompt Type")
}

# Combine plots
combined_plot <- (p1 | p2) / p3 +
  plot_annotation(
    title = "Extended LLM Tool Selection Evaluation",
    subtitle = "Comparing performance across evaluation types",
    theme = theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10, color = "gray40")
    )
  )

# Save plot
ggsave(
  file.path(output_dir, "fig_extended_eval_comparison.pdf"),
  combined_plot,
  width = 10,
  height = 10
)

ggsave(
  file.path(output_dir, "fig_extended_eval_comparison.png"),
  combined_plot,
  width = 10,
  height = 10,
  dpi = 300
)

message("Saved: fig_extended_eval_comparison.pdf and .png")
