# ============================================================================
# fig_accuracy_by_category.R
# Heatmap showing accuracy by category for each model
# ============================================================================

## SETUP ----
library(tidyverse)
library(viridis)

OUTPUT <- "../figures/"

## DATA (from tab_category_accuracy.tex) ----
categories <- c("Regression", "Panel", "Causal", "Discrete",
                 "Time Series", "Hypothesis", "ML", "Visualization")

models <- c("Claude 3.5 Haiku", "Qwen 2.5 72B", "Llama 3.3 70B",
            "GPT-5 Nano", "GPT-4o Mini", "GPT-4.1 Nano", "Ministral 3B")

# Rows: models (in order above), Cols: categories (in order above)
acc_matrix <- matrix(c(
  100, 100, 100, 100, 100, 100, 100, 100,  # Haiku
  100, 100, 100, 100, 100, 100, 100, 100,  # Qwen 2.5 72B
  100,  90, 100, 100, 100, 100, 100, 100,  # Llama 3.3 70B
  100, 100, 100, 100, 100,  92,  92, 100,  # GPT-5 Nano
  100, 100, 100,  90, 100,  92, 100, 100,  # GPT-4o Mini
   90, 100, 100,  80, 100, 100, 100, 100,  # GPT-4.1 Nano
   90, 100,  91, 100,  67,  83, 100, 100   # Ministral 3B
), nrow = 7, ncol = 8, byrow = TRUE)

category_accuracy <- expand.grid(
  model = models,
  category = categories,
  stringsAsFactors = FALSE
) %>%
  mutate(
    accuracy = as.vector(acc_matrix),
    model = factor(model, levels = rev(models)),
    category = factor(category, levels = categories)
  )

## PLOT ----
p <- ggplot(category_accuracy, aes(x = category, y = model, fill = accuracy)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f", accuracy)),
            size = 5.5, color = ifelse(category_accuracy$accuracy < 70, "white", "black")) +
  scale_fill_viridis(
    option = "D",
    limits = c(0, 100),
    breaks = c(0, 25, 50, 75, 100),
    labels = function(x) paste0(x, "%"),
    name = "Accuracy"
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = NULL
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14),
    axis.text.y = element_text(size = 14),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.text = element_text(size = 13),
    legend.title = element_text(size = 14),
    plot.margin = margin(10, 10, 10, 10)
  )

## WRITE TO DISK ----
ggsave(
  paste0(OUTPUT, "fig_accuracy_by_category.pdf"),
  plot = p,
  width = 8,
  height = 5,
  device = cairo_pdf
)

message("Created: ", OUTPUT, "fig_accuracy_by_category.pdf")
