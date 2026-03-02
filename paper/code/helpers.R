# helpers.R
# Shared utility functions for paper exhibit generation
#
# Usage: source("helpers.R")

library(dplyr)
library(stringr)

#' Categorize benchmark methods into groups
#'
#' @param method Character vector of method names (e.g., "ols", "panel_fe", "kmeans")
#' @return Character vector of category labels
categorize_method <- function(method) {
  case_when(
    grepl("^ols|^quantreg|^gls$|^nls$|^loess$|^diagnostics$", method, ignore.case = TRUE) ~ "Regression",
    grepl("^panel|^hausman$|^hdfe$|^feglm$", method, ignore.case = TRUE) ~ "Panel",
    grepl("logit|probit|negbin", method, ignore.case = TRUE) ~ "Discrete",
    grepl("kmeans|pca|dbscan|hierarchical", method, ignore.case = TRUE) ~ "ML",
    grepl("arima|mstl|stl|holt|^ar$|var_model|granger|changepoint|acf", method, ignore.case = TRUE) ~ "Time Series",
    grepl("anova|ttest|shapiro|chisq|fisher|wilcoxon|kruskal", method, ignore.case = TRUE) ~ "Statistics",
    grepl("iv2sls|did$|ipw|doubly", method, ignore.case = TRUE) ~ "Causal",
    grepl("kaplan|cox_ph", method, ignore.case = TRUE) ~ "Survival",
    grepl("sort|filter|group|select|standardize|lag|lead|diff", method, ignore.case = TRUE) ~ "Munging",
    TRUE ~ "Other"
  )
}

#' Clean method names for display
#'
#' @param method Character vector of method names
#' @return Character vector of cleaned labels
clean_method_label <- function(method) {
  method %>%
    str_replace_all("_", " ") %>%
    str_replace_all("hc([0-3])", "HC\\1") %>%
    str_replace("^ols$", "OLS") %>%
    str_replace("^ols ", "OLS+") %>%
    str_replace("panel fe", "Panel FE") %>%
    str_replace("panel re", "Panel RE") %>%
    str_replace("panel hdfe", "Panel HDFE") %>%
    str_to_title() %>%
    # Fix abbreviations that str_to_title mangles
    str_replace("Iv2sls", "IV/2SLS") %>%
    str_replace("Pca", "PCA") %>%
    str_replace("Dbscan", "DBSCAN") %>%
    str_replace("Arima", "ARIMA") %>%
    str_replace("Acf", "ACF") %>%
    str_replace("Mstl", "MSTL") %>%
    str_replace("Stl", "STL") %>%
    str_replace("Nls", "NLS") %>%
    str_replace("Gls", "GLS") %>%
    str_replace("Var ", "VAR ") %>%
    str_replace("Negbin", "NegBin") %>%
    str_replace("Hdfe", "HDFE") %>%
    str_replace("Chisq", "Chi-sq")
}

#' Load and flatten benchmark summary JSON
#'
#' @param path Path to benchmark_summary.json
#' @return Tibble with method, n, r_median_us, rust_median_us, speedup columns
load_benchmark_summary <- function(path) {
  if (!file.exists(path)) {
    stop("Benchmark summary not found: ", path)
  }

  summary_data <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  methods_list <- summary_data$methods

  purrr::map_dfr(names(methods_list), function(method) {
    method_results <- methods_list[[method]]
    purrr::map_dfr(method_results, function(r) {
      tibble(
        method = method,
        n = r$n,
        r_median_us = r$r_median_us,
        rust_median_us = r$rust_median_us,
        speedup = r$speedup
      )
    })
  })
}

#' Standard ggplot2 theme for paper figures
#'
#' @param base_size Base font size (default: 11)
#' @return ggplot2 theme object
theme_paper <- function(base_size = 11) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      legend.position = "bottom",
      legend.title = ggplot2::element_text(size = base_size - 1),
      panel.grid.minor = ggplot2::element_blank(),
      plot.margin = ggplot2::margin(10, 20, 10, 10)
    )
}

#' Format speedup value for display
#'
#' @param x Numeric speedup value
#' @return Character string like "2.5x"
format_speedup <- function(x) {
  sprintf("%.1fx", x)
}

#' Standard output paths
OUTPUT_FIGURES <- "../figures/"
OUTPUT_TABLES <- "../tables/"
INPUT_BENCHMARK <- "rust_validation/results/summaries/"
