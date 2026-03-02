# fisher_exact.R - Fisher's exact test using stats::fisher.test

run_method <- function(data, dep_var, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       alternative = "two.sided") {

  # Build contingency table from two columns
  if (!is.null(indep_vars) && length(indep_vars) >= 1) {
    x <- data[[dep_var]]
    y <- data[[indep_vars[1]]]
    tbl <- table(x, y)
  } else {
    # Assume dep_var points to a pre-formed matrix or table
    # Use first two numeric columns as a 2x2 table
    num_cols <- names(data)[sapply(data, is.numeric)]
    if (length(num_cols) >= 2) {
      tbl <- matrix(c(data[[num_cols[1]]], data[[num_cols[2]]]),
                    nrow = 2, byrow = FALSE)
    } else {
      stop("Need at least two columns for a contingency table")
    }
  }

  # Run Fisher's exact test (use simulate for large tables to avoid workspace overflow)
  result <- tryCatch(
    fisher.test(tbl, alternative = alternative),
    error = function(e) {
      if (grepl("workspace|FEXACT", e$message)) {
        fisher.test(tbl, alternative = alternative, simulate.p.value = TRUE, B = 10000)
      } else {
        stop(e)
      }
    }
  )

  # Compute sample odds ratio for 2x2 tables
  sample_or <- NA
  if (nrow(tbl) == 2 && ncol(tbl) == 2) {
    if (tbl[1, 2] * tbl[2, 1] != 0) {
      sample_or <- (tbl[1, 1] * tbl[2, 2]) / (tbl[1, 2] * tbl[2, 1])
    } else {
      sample_or <- Inf
    }
  }

  res <- list(
    p_value = result$p.value,
    alternative = alternative,
    n_obs = sum(tbl)
  )

  # Odds ratio and CI (only for 2x2 tables)
  if (!is.null(result$estimate)) {
    res$odds_ratio_cml = as.numeric(result$estimate)
    res$sample_odds_ratio = sample_or
  }
  if (!is.null(result$conf.int)) {
    res$conf_int_lower = result$conf.int[1]
    res$conf_int_upper = result$conf.int[2]
  }

  res
}
