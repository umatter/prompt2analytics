# sort.R - Sort dataset by columns
# R reference: dplyr::arrange or base::order

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       sort_col = NULL) {
  # Sort by sort_col, first column in indep_vars, dep_var, or first numeric column
  if (is.null(sort_col)) {
    sort_col <- if (!is.null(dep_var) && dep_var %in% names(data)) {
      dep_var
    } else if (!is.null(indep_vars) && length(indep_vars) > 0) {
      indep_vars[1]
    } else {
      names(data)[sapply(data, is.numeric)][1]
    }
  }

  if (is.null(sort_col) || !sort_col %in% names(data)) {
    stop("No valid sort column found")
  }

  # Sort using base R (faster than dplyr for benchmarking)
  result <- data[order(data[[sort_col]]), ]

  list(
    n_rows = nrow(result),
    n_cols = ncol(result),
    sort_column = sort_col
  )
}
