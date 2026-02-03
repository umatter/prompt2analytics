# filter.R - Filter rows based on condition
# R reference: dplyr::filter or base R subset

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {
  # Filter first numeric column > median
  num_cols <- names(data)[sapply(data, is.numeric)]
  if (length(num_cols) == 0) {
    stop("No numeric columns to filter on")
  }

  filter_col <- if (!is.null(indep_vars) && length(indep_vars) > 0) {
    indep_vars[1]
  } else {
    num_cols[1]
  }

  threshold <- median(data[[filter_col]], na.rm = TRUE)
  result <- data[data[[filter_col]] > threshold, ]

  list(
    n_rows_before = nrow(data),
    n_rows_after = nrow(result),
    filter_column = filter_col,
    threshold = threshold
  )
}
