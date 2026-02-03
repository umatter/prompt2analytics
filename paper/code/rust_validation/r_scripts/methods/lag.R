# lag.R - Create lagged column
# R reference: dplyr::lag or stats::lag

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = 1,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Select column to lag (use first indep_var or first numeric column)
  if (!is.null(indep_vars) && length(indep_vars) > 0) {
    lag_col <- indep_vars[1]
  } else {
    lag_col <- names(data)[sapply(data, is.numeric)][1]
  }

  if (is.null(lag_col) || !lag_col %in% names(data)) {
    stop("No valid column to lag")
  }

  # Use k as the lag amount (default 1)
  lag_amount <- if (is.null(k)) 1 else k

  # Create lagged column using dplyr-style lag (shift down, NA at start)
  original <- data[[lag_col]]
  lagged <- c(rep(NA, lag_amount), original[1:(length(original) - lag_amount)])

  result <- data
  result[[paste0(lag_col, "_lag", lag_amount)]] <- lagged

  list(
    n_rows = nrow(result),
    n_cols = ncol(result),
    lagged_column = lag_col,
    lag_amount = lag_amount,
    n_na = sum(is.na(lagged))
  )
}
