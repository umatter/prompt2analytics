# select.R - Select columns from dataset
# R reference: dplyr::select or base::subset

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Select columns specified in indep_vars
  if (!is.null(indep_vars) && length(indep_vars) > 0) {
    select_cols <- indep_vars
  } else {
    # Default: select first 2 columns
    select_cols <- names(data)[1:min(2, ncol(data))]
  }

  # Verify columns exist
  valid_cols <- select_cols[select_cols %in% names(data)]
  if (length(valid_cols) == 0) {
    stop("No valid columns to select")
  }

  # Select using base R
  result <- data[, valid_cols, drop = FALSE]

  list(
    n_rows = nrow(result),
    n_cols = ncol(result),
    selected_columns = valid_cols
  )
}
