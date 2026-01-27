# standardize.R - Standardize columns (Z-score normalization)
# R reference: base::scale

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Select columns to standardize
  if (!is.null(indep_vars) && length(indep_vars) > 0) {
    cols_to_scale <- indep_vars
  } else {
    # Default: all numeric columns
    cols_to_scale <- names(data)[sapply(data, is.numeric)]
  }

  # Verify columns exist and are numeric
  valid_cols <- cols_to_scale[cols_to_scale %in% names(data)]
  valid_cols <- valid_cols[sapply(data[, valid_cols, drop = FALSE], is.numeric)]

  if (length(valid_cols) == 0) {
    stop("No valid numeric columns to standardize")
  }

  # Standardize using base R scale()
  result <- data
  scaled_data <- scale(data[, valid_cols, drop = FALSE])
  result[, valid_cols] <- as.data.frame(scaled_data)

  # Extract means and standard deviations used for scaling
  means <- attr(scaled_data, "scaled:center")
  sds <- attr(scaled_data, "scaled:scale")

  list(
    n_rows = nrow(result),
    n_cols = ncol(result),
    standardized_columns = valid_cols,
    means = setNames(as.list(means), valid_cols),
    std_devs = setNames(as.list(sds), valid_cols)
  )
}
