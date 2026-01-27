# group_by.R - Group by and aggregate
# R reference: dplyr::group_by + summarize or stats::aggregate

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {
  # Use cluster_var as group column if provided, otherwise use entity_var or first column
  group_col <- if (!is.null(cluster_var)) {
    cluster_var
  } else if (!is.null(entity_var)) {
    entity_var
  } else if (!is.null(indep_vars) && length(indep_vars) > 0) {
    indep_vars[1]
  } else {
    # Find a categorical/integer column suitable for grouping
    cat_cols <- names(data)[sapply(data, function(x) is.factor(x) || is.character(x) || is.integer(x))]
    if (length(cat_cols) > 0) cat_cols[1] else names(data)[1]
  }

  # Find numeric columns to aggregate (use dep_var if provided)
  agg_col <- if (!is.null(dep_var)) {
    dep_var
  } else {
    num_cols <- names(data)[sapply(data, is.numeric)]
    num_cols <- setdiff(num_cols, group_col)
    if (length(num_cols) == 0) stop("No numeric columns to aggregate")
    num_cols[1]
  }

  # Use aggregate for benchmarking (base R, no dependencies)
  result <- aggregate(data[[agg_col]], by = list(group = data[[group_col]]), FUN = mean)
  names(result) <- c(group_col, paste0(agg_col, "_mean"))

  list(
    n_groups = nrow(result),
    group_column = group_col,
    agg_column = agg_col
  )
}
