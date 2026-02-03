# kmeans.R - K-Means clustering using stats::kmeans

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = 3,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Set seed for reproducibility
  set.seed(seed)

  # Select columns for clustering
  if (!is.null(indep_vars)) {
    cluster_data <- data[, indep_vars, drop = FALSE]
  } else {
    # Use all numeric columns
    cluster_data <- data[, sapply(data, is.numeric), drop = FALSE]
  }

  # Convert to matrix
  X <- as.matrix(cluster_data)

  # Fit k-means
  model <- kmeans(X, centers = k, nstart = 25, iter.max = 100)

  # Extract results
  centers <- model$centers
  rownames(centers) <- paste0("cluster_", 1:nrow(centers))

  list(
    centers = lapply(1:nrow(centers), function(i) setNames(as.list(centers[i, ]), colnames(centers))),
    cluster_sizes = as.list(model$size),
    within_ss = as.list(model$withinss),
    total_within_ss = model$tot.withinss,
    between_ss = model$betweenss,
    total_ss = model$totss,
    k = k,
    n_obs = nrow(X),
    n_vars = ncol(X),
    iterations = model$iter
  )
}
