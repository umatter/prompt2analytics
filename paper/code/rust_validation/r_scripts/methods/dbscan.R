# dbscan.R - DBSCAN clustering using dbscan package

suppressPackageStartupMessages({
  library(dbscan)
})

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       eps = 0.5, minPts = 5) {

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

  # Fit DBSCAN
  model <- dbscan(X, eps = eps, minPts = minPts)

  # Count clusters (excluding noise which is labeled 0)
  cluster_labels <- model$cluster
  n_clusters <- length(unique(cluster_labels[cluster_labels > 0]))
  n_noise <- sum(cluster_labels == 0)

  # Cluster sizes (excluding noise)
  cluster_sizes <- as.list(table(cluster_labels[cluster_labels > 0]))

  list(
    cluster_labels = as.list(cluster_labels),
    n_clusters = n_clusters,
    n_noise = n_noise,
    cluster_sizes = cluster_sizes,
    eps = eps,
    minPts = minPts,
    n_obs = nrow(X),
    n_vars = ncol(X)
  )
}
