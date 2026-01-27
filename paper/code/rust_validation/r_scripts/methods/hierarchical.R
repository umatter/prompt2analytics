# hierarchical.R - Hierarchical clustering using stats::hclust

run_method <- function(data, dep_var = NULL, indep_vars = NULL, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = 3,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42,
                       linkage = "ward.D2") {

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

  # Compute distance matrix
  dist_matrix <- dist(X)

  # Fit hierarchical clustering
  model <- hclust(dist_matrix, method = linkage)

  # Cut tree to get k clusters
  cluster_labels <- cutree(model, k = k)

  # Cluster sizes
  cluster_sizes <- as.list(table(cluster_labels))

  list(
    cluster_labels = as.list(cluster_labels),
    n_clusters = k,
    cluster_sizes = cluster_sizes,
    height = as.list(model$height),
    merge = model$merge,
    order = as.list(model$order),
    linkage = linkage,
    n_obs = nrow(X),
    n_vars = ncol(X)
  )
}
