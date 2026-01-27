# pca.R - Principal Component Analysis using stats::prcomp

run_method <- function(data, dep_var, indep_vars, entity_var = NULL, time_var = NULL,
                       cluster_var = NULL, instrument_vars = NULL, k = NULL,
                       n_components = NULL, arima_order = NULL, robust = NULL, seed = 42) {

  # Select columns for PCA
  if (!is.null(indep_vars)) {
    pca_data <- data[, indep_vars, drop = FALSE]
  } else {
    # Use all numeric columns
    pca_data <- data[, sapply(data, is.numeric), drop = FALSE]
  }

  # Convert to matrix
  X <- as.matrix(pca_data)

  # Fit PCA (centered and scaled)
  model <- prcomp(X, center = TRUE, scale. = TRUE)

  # Extract results
  n_comp <- if (!is.null(n_components)) min(n_components, ncol(X)) else ncol(X)

  # Variance explained
  var_explained <- model$sdev^2
  prop_var <- var_explained / sum(var_explained)
  cumulative_var <- cumsum(prop_var)

  # Loadings (rotation matrix)
  loadings <- model$rotation[, 1:n_comp, drop = FALSE]
  loadings_list <- lapply(1:ncol(loadings), function(i) {
    setNames(as.list(loadings[, i]), rownames(loadings))
  })
  names(loadings_list) <- paste0("PC", 1:n_comp)

  list(
    loadings = loadings_list,
    sdev = as.list(model$sdev[1:n_comp]),
    variance_explained = as.list(var_explained[1:n_comp]),
    proportion_variance = as.list(prop_var[1:n_comp]),
    cumulative_variance = as.list(cumulative_var[1:n_comp]),
    n_components = n_comp,
    n_obs = nrow(X),
    n_vars = ncol(X),
    center = setNames(as.list(model$center), colnames(X)),
    scale = setNames(as.list(model$scale), colnames(X))
  )
}
