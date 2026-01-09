#!/usr/bin/env Rscript
# Machine Learning Benchmarks for Cross-Language Comparison
# Compares R base and cluster packages against p2a Rust implementation

library(microbenchmark)

# Ensure reproducibility
set.seed(42)

# Generate cluster data with same DGP as Rust benchmarks
generate_cluster_data <- function(n, k = 5, n_clusters = 3) {
  data <- matrix(0, nrow = n, ncol = k)

  for (i in 1:n) {
    cluster <- ((i - 1) %% n_clusters) + 1
    center <- (cluster - 1) * 3.0  # Cluster centers at 0, 3, 6

    for (j in 1:k) {
      data[i, j] <- center + runif(1, 0, 0.5)
    }
  }

  data
}

# Benchmark K-Means
benchmark_kmeans <- function() {
  results <- list()

  for (n in c(100, 1000, 5000)) {
    cat(sprintf("Benchmarking K-Means with n=%d\n", n))
    data <- generate_cluster_data(n, k = 5, n_clusters = 3)

    bm <- microbenchmark(
      kmeans(data, centers = 3, nstart = 5, iter.max = 100),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("kmeans_", n)]] <- summary(bm)
  }

  results
}

# Benchmark DBSCAN
benchmark_dbscan <- function() {
  # Check if dbscan package is available
  if (!require(dbscan, quietly = TRUE)) {
    cat("dbscan package not installed, skipping DBSCAN benchmarks\n")
    return(list())
  }

  results <- list()

  for (n in c(100, 500, 1000)) {
    cat(sprintf("Benchmarking DBSCAN with n=%d\n", n))
    data <- generate_cluster_data(n, k = 5, n_clusters = 3)

    bm <- microbenchmark(
      dbscan::dbscan(data, eps = 0.5, minPts = 5),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("dbscan_", n)]] <- summary(bm)
  }

  results
}

# Benchmark Hierarchical Clustering
benchmark_hierarchical <- function() {
  results <- list()

  for (n in c(50, 100, 200)) {
    cat(sprintf("Benchmarking Hierarchical with n=%d\n", n))
    data <- generate_cluster_data(n, k = 5, n_clusters = 3)

    bm <- microbenchmark(
      {
        d <- dist(data)
        hclust(d, method = "ward.D2")
      },
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("hierarchical_", n)]] <- summary(bm)
  }

  results
}

# Benchmark PCA
benchmark_pca <- function() {
  results <- list()

  for (n in c(100, 1000, 5000)) {
    cat(sprintf("Benchmarking PCA with n=%d\n", n))
    data <- generate_cluster_data(n, k = 10, n_clusters = 3)

    bm <- microbenchmark(
      prcomp(data, center = TRUE, scale. = FALSE),
      times = 50,
      unit = "microseconds"
    )

    results[[paste0("pca_", n)]] <- summary(bm)
  }

  results
}

# Run benchmarks
cat("=== K-Means Benchmarks ===\n")
kmeans_results <- benchmark_kmeans()

cat("\n=== DBSCAN Benchmarks ===\n")
dbscan_results <- benchmark_dbscan()

cat("\n=== Hierarchical Clustering Benchmarks ===\n")
hierarchical_results <- benchmark_hierarchical()

cat("\n=== PCA Benchmarks ===\n")
pca_results <- benchmark_pca()

# Save results
save_results <- function(results, filename) {
  if (length(results) == 0) {
    cat(sprintf("No results to save for %s\n", filename))
    return()
  }

  df <- do.call(rbind, lapply(names(results), function(name) {
    r <- results[[name]]
    data.frame(
      method = name,
      mean_us = r$mean,
      median_us = r$median,
      min_us = r$min,
      max_us = r$max,
      n_eval = r$neval
    )
  }))

  write.csv(df, filename, row.names = FALSE)
  cat(sprintf("Results saved to %s\n", filename))
}

# Create results directory if needed
dir.create("results", showWarnings = FALSE)

save_results(kmeans_results, "results/ml_kmeans.csv")
save_results(dbscan_results, "results/ml_dbscan.csv")
save_results(hierarchical_results, "results/ml_hierarchical.csv")
save_results(pca_results, "results/ml_pca.csv")

# Print summary
cat("\n=== Summary ===\n")
cat("K-Means:\n")
print(do.call(rbind, kmeans_results))
cat("\nDBSCAN:\n")
if (length(dbscan_results) > 0) print(do.call(rbind, dbscan_results))
cat("\nHierarchical:\n")
print(do.call(rbind, hierarchical_results))
cat("\nPCA:\n")
print(do.call(rbind, pca_results))
