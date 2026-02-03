#!/usr/bin/env Rscript
# Clustering Benchmarks - Comprehensive comparison for new methods
# Compares R implementations against p2a Rust implementation

library(microbenchmark)

# Install and load required packages
packages <- c("cluster", "mclust", "dbscan", "apcluster", "kernlab", "fpc", "clusterCrit")
for (pkg in packages) {
  if (!require(pkg, quietly = TRUE, character.only = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

set.seed(42)

# Helper for string concatenation
`%+%` <- function(a, b) paste0(a, b)

# Generate cluster data with clear separation
generate_cluster_data <- function(n, k = 5, n_clusters = 3, separation = 5.0) {
  data <- matrix(0, nrow = n, ncol = k)

  for (i in 1:n) {
    cluster <- ((i - 1) %% n_clusters) + 1
    center <- (cluster - 1) * separation

    for (j in 1:k) {
      data[i, j] <- center + rnorm(1, 0, 0.5)
    }
  }

  data
}

# Test sizes
sizes <- c(100, 500, 1000)

cat("=" %+% strrep("=", 60) %+% "\n")
cat("CLUSTERING BENCHMARKS - R vs Rust Comparison\n")
cat("=" %+% strrep("=", 60) %+% "\n\n")

results <- data.frame()

# =============================================================================
# 1. Silhouette
# =============================================================================
cat("1. SILHOUETTE COEFFICIENT\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)
  km <- kmeans(data, 3, nstart = 1)
  labels <- km$cluster
  d <- dist(data)

  bm <- microbenchmark(
    cluster::silhouette(labels, d),
    times = 20,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "silhouette",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 2. Calinski-Harabasz Index
# =============================================================================
cat("\n2. CALINSKI-HARABASZ INDEX\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)
  km <- kmeans(data, 3, nstart = 1)
  labels <- km$cluster

  bm <- microbenchmark(
    fpc::calinhara(data, labels),
    times = 20,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "calinski_harabasz",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 3. Davies-Bouldin Index
# =============================================================================
cat("\n3. DAVIES-BOULDIN INDEX\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)
  km <- kmeans(data, 3, nstart = 1)
  labels <- km$cluster

  bm <- microbenchmark(
    clusterCrit::intCriteria(data, as.integer(labels), "Davies_Bouldin"),
    times = 20,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "davies_bouldin",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 4. Dunn Index
# =============================================================================
cat("\n4. DUNN INDEX\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes[1:2]) {  # Skip n=1000 as Dunn is O(n^2)
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)
  km <- kmeans(data, 3, nstart = 1)
  labels <- km$cluster

  bm <- microbenchmark(
    clusterCrit::intCriteria(data, as.integer(labels), "Dunn"),
    times = 10,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "dunn_index",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 5. Gap Statistic
# =============================================================================
cat("\n5. GAP STATISTIC\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in c(100, 200)) {  # Smaller sizes due to bootstrap cost
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  bm <- microbenchmark(
    cluster::clusGap(data, FUN = kmeans, K.max = 5, B = 20, nstart = 1),
    times = 5,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "gap_statistic",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 6. K-Medoids (PAM)
# =============================================================================
cat("\n6. K-MEDOIDS (PAM)\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  bm <- microbenchmark(
    cluster::pam(data, k = 3),
    times = 10,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "kmedoids",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 7. Spectral Clustering
# =============================================================================
cat("\n7. SPECTRAL CLUSTERING\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in c(100, 200, 500)) {  # Smaller due to eigendecomposition cost
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  bm <- microbenchmark(
    kernlab::specc(data, centers = 3),
    times = 10,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "spectral_clustering",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 8. Affinity Propagation
# =============================================================================
cat("\n8. AFFINITY PROPAGATION\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in c(100, 200)) {  # Very expensive for larger n
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  # Compute similarity matrix
  sim <- negDistMat(data, r = 2)

  bm <- microbenchmark(
    apcluster::apcluster(sim, maxits = 100, convits = 10),
    times = 5,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "affinity_propagation",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 9. HDBSCAN
# =============================================================================
cat("\n9. HDBSCAN\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  bm <- microbenchmark(
    dbscan::hdbscan(data, minPts = 5),
    times = 20,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "hdbscan",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 10. OPTICS
# =============================================================================
cat("\n10. OPTICS\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  bm <- microbenchmark(
    dbscan::optics(data, minPts = 5),
    times = 20,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "optics",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 11. Gaussian Mixture Model
# =============================================================================
cat("\n11. GAUSSIAN MIXTURE MODEL\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  data <- generate_cluster_data(n, k = 5, n_clusters = 3)

  bm <- microbenchmark(
    mclust::Mclust(data, G = 3, modelNames = "VVV", verbose = FALSE),
    times = 10,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "gaussian_mixture",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 12. Rand Index / Adjusted Rand Index
# =============================================================================
cat("\n12. ADJUSTED RAND INDEX\n")
cat("-" %+% strrep("-", 40) %+% "\n")

for (n in sizes) {
  true_labels <- rep(1:3, length.out = n)
  pred_labels <- sample(1:3, n, replace = TRUE)

  bm <- microbenchmark(
    mclust::adjustedRandIndex(true_labels, pred_labels),
    times = 100,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "rand_index",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# 13. NMI (Normalized Mutual Information)
# =============================================================================
cat("\n13. NORMALIZED MUTUAL INFORMATION\n")
cat("-" %+% strrep("-", 40) %+% "\n")

# Using aricode package for NMI
if (!require(aricode, quietly = TRUE)) {
  install.packages("aricode", repos = "https://cloud.r-project.org")
  library(aricode)
}

for (n in sizes) {
  true_labels <- rep(1:3, length.out = n)
  pred_labels <- sample(1:3, n, replace = TRUE)

  bm <- microbenchmark(
    aricode::NMI(true_labels, pred_labels),
    times = 100,
    unit = "milliseconds"
  )

  cat(sprintf("  n=%4d: median=%.3f ms, mean=%.3f ms\n",
              n, median(bm$time)/1e6, mean(bm$time)/1e6))

  results <- rbind(results, data.frame(
    method = "nmi",
    n = n,
    median_ms = median(bm$time)/1e6,
    mean_ms = mean(bm$time)/1e6,
    language = "R"
  ))
}

# =============================================================================
# Save Results
# =============================================================================
cat("\n" %+% "=" %+% strrep("=", 60) %+% "\n")
cat("SAVING RESULTS\n")
cat("=" %+% strrep("=", 60) %+% "\n")

output_file <- "r_clustering_benchmark_results.csv"
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("Results saved to %s\n", output_file))

# Print summary table
cat("\n=== SUMMARY TABLE ===\n")
print(results)
