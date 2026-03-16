# Batch 3 R Benchmarks
# Compare: skmeans, fastcluster, dynamicTreeCut, mixtools, kprototypes

library(microbenchmark)

# Check and load required packages
packages <- c("skmeans", "fastcluster", "dynamicTreeCut", "mixtools", "clustMixType")
for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  library(pkg, character.only = TRUE)
}

# Generate cluster data
generate_cluster_data <- function(n, k, n_clusters) {
  set.seed(42)
  data <- matrix(0, nrow = n, ncol = k)
  for (i in 1:n) {
    cluster <- ((i - 1) %% n_clusters)
    center <- cluster * 5.0
    data[i, ] <- center + rnorm(k, 0, 0.5)
  }
  return(data)
}

# Generate univariate mixture data
generate_univariate_mixture <- function(n) {
  set.seed(42)
  data <- numeric(n)
  for (i in 1:n) {
    if (i %% 2 == 1) {
      data[i] <- rnorm(1, 0, 1)
    } else {
      data[i] <- rnorm(1, 10, 1)
    }
  }
  return(data)
}

cat("BATCH 3 PERFORMANCE BENCHMARKS (R)\n")
cat("===================================\n\n")

sizes <- c(100, 1000, 10000)
results <- data.frame()

# skmeans benchmarks
cat("=== skmeans (Spherical K-Means) ===\n")
for (n in sizes) {
  data <- generate_cluster_data(n, 10, 3)
  iters <- ifelse(n <= 500, 10, 5)

  timing <- microbenchmark(
    skmeans(data, k = 3, control = list(nruns = 5, verbose = FALSE)),
    times = iters
  )

  mean_ms <- mean(timing$time) / 1e6
  cat(sprintf("skmeans n=%d: %.3f ms\n", n, mean_ms))
  results <- rbind(results, data.frame(method = "skmeans", n = n, time_ms = mean_ms, lang = "R"))
}
cat("\n")

# fastcluster benchmarks
cat("=== fastcluster (Fast Hierarchical) ===\n")
for (n in sizes) {
  data <- generate_cluster_data(n, 5, 3)
  iters <- ifelse(n <= 500, 10, 3)

  timing <- microbenchmark(
    fastcluster::hclust(dist(data), method = "ward.D2"),
    times = iters
  )

  mean_ms <- mean(timing$time) / 1e6
  cat(sprintf("fastcluster n=%d: %.3f ms\n", n, mean_ms))
  results <- rbind(results, data.frame(method = "fastcluster", n = n, time_ms = mean_ms, lang = "R"))
}
cat("\n")

# dynamicTreeCut benchmarks
cat("=== dynamicTreeCut ===\n")
for (n in sizes) {
  data <- generate_cluster_data(n, 5, 3)
  iters <- ifelse(n <= 500, 10, 3)

  # Pre-compute hierarchical clustering
  hc <- fastcluster::hclust(dist(data), method = "ward.D2")

  timing <- microbenchmark(
    cutreeDynamic(hc, distM = as.matrix(dist(data)), deepSplit = 2, minClusterSize = 2),
    times = iters
  )

  mean_ms <- mean(timing$time) / 1e6
  cat(sprintf("dynamicTreeCut n=%d: %.3f ms\n", n, mean_ms))
  results <- rbind(results, data.frame(method = "dynamicTreeCut", n = n, time_ms = mean_ms, lang = "R"))
}
cat("\n")

# mixtools benchmarks (univariate)
cat("=== normalmixEM (Univariate Mixture) ===\n")
for (n in sizes) {
  data <- generate_univariate_mixture(n)
  iters <- 10

  timing <- microbenchmark(
    normalmixEM(data, k = 2, maxit = 100, epsilon = 1e-6, verb = FALSE),
    times = iters
  )

  mean_ms <- mean(timing$time) / 1e6
  cat(sprintf("normalmixEM n=%d: %.3f ms\n", n, mean_ms))
  results <- rbind(results, data.frame(method = "normalmixEM", n = n, time_ms = mean_ms, lang = "R"))
}
cat("\n")

# mixtools benchmarks (multivariate)
cat("=== mvnormalmixEM (Multivariate Mixture) ===\n")
for (n in sizes) {
  data <- generate_cluster_data(n, 5, 2)
  iters <- 10

  timing <- microbenchmark(
    mvnormalmixEM(data, k = 2, maxit = 100, epsilon = 1e-6, verb = FALSE),
    times = iters
  )

  mean_ms <- mean(timing$time) / 1e6
  cat(sprintf("mvnormalmixEM n=%d: %.3f ms\n", n, mean_ms))
  results <- rbind(results, data.frame(method = "mvnormalmixEM", n = n, time_ms = mean_ms, lang = "R"))
}
cat("\n")

# kprototypes benchmarks
cat("=== kprototypes (Mixed Data) ===\n")
for (n in sizes) {
  numeric_data <- generate_cluster_data(n, 5, 3)
  categorical_data <- data.frame(
    cat1 = factor((1:n - 1) %% 3),
    cat2 = factor((1:n - 1) %% 5)
  )
  mixed_data <- cbind(as.data.frame(numeric_data), categorical_data)
  iters <- ifelse(n <= 500, 10, 5)

  timing <- microbenchmark(
    kproto(mixed_data, k = 3, nstart = 5, verbose = FALSE),
    times = iters
  )

  mean_ms <- mean(timing$time) / 1e6
  cat(sprintf("kprototypes n=%d: %.3f ms\n", n, mean_ms))
  results <- rbind(results, data.frame(method = "kprototypes", n = n, time_ms = mean_ms, lang = "R"))
}

cat("\n=== Summary ===\n")
cat("All benchmarks completed. Times are in milliseconds per iteration.\n")

# Save results
write.csv(results, "r_batch3_benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to r_batch3_benchmark_results.csv\n")
