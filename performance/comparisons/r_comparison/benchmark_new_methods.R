#!/usr/bin/env Rscript
# Performance benchmark for new statistical methods
# Compares R implementation timing

library(lmtest)
library(sandwich)
library(nnet)
library(MASS)
library(microbenchmark)

cat("================================================================\n")
cat("R Performance Benchmark: New Statistical Methods\n")
cat("================================================================\n\n")

# Sample sizes to test
sample_sizes <- c(100, 1000, 10000)

# Number of benchmark iterations
n_iter <- 10

results <- data.frame()

# ==============================================================================
# 1. Breusch-Godfrey Test
# ==============================================================================
cat("1. Breusch-Godfrey Test (bgtest)\n")
cat("--------------------------------\n")

for (n in sample_sizes) {
  set.seed(123)
  x <- 1:n
  e <- numeric(n)
  e[1] <- rnorm(1)
  for(i in 2:n) e[i] <- 0.5 * e[i-1] + rnorm(1)
  y <- 2 + 0.5 * x + e

  model <- lm(y ~ x)

  timing <- microbenchmark(
    bgtest(model, order = 1),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6  # Convert to ms
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "bgtest",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 2. RESET Test
# ==============================================================================
cat("2. RESET Test (resettest)\n")
cat("-------------------------\n")

for (n in sample_sizes) {
  set.seed(456)
  x <- runif(n, 1, 10)
  y <- 2 + 0.5 * x + 0.1 * x^2 + rnorm(n, sd = 0.5)

  model <- lm(y ~ x)

  timing <- microbenchmark(
    resettest(model, power = 2:3),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "resettest",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 3. Wald Test
# ==============================================================================
cat("3. Wald Test (waldtest)\n")
cat("-----------------------\n")

for (n in sample_sizes) {
  set.seed(789)
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 1 + 2*x1 + 0.5*x2 + rnorm(n)

  full_model <- lm(y ~ x1 + x2)
  reduced_model <- lm(y ~ x1)

  timing <- microbenchmark(
    waldtest(reduced_model, full_model),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "waldtest",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 4. HAC Standard Errors (Newey-West)
# ==============================================================================
cat("4. HAC Standard Errors (vcovHAC)\n")
cat("--------------------------------\n")

for (n in sample_sizes) {
  set.seed(101)
  x <- rnorm(n)
  e <- numeric(n)
  e[1] <- rnorm(1)
  for(i in 2:n) e[i] <- 0.5 * e[i-1] + rnorm(1)
  y <- 1 + 2*x + e

  model <- lm(y ~ x)

  timing <- microbenchmark(
    vcovHAC(model),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "vcovHAC",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 5. Granger Causality Test
# ==============================================================================
cat("5. Granger Causality Test (grangertest)\n")
cat("---------------------------------------\n")

for (n in sample_sizes) {
  set.seed(202)
  x <- cumsum(rnorm(n))
  y <- numeric(n)
  y[1] <- rnorm(1)
  for(i in 2:n) y[i] <- 0.3 * x[i-1] + 0.5 * y[i-1] + rnorm(1)

  timing <- microbenchmark(
    grangertest(y ~ x, order = 2),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "grangertest",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 6. Multinomial Logit
# ==============================================================================
cat("6. Multinomial Logit (multinom)\n")
cat("-------------------------------\n")

for (n in sample_sizes) {
  set.seed(404)
  x <- rnorm(n)
  probs <- cbind(exp(0), exp(0.5 + 1*x), exp(1 + 2*x))
  probs <- probs / rowSums(probs)
  y <- apply(probs, 1, function(p) sample(c("A", "B", "C"), 1, prob = p))

  data <- data.frame(y = factor(y), x = x)

  timing <- microbenchmark(
    multinom(y ~ x, data = data, trace = FALSE, MaxNWts = 10000),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "multinom",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 7. Ordered Logit (polr)
# ==============================================================================
cat("7. Ordered Logit (polr)\n")
cat("-----------------------\n")

for (n in sample_sizes) {
  set.seed(505)
  x <- rnorm(n)
  latent <- 1.5 * x + rlogis(n)
  y <- cut(latent, breaks = c(-Inf, -1, 1, Inf), labels = c("Low", "Med", "High"))

  data <- data.frame(y = ordered(y, levels = c("Low", "Med", "High")), x = x)

  timing <- microbenchmark(
    polr(y ~ x, data = data, method = "logistic"),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "polr",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# 8. Negative Binomial (glm.nb)
# ==============================================================================
cat("8. Negative Binomial (glm.nb)\n")
cat("-----------------------------\n")

for (n in sample_sizes) {
  set.seed(606)
  x <- runif(n, 0, 3)
  mu <- exp(0.5 + 0.8 * x)
  y <- rnbinom(n, size = 2, mu = mu)

  data <- data.frame(y = y, x = x)

  timing <- microbenchmark(
    glm.nb(y ~ x, data = data),
    times = n_iter,
    unit = "ms"
  )

  mean_time <- mean(timing$time) / 1e6
  cat(sprintf("  n=%5d: %.3f ms\n", n, mean_time))

  results <- rbind(results, data.frame(
    method = "glm.nb",
    n = n,
    time_ms = mean_time
  ))
}
cat("\n")

# ==============================================================================
# Summary Table
# ==============================================================================
cat("================================================================\n")
cat("SUMMARY: R Performance (milliseconds)\n")
cat("================================================================\n")
cat(sprintf("%-15s %8s %8s %8s\n", "Method", "n=100", "n=1000", "n=10000"))
cat(sprintf("%-15s %8s %8s %8s\n", "------", "-----", "------", "-------"))

for (method in unique(results$method)) {
  row <- results[results$method == method, ]
  cat(sprintf("%-15s %8.2f %8.2f %8.2f\n",
              method,
              row$time_ms[row$n == 100],
              row$time_ms[row$n == 1000],
              row$time_ms[row$n == 10000]))
}
cat("================================================================\n")

# Save results to CSV
write.csv(results, "r_benchmark_results.csv", row.names = FALSE)
cat("\nResults saved to r_benchmark_results.csv\n")
