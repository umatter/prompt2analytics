#!/usr/bin/env Rscript
# Batch 2 R Benchmarks: toeplitz, line, cpgram, supsmu, constrOptim, ppr,
#                       se.contrast, model.tables
# Compares R implementations against p2a-core Rust
#
# Run with: Rscript benchmark_batch2.R

set.seed(42)

# Simple benchmark function using system.time
benchmark_fn <- function(expr, times = 100) {
  results <- numeric(times)
  for (i in 1:times) {
    results[i] <- system.time(eval(expr))[["elapsed"]]
  }
  median(results) * 1e6  # Convert to microseconds
}

cat("============================================================================\n")
cat("Batch 2 R Benchmarks: toeplitz, line, cpgram, supsmu, constrOptim, ppr,\n")
cat("                      se.contrast, model.tables\n")
cat("============================================================================\n\n")

# ============================================================================
# toeplitz benchmarks
# ============================================================================
cat("=== toeplitz R Benchmarks ===\n")

for (size in c(10, 50, 100, 500)) {
  x <- 1 / (1:size)

  cat(sprintf("\nn=%d:\n", size))
  med <- benchmark_fn(quote(toeplitz(x)), times = 50)
  cat(sprintf("  toeplitz (symmetric): %.2f us (median)\n", med))
}

# ============================================================================
# line benchmarks (Tukey's resistant line)
# ============================================================================
cat("\n=== line R Benchmarks ===\n")

for (n in c(20, 100, 500, 1000)) {
  x <- 1:n
  y <- 2 * x + 5 + rnorm(n, sd = 5)

  cat(sprintf("\nn=%d:\n", n))
  med <- benchmark_fn(quote(line(x, y)), times = 50)
  cat(sprintf("  line: %.2f us (median)\n", med))
}

# ============================================================================
# cpgram benchmarks (cumulative periodogram)
# ============================================================================
cat("\n=== cpgram R Benchmarks ===\n")
cat("(Note: cpgram is primarily a plotting function in R, timing the computation)\n")

for (n in c(64, 256, 1024, 4096)) {
  x <- rnorm(n)

  cat(sprintf("\nn=%d:\n", n))
  # cpgram doesn't return values easily, so we time spectrum which is the core computation
  med <- benchmark_fn(quote({
    spec <- spectrum(x, plot = FALSE)
    cumsum(spec$spec) / sum(spec$spec)
  }), times = 50)
  cat(sprintf("  cpgram (via spectrum): %.2f us (median)\n", med))
}

# ============================================================================
# supsmu benchmarks (Friedman's SuperSmoother)
# ============================================================================
cat("\n=== supsmu R Benchmarks ===\n")

for (n in c(50, 200, 1000)) {
  x <- (1:n) / n
  y <- sin(x * 2 * pi) + rnorm(n, sd = 0.25)

  cat(sprintf("\nn=%d:\n", n))

  # Auto span selection
  med <- benchmark_fn(quote(supsmu(x, y)), times = 50)
  cat(sprintf("  supsmu (auto span): %.2f us (median)\n", med))

  # Fixed span
  med <- benchmark_fn(quote(supsmu(x, y, span = 0.1)), times = 50)
  cat(sprintf("  supsmu (span=0.1): %.2f us (median)\n", med))
}

# ============================================================================
# constrOptim benchmarks (constrained optimization)
# ============================================================================
cat("\n=== constrOptim R Benchmarks ===\n")

# 2D quadratic: minimize (x-2)^2 + (y-3)^2 subject to x + y >= 1
f_2d <- function(x) (x[1] - 2)^2 + (x[2] - 3)^2
grad_2d <- function(x) c(2 * (x[1] - 2), 2 * (x[2] - 3))

cat("\n2D quadratic (x + y >= 1):\n")
ui <- matrix(c(1, 1), nrow = 1)
ci <- 1
med <- benchmark_fn(quote(constrOptim(c(0.5, 0.5), f_2d, grad_2d, ui, ci)), times = 50)
cat(sprintf("  constrOptim (Nelder-Mead w/grad): %.2f us (median)\n", med))

med <- benchmark_fn(quote(constrOptim(c(0.5, 0.5), f_2d, NULL, ui, ci)), times = 50)
cat(sprintf("  constrOptim (Nelder-Mead no grad): %.2f us (median)\n", med))

# 10D problem
cat("\n10D quadratic (x_i >= 0):\n")
f_10d <- function(x) sum((x - (0:9))^2)
ui_10d <- diag(10)
ci_10d <- rep(0, 10)
med <- benchmark_fn(quote(constrOptim(rep(5, 10), f_10d, NULL, ui_10d, ci_10d)), times = 50)
cat(sprintf("  constrOptim 10D: %.2f us (median)\n", med))

# ============================================================================
# ppr benchmarks (projection pursuit regression)
# ============================================================================
cat("\n=== ppr R Benchmarks ===\n")

for (config in list(c(100, 5), c(500, 10), c(1000, 5))) {
  n <- config[1]
  p <- config[2]

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  # y based on a projection
  alpha <- (1:p) / p
  proj <- X %*% alpha
  y <- sin(proj) + rnorm(n, sd = 0.1)

  cat(sprintf("\nn=%d, p=%d:\n", n, p))

  med <- benchmark_fn(quote(ppr(X, y, nterms = 1, max.terms = 1)), times = 20)
  cat(sprintf("  ppr (nterms=1): %.2f us (median)\n", med))

  med <- benchmark_fn(quote(ppr(X, y, nterms = 3, max.terms = 3)), times = 20)
  cat(sprintf("  ppr (nterms=3): %.2f us (median)\n", med))
}

# ============================================================================
# se.contrast benchmarks
# ============================================================================
cat("\n=== se.contrast R Benchmarks ===\n")

for (n_per_group in c(10, 50, 100)) {
  k <- 4  # 4 groups
  n <- n_per_group * k

  value <- numeric(n)
  group <- character(n)
  for (i in 1:k) {
    idx <- ((i-1) * n_per_group + 1):(i * n_per_group)
    value[idx] <- (i - 1) * 5 + rnorm(n_per_group, sd = 1)
    group[idx] <- paste0("G", i)
  }
  data <- data.frame(value = value, group = factor(group))

  cat(sprintf("\nk=%d groups, n=%d per group:\n", k, n_per_group))

  # Fit ANOVA
  fit <- aov(value ~ group, data = data)

  # Treatment contrasts
  contrasts <- contr.treatment(k)
  med <- benchmark_fn(quote(se.contrast(fit, contrasts)), times = 50)
  cat(sprintf("  se.contrast (treatment): %.2f us (median)\n", med))

  # Helmert contrasts
  contrasts <- contr.helmert(k)
  med <- benchmark_fn(quote(se.contrast(fit, contrasts)), times = 50)
  cat(sprintf("  se.contrast (helmert): %.2f us (median)\n", med))
}

# ============================================================================
# model.tables benchmarks
# ============================================================================
cat("\n=== model.tables R Benchmarks ===\n")

for (n_per_group in c(10, 50, 100)) {
  k <- 5  # 5 groups
  n <- n_per_group * k

  value <- numeric(n)
  group <- character(n)
  for (i in 1:k) {
    idx <- ((i-1) * n_per_group + 1):(i * n_per_group)
    value[idx] <- (i - 1) * 3 + rnorm(n_per_group, sd = 1)
    group[idx] <- paste0("G", i)
  }
  data <- data.frame(value = value, group = factor(group))
  fit <- aov(value ~ group, data = data)

  cat(sprintf("\nk=%d groups, n=%d per group:\n", k, n_per_group))

  med <- benchmark_fn(quote(model.tables(fit, "means", se = TRUE)), times = 50)
  cat(sprintf("  model.tables (means): %.2f us (median)\n", med))

  med <- benchmark_fn(quote(model.tables(fit, "effects", se = TRUE)), times = 50)
  cat(sprintf("  model.tables (effects): %.2f us (median)\n", med))
}

# Two-way model.tables
cat("\nTwo-way ANOVA model.tables:\n")
for (size in c(3, 5, 10)) {
  n_per_cell <- 5
  n <- size * size * n_per_cell

  value <- numeric(n)
  factorA <- character(n)
  factorB <- character(n)
  idx <- 1
  for (i in 1:size) {
    for (j in 1:size) {
      for (rep in 1:n_per_cell) {
        value[idx] <- (i + j) * 2 + rnorm(1)
        factorA[idx] <- paste0("A", i)
        factorB[idx] <- paste0("B", j)
        idx <- idx + 1
      }
    }
  }
  data <- data.frame(value = value, factorA = factor(factorA), factorB = factor(factorB))
  fit <- aov(value ~ factorA * factorB, data = data)

  cat(sprintf("\n%dx%d factorial:\n", size, size))
  med <- benchmark_fn(quote(model.tables(fit, "means")), times = 50)
  cat(sprintf("  model.tables (two-way): %.2f us (median)\n", med))
}

# ============================================================================
# Validation section
# ============================================================================
cat("\n\n============================================================================\n")
cat("Validation Results (for comparing with Rust output)\n")
cat("============================================================================\n")

# toeplitz validation
cat("\n--- toeplitz validation ---\n")
x <- c(1, 0.5, 0.25, 0.125)
mat <- toeplitz(x)
cat("toeplitz(c(1, 0.5, 0.25, 0.125)):\n")
print(mat)

# line validation
cat("\n--- line validation ---\n")
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.2, 15.9, 17.8, 20.1)
fit <- line(x, y)
cat(sprintf("line(x, y):\n  intercept: %.6f\n  slope: %.6f\n", fit$coefficients[1], fit$coefficients[2]))
cat(sprintf("  residuals: %s\n", paste(round(fit$residuals, 4), collapse = ", ")))

# supsmu validation
cat("\n--- supsmu validation ---\n")
x <- (1:20) / 20
y <- sin(x * 2 * pi) + c(0.1, -0.05, 0.08, -0.12, 0.03, -0.07, 0.11, -0.02, 0.06, -0.09,
                         0.04, -0.08, 0.07, -0.03, 0.09, -0.06, 0.02, -0.1, 0.05, -0.04)
fit <- supsmu(x, y, span = 0.2)
cat(sprintf("supsmu (first 5 y values): %s\n", paste(round(fit$y[1:5], 6), collapse = ", ")))

# constrOptim validation
cat("\n--- constrOptim validation ---\n")
f <- function(x) (x[1] - 2)^2 + (x[2] - 3)^2
grad <- function(x) c(2 * (x[1] - 2), 2 * (x[2] - 3))
ui <- matrix(c(1, 1), nrow = 1)
ci <- 1
result <- constrOptim(c(0.5, 0.5), f, grad, ui, ci)
cat(sprintf("constrOptim result:\n  par: (%.6f, %.6f)\n  value: %.6f\n  convergence: %d\n",
            result$par[1], result$par[2], result$value, result$convergence))

# ppr validation
cat("\n--- ppr validation ---\n")
set.seed(42)
X <- matrix(rnorm(100 * 3), nrow = 100, ncol = 3)
y <- sin(X[,1] * 2 + X[,2]) + rnorm(100, sd = 0.1)
fit <- ppr(X, y, nterms = 1)
cat(sprintf("ppr (nterms=1):\n  alpha: %s\n  spar: %.6f\n",
            paste(round(fit$alpha, 6), collapse = ", "), fit$spar))
cat(sprintf("  fitted[1:5]: %s\n", paste(round(fit$fitted.values[1:5], 6), collapse = ", ")))

# se.contrast validation
cat("\n--- se.contrast validation ---\n")
value <- c(5.1, 4.9, 5.2, 4.8, 5.0,  # Group A, mean ~5
           7.2, 6.8, 7.1, 7.0, 6.9,  # Group B, mean ~7
           9.0, 9.2, 8.9, 9.1, 9.0)  # Group C, mean ~9
group <- factor(rep(c("A", "B", "C"), each = 5))
data <- data.frame(value = value, group = group)
fit <- aov(value ~ group, data = data)
contrasts <- matrix(c(1, -1, 0, 0, 1, -1), nrow = 3)  # A vs B, B vs C
se <- se.contrast(fit, contrasts)
cat(sprintf("se.contrast result:\n  SE[1] (A vs B): %.6f\n  SE[2] (B vs C): %.6f\n", se[1], se[2]))

# model.tables validation
cat("\n--- model.tables validation ---\n")
fit <- aov(value ~ group, data = data)
mt <- model.tables(fit, "means", se = TRUE)
cat("model.tables (means):\n")
cat(sprintf("  grand mean: %.6f\n", mt$tables$`Grand mean`))
cat(sprintf("  group means: A=%.4f, B=%.4f, C=%.4f\n",
            mt$tables$group["A"], mt$tables$group["B"], mt$tables$group["C"]))
cat(sprintf("  SE: %.6f\n", mt$se$group))

mt_eff <- model.tables(fit, "effects", se = TRUE)
cat("model.tables (effects):\n")
cat(sprintf("  effects: A=%.4f, B=%.4f, C=%.4f\n",
            mt_eff$tables$group["A"], mt_eff$tables$group["B"], mt_eff$tables$group["C"]))

cat("\n============================================================================\n")
cat("Benchmark complete.\n")
cat("============================================================================\n")
